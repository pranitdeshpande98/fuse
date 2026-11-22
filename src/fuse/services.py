"""Opens everything Fuse needs (database pool, checkpoints, actions log, Google, memory),
once, for whichever process is starting: the Telegram bot or a background worker."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from fuse.actions import PostgresActionStore
from fuse.config import Settings
from fuse.contacts import PostgresContactStore
from fuse.crypto import SecretBox
from fuse.db import open_pool
from fuse.google.auth import TokenStore
from fuse.google.drafts import GmailDraftSink
from fuse.google.reader import GoogleApiReader
from fuse.jobalerts.client import JobAlertsClient
from fuse.jobs.store import PostgresJobStore
from fuse.memory.embeddings import FastEmbedder
from fuse.memory.store import PostgresMemoryStore
from fuse.reminders.store import PostgresReminderStore
from fuse.skills.loader import Skill, load_skills
from fuse.tasks.queue import PostgresTaskStore
from fuse.usage import PostgresUsageStore, set_usage_store

log = logging.getLogger("fuse.services")


@dataclass
class Services:
    pool: AsyncConnectionPool
    checkpointer: AsyncPostgresSaver
    actions: PostgresActionStore
    google: GoogleApiReader | None
    drafts: GmailDraftSink | None
    memory: PostgresMemoryStore
    tokens: TokenStore | None
    tasks: PostgresTaskStore
    contacts: PostgresContactStore
    jobs: PostgresJobStore
    skills: list[Skill]
    job_alerts: JobAlertsClient | None
    reminders: PostgresReminderStore
    usage: PostgresUsageStore


@asynccontextmanager
async def open_services(settings: Settings) -> AsyncIterator[Services]:
    async with open_pool(settings.database_url.get_secret_value()) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()  # creates LangGraph's tables on first run; no-op afterwards
        actions = PostgresActionStore(pool)
        await actions.ensure_schema()

        tokens, google, drafts = None, None, None
        if settings.encryption_key.get_secret_value():
            tokens = TokenStore(pool, SecretBox(settings.encryption_key.get_secret_value()))
            await tokens.ensure_schema()
            google = GoogleApiReader(tokens)
            drafts = GmailDraftSink(tokens)
        else:
            log.warning("ENCRYPTION_KEY not set: Gmail and Calendar tools are disabled")

        log.info("Loading the embedding model (the first run downloads ~130 MB)")
        embedder = FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir)
        await embedder.embed(["warm up"])  # load it now, not during your first message
        memory = PostgresMemoryStore(pool, embedder)
        await memory.ensure_schema()

        tasks = PostgresTaskStore(pool)
        await tasks.ensure_schema()
        contacts = PostgresContactStore(pool)
        await contacts.ensure_schema()

        jobs = PostgresJobStore(pool)
        await jobs.ensure_schema()
        reminders = PostgresReminderStore(pool)
        await reminders.ensure_schema()
        skills = load_skills(settings.skills_dir)
        log.info("Skills: %s", ", ".join(s.name for s in skills) or "none found")
        job_alerts = JobAlertsClient(settings.job_alerts_url) if settings.job_alerts_url else None
        usage = PostgresUsageStore(pool)
        await usage.ensure_schema()
        set_usage_store(usage)  # every model call in this process is counted from here on

        yield Services(
            pool=pool,
            checkpointer=checkpointer,
            actions=actions,
            google=google,
            drafts=drafts,
            memory=memory,
            tokens=tokens,
            tasks=tasks,
            contacts=contacts,
            jobs=jobs,
            skills=skills,
            job_alerts=job_alerts,
            reminders=reminders,
            usage=usage,
        )
