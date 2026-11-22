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
from fuse.memory.embeddings import Embedder, FastEmbedder
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
    embedder: Embedder


# Any fixed number shared by every Fuse process: the key of the Postgres advisory lock that
# lets only one process create or update tables at a time.
SCHEMA_LOCK_KEY = 7_340_001


@asynccontextmanager
async def schema_lock(pool: AsyncConnectionPool) -> AsyncIterator[None]:
    """Run the table setup in one process at a time.

    The bot and the worker start together, and `CREATE TABLE IF NOT EXISTS` is not safe when
    two connections run it at the same moment: the second fails with a duplicate-key error on
    Postgres's own catalog. That crashed the bot on the first start after a new table was
    added. An advisory lock makes the second process wait until the first has finished,
    after which every IF NOT EXISTS is a no-op.
    """
    async with pool.connection() as conn:
        await conn.execute("SELECT pg_advisory_lock(%s)", (SCHEMA_LOCK_KEY,))
        try:
            yield
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (SCHEMA_LOCK_KEY,))


@asynccontextmanager
async def open_services(settings: Settings) -> AsyncIterator[Services]:
    log.info("Loading the embedding model (the first run downloads ~130 MB)")
    embedder = FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir)
    await embedder.embed(["warm up"])  # load it now, not during your first message
    skills = load_skills(settings.skills_dir)
    log.info("Skills: %s", ", ".join(s.name for s in skills) or "none found")

    async with open_pool(settings.database_url.get_secret_value()) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        actions = PostgresActionStore(pool)
        memory = PostgresMemoryStore(pool, embedder)
        tasks = PostgresTaskStore(pool)
        contacts = PostgresContactStore(pool)
        jobs = PostgresJobStore(pool)
        reminders = PostgresReminderStore(pool)
        usage = PostgresUsageStore(pool)

        tokens, google, drafts = None, None, None
        if settings.encryption_key.get_secret_value():
            tokens = TokenStore(pool, SecretBox(settings.encryption_key.get_secret_value()))
            google = GoogleApiReader(tokens)
            drafts = GmailDraftSink(tokens)
        else:
            log.warning("ENCRYPTION_KEY not set: Gmail and Calendar tools are disabled")

        async with schema_lock(pool):
            await checkpointer.setup()  # LangGraph's tables on first run; no-op afterwards
            stores = [actions, memory, tasks, contacts, jobs, reminders, usage]
            for store in [*stores, *([tokens] if tokens else [])]:
                await store.ensure_schema()

        set_usage_store(usage)  # every model call in this process is counted from here on
        job_alerts = JobAlertsClient(settings.job_alerts_url) if settings.job_alerts_url else None

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
            embedder=embedder,
        )
