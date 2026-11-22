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
from fuse.skills.loader import Skill
from fuse.tasks.queue import PostgresTaskStore

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

        yield Services(pool, checkpointer, actions, google, drafts, memory, tokens, tasks, contacts)
