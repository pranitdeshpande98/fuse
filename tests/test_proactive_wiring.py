"""Runs the bot's real proactive wiring against the test database, with fake Google and no
network, then one scheduler tick. Catches mistakes in how the pieces are put together."""

import os

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from fuse.actions import PostgresActionStore
from fuse.channels.telegram_bot import build_application, build_runner, setup_proactive
from fuse.config import Settings
from fuse.db import open_pool
from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.store import PostgresMemoryStore
from tests.test_gmail_watch import FakeFeed

DB_URL = os.environ.get("TEST_DATABASE_URL")


class FakeGoogle(FakeFeed):
    def __init__(self):
        super().__init__(latest="100")
        self.calendar_checked = False

    async def upcoming_events(self, days_ahead):
        self.calendar_checked = True
        return []

    async def search_messages(self, query, max_results):
        return []

    async def read_thread(self, thread_id):
        return []


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_setup_proactive_registers_every_job_and_ticks():
    settings = Settings(
        _env_file=None,
        telegram_bot_token="123:fake",
        telegram_allowed_chat_id=42,
        llm_primary_api_key="not-used",  # no model is called during this tick
    )
    app = build_application(settings)
    google = FakeGoogle()
    async with open_pool(DB_URL) as pool:
        async with pool.connection() as conn:
            for table in ("sync_cursors", "job_runs"):
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        actions = PostgresActionStore(pool)
        await actions.ensure_schema()
        memory = PostgresMemoryStore(pool, HashingEmbedder())
        await memory.ensure_schema()
        app.bot_data["runner"] = build_runner(settings, checkpointer, actions, None, None, memory)

        await setup_proactive(app, settings, pool, google, memory, checkpointer)

        scheduler = app.bot_data["scheduler"]
        assert {j.name for j in scheduler.daily} == {"morning_brief", "memory_consolidation"}
        assert {j.name for j in scheduler.interval} == {
            "gmail_watch",
            "decide",
            "meeting_reminders",
        }
        for job in scheduler.daily:  # don't let a real brief/consolidation run in the tick
            job.catch_up = job.catch_up * 0
        await scheduler.tick()
        assert google.calendar_checked
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT value FROM sync_cursors WHERE name = 'gmail'")
            assert (await cur.fetchone())["value"] == "100"  # first poll set the cursor
