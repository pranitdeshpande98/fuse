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


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_setup_with_job_alerts_and_scheduled_skills():
    from pathlib import Path
    from types import SimpleNamespace

    from fuse.jobalerts.client import JobAlertsClient
    from fuse.jobalerts.service import ScanRunner
    from fuse.jobs.store import InMemoryJobStore
    from fuse.proactive.notify import RecordingNotifier
    from fuse.skills.loader import load_skills
    from fuse.tasks.queue import InMemoryTaskStore
    from tests.fake_job_alerts import FakeJobAlerts

    settings = Settings(
        _env_file=None,
        telegram_bot_token="123:fake",
        telegram_allowed_chat_id=42,
        llm_primary_api_key="not-used",
    )
    app = build_application(settings)
    fake = FakeJobAlerts()
    client = JobAlertsClient("http://127.0.0.1:5000", fake.transport())
    async with open_pool(DB_URL) as pool:
        async with pool.connection() as conn:
            await conn.execute("DROP TABLE IF EXISTS sync_cursors")
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        actions = PostgresActionStore(pool)
        await actions.ensure_schema()
        memory = PostgresMemoryStore(pool, HashingEmbedder())
        await memory.ensure_schema()
        app.bot_data["runner"] = build_runner(settings, checkpointer, actions, None, None, memory)
        app.bot_data["services"] = SimpleNamespace(
            tasks=InMemoryTaskStore(),
            skills=load_skills(Path(__file__).parent.parent / "skills"),
            jobs=InMemoryJobStore(),
            contacts=None,
            job_alerts=client,
        )
        app.bot_data["scan_runner"] = ScanRunner(client, RecordingNotifier())

        await setup_proactive(app, settings, pool, None, memory, checkpointer)

        scheduler = app.bot_data["scheduler"]
        daily = {j.name for j in scheduler.daily}
        assert {"skill:weekly-review", "skill:follow-up-nudges"} <= daily
        assert "job_alerts_matches" in {j.name for j in scheduler.interval}
        core = app.bot_data["core"]
        assert core.scan_runner is app.bot_data["scan_runner"]
        assert core.scan_runner.after is not None  # a scan's matches follow it
        assert core.scan_runner.before is not None  # bookmark set before the first scan
        assert "2 of 3 portals active" in (await core.command("t", "dashboard"))[0].text
