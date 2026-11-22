"""Opens the real services against the test database: the same code `uv run fuse` runs at
startup. (A wiring bug here once slipped past tests that built the services by hand.)"""

import os

import pytest

from fuse.config import Settings

DB_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_open_services_builds_everything_the_bot_and_worker_use(monkeypatch, tmp_path):
    import fuse.services as services_module
    from fuse.memory.embeddings import HashingEmbedder

    class QuickEmbedder(HashingEmbedder):  # the real model would download 130 MB
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(services_module, "FastEmbedder", QuickEmbedder)
    skills_dir = tmp_path / "skills" / "demo"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: demo\ndescription: d\n---\nDo it.\n")
    settings = Settings(
        _env_file=None,
        database_url=DB_URL,
        skills_dir=str(tmp_path / "skills"),
        job_alerts_url="http://127.0.0.1:5000",
    )
    async with services_module.open_services(settings) as services:
        assert [s.name for s in services.skills] == ["demo"]
        assert services.job_alerts is not None
        for name in ("tasks", "contacts", "jobs", "reminders", "memory", "actions"):
            assert getattr(services, name) is not None, name
        assert services.google is None  # no ENCRYPTION_KEY in this test

        # And the agents both processes build from them
        from fuse.agent.factory import build_runner

        build_runner(
            settings.model_copy(update={"llm_primary_api_key": "not-used"}),
            services.checkpointer,
            services.actions,
            services.google,
            services.drafts,
            services.memory,
            contacts=services.contacts,
            skills=services.skills,
            jobs=services.jobs,
            job_alerts=services.job_alerts,
            reminders=services.reminders,
        )


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_bot_and_worker_starting_together_on_fresh_tables(monkeypatch, tmp_path):
    """The bot and the worker set up the same new tables at the same moment. Without the
    schema lock, one of them crashed with a duplicate-key error on Postgres's catalog."""
    import asyncio

    import fuse.services as services_module
    from fuse.db import open_pool
    from fuse.memory.embeddings import HashingEmbedder

    class QuickEmbedder(HashingEmbedder):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(services_module, "FastEmbedder", QuickEmbedder)
    async with open_pool(DB_URL) as pool, pool.connection() as conn:
        for table in ("job_applications", "reminders", "llm_usage", "contacts"):
            await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    settings = Settings(_env_file=None, database_url=DB_URL, skills_dir=str(tmp_path))

    async def start_one():
        async with services_module.open_services(settings) as services:
            return services.jobs is not None

    for _ in range(3):  # a few rounds, to give a race every chance to show up
        assert await asyncio.gather(start_one(), start_one(), start_one()) == [True] * 3
