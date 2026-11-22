"""Smoke test: run the bot's real startup hook and build the real worker, against the test
database, with a fake Telegram token and no network. This is what `uv run fuse` does first;
if any piece is wired wrong (a missing field, a wrong argument), it fails here, not on your
laptop."""

import os

import pytest

DB_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture
def quick_embedder(monkeypatch):
    import fuse.services as services_module
    from fuse.memory.embeddings import HashingEmbedder

    class QuickEmbedder(HashingEmbedder):  # the real model would download 130 MB
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(services_module, "FastEmbedder", QuickEmbedder)


def settings_for_test():
    from fuse.config import Settings

    return Settings(
        _env_file=None,
        database_url=DB_URL,
        telegram_bot_token="123:fake",
        telegram_allowed_chat_id=42,
        llm_primary_api_key="not-used",
        job_alerts_url="http://127.0.0.1:5000",
    )


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_bot_starts_up_with_everything_wired(quick_embedder):
    from fuse.channels.core import COMMAND_NAMES
    from fuse.channels.telegram_bot import build_application

    app = build_application(settings_for_test())
    await app.post_init(app)
    try:
        core = app.bot_data["core"]
        assert set(core.commands) == set(COMMAND_NAMES)
        for name in ("tasks", "contacts", "jobs", "reminders", "memory", "job_alerts", "events"):
            assert getattr(core, name) is not None, name
        assert len(core.skills) == 4  # the repo's starter skills
        tools = set(app.bot_data["runner"]._graph.nodes["tools"].bound.tools_by_name)
        assert {
            "start_background_task",
            "scan_job_portals",
            "set_reminder",
            "use_skill",
            "save_job_application",
            "recent_job_matches",
            "draft_email_reply",
            "web_search",
        } <= tools
        interval = {j.name for j in app.bot_data["scheduler"].interval}
        assert {"reminders", "job_alerts_matches"} <= interval
    finally:
        await app.post_shutdown(app)


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_worker_builds_with_everything_wired(quick_embedder):
    from fuse.services import open_services
    from fuse.tasks.worker import build_worker

    async with open_services(settings_for_test()) as services:
        worker = build_worker(settings_for_test(), services, services.tasks)
        tools = set(worker.runner._graph.nodes["tools"].bound.tools_by_name)
        assert {"report_progress", "set_reminder", "use_skill", "recent_job_matches"} <= tools
        assert "scan_job_portals" not in tools  # scans are started from the bot only
