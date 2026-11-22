import os
from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import fuse.usage as usage
from fuse.db import open_pool
from fuse.proactive.jobs import InMemoryJobLog
from fuse.proactive.notify import RecordingNotifier
from fuse.usage import (
    USAGE_CALLBACK,
    InMemoryUsageStore,
    PostgresUsageStore,
    QuotaWatcher,
    model_name,
    usage_lines,
)

DB_URL = os.environ.get("TEST_DATABASE_URL")


def reply(tokens: int, model: str = "openai/gpt-oss-120b") -> LLMResult:
    message = AIMessage(
        "hi",
        usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


@pytest.fixture
def store(monkeypatch):
    s = InMemoryUsageStore()
    monkeypatch.setattr(usage, "_store", s)
    return s


async def test_every_reply_is_counted_per_model(store):
    await USAGE_CALLBACK.on_llm_end(reply(1200))
    await USAGE_CALLBACK.on_llm_end(reply(800))
    await USAGE_CALLBACK.on_llm_end(reply(500, "openai/gpt-oss-20b"))
    assert await store.on(usage.today()) == {
        "openai/gpt-oss-120b": (2000, 2),
        "openai/gpt-oss-20b": (500, 1),
    }
    assert usage_lines(await store.on(usage.today()), 200_000)[0] == (
        "- openai/gpt-oss-120b: 2,000 tokens in 2 calls (1% of 200,000)"
    )
    assert model_name(LLMResult(generations=[[]], llm_output={"model_name": "m"})) == "m"


async def test_a_broken_store_never_breaks_a_reply(monkeypatch):
    class Broken:
        async def record(self, *args):
            raise RuntimeError("database down")

    monkeypatch.setattr(usage, "_store", Broken())
    await USAGE_CALLBACK.on_llm_end(reply(100))  # no exception


async def test_warning_once_a_day_per_model_at_80_percent(store):
    notifier, job_log = RecordingNotifier(), InMemoryJobLog()
    watcher = QuotaWatcher(store, notifier, job_log, quota=10_000)
    await store.record(usage.today(), "openai/gpt-oss-120b", 7_000)
    assert await watcher.check() == []
    await store.record(usage.today(), "openai/gpt-oss-120b", 1_500)
    assert await watcher.check() == ["openai/gpt-oss-120b"]
    assert await watcher.check() == []  # not again today
    assert "85% of its ~10,000" in notifier.texts[0]
    assert await QuotaWatcher(store, notifier, job_log, quota=0).check() == []  # turned off


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_postgres_usage_store():
    async with open_pool(DB_URL) as pool:
        s = PostgresUsageStore(pool)
        await s.ensure_schema()
        day = (datetime.now(UTC) - timedelta(days=400)).date()  # a day no other test uses
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM llm_usage WHERE day = %s", (day,))
        await s.record(day, "m", 100)
        await s.record(day, "m", 50)
        assert await s.on(day) == {"m": (150, 2)}
