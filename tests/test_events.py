import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from fuse.db import open_pool
from fuse.proactive.events import InMemoryEventStore, PostgresEventStore

DB_URL = os.environ.get("TEST_DATABASE_URL")
NOW = datetime(2026, 11, 23, 10, 0, tzinfo=ZoneInfo("America/Chicago"))


@pytest.fixture(params=["memory", "postgres"])
async def events(request):
    if request.param == "memory":
        yield InMemoryEventStore()
        return
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    async with open_pool(DB_URL) as pool:
        store = PostgresEventStore(pool)
        await store.ensure_schema()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM events")
        yield store


async def test_same_email_twice_is_one_event(events):
    first = await events.add("email", "gmail:m1", {"subject": "Hi"})
    second = await events.add("email", "gmail:m1", {"subject": "Hi"})
    assert first is not None and first.payload == {"subject": "Hi"}
    assert second is None
    assert [e.id for e in await events.pending()] == [first.id]


async def test_decided_events_leave_the_pending_queue(events):
    a = await events.add("email", "gmail:a", {})
    b = await events.add("email", "gmail:b", {})
    await events.decide(a.id, "ignored", "newsletter", NOW)
    assert [e.id for e in await events.pending()] == [b.id]
    stored = await events.get(a.id)
    assert (stored.status, stored.reason) == ("ignored", "newsletter")


async def test_sent_since_counts_recent_alerts_only(events):
    for i, minutes_ago in enumerate([90, 30, 5]):
        e = await events.add("email", f"gmail:{i}", {})
        await events.decide(
            e.id, "notified", "from your manager", NOW - timedelta(minutes=minutes_ago)
        )
    held = await events.add("email", "gmail:held", {})
    await events.decide(held.id, "held", "quiet hours", NOW)
    assert await events.sent_since(NOW - timedelta(hours=1)) == 2


async def test_held_events_are_taken_once(events):
    e = await events.add("email", "gmail:night", {"subject": "Late"})
    await events.decide(e.id, "held", "quiet hours", NOW)
    assert [x.payload["subject"] for x in await events.take_held()] == ["Late"]
    assert await events.take_held() == []
    assert (await events.get(e.id)).status == "delivered"
