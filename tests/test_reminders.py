import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from fuse.db import open_pool
from fuse.proactive.notify import RecordingNotifier
from fuse.proactive.schedule import FixedClock
from fuse.reminders.service import (
    ReminderSender,
    handle_reminder_button,
    make_reminder_tools,
    parse_when,
    reminders_text,
)
from fuse.reminders.store import InMemoryReminderStore, PostgresReminderStore

CHICAGO = ZoneInfo("America/Chicago")
NOW = datetime(2026, 11, 22, 10, 30, tzinfo=CHICAGO)
DB_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(params=["memory", "postgres"])
async def store(request):
    if request.param == "memory":
        yield InMemoryReminderStore()
        return
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    async with open_pool(DB_URL) as pool:
        s = PostgresReminderStore(pool)
        await s.ensure_schema()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM reminders")
        yield s


def test_parse_when_is_local_time():
    assert parse_when("2026-11-23 09:00", CHICAGO) == datetime(2026, 11, 23, 9, 0, tzinfo=CHICAGO)
    assert parse_when("2026-11-23", CHICAGO).hour == 9  # a date alone means 9:00
    with pytest.raises(ValueError, match="2026-11-23 09:00"):
        parse_when("tomorrow morning", CHICAGO)


async def test_set_list_and_cancel(store):
    tools = {t.name: t for t in make_reminder_tools(store, FixedClock(NOW))}
    out = await tools["set_reminder"].ainvoke({"text": "Call Pat", "when": "2026-11-23 09:00"})
    assert out.startswith("Reminder #") and "Mon 23 Nov 09:00: Call Pat" in out
    assert "already passed" in await tools["set_reminder"].ainvoke(
        {"text": "x", "when": "2026-11-21 09:00"}
    )
    assert "Not set" in await tools["set_reminder"].ainvoke({"text": "x", "when": "soon"})
    listed = await tools["list_reminders"].ainvoke({})
    assert "Call Pat" in listed
    rid = (await store.upcoming())[0].id
    assert "Cancelled" in await tools["cancel_reminder"].ainvoke({"reminder_id": rid})
    assert "No upcoming reminders" in await reminders_text(store, FixedClock(NOW))


async def test_due_reminders_are_sent_once_with_buttons(store):
    clock = FixedClock(NOW)
    await store.add("Call Pat", NOW + timedelta(minutes=5))
    notifier = RecordingNotifier()
    sender = ReminderSender(store, notifier, clock)
    assert await sender.check() == 0
    clock.advance(minutes=6)
    assert await sender.check() == 1
    assert await sender.check() == 0  # not twice
    text, buttons = notifier.sent[0]
    assert text == "Reminder: Call Pat"
    assert [b.label for b in buttons] == ["Done", "Snooze 1 h", "Tomorrow 9:00"]


async def test_snooze_tomorrow_and_done(store):
    clock = FixedClock(NOW)
    r = await store.add("Call Pat", NOW)
    notifier = RecordingNotifier()
    await ReminderSender(store, notifier, clock).check()
    assert "11:30" in await handle_reminder_button(store, clock, f"rm:snooze:{r.id}")
    clock.advance(hours=1, minutes=1)
    await ReminderSender(store, notifier, clock).check()
    assert len(notifier.sent) == 2  # came back after the snooze
    assert "Mon 23 Nov 09:00" in await handle_reminder_button(store, clock, f"rm:tomorrow:{r.id}")
    assert await handle_reminder_button(store, clock, f"rm:done:{r.id}") == "Done."
    assert (await store.get(r.id)).status == "done"
    assert "can't find" in await handle_reminder_button(store, clock, "rm:done:999")


async def test_a_failed_send_is_retried():
    class Flaky(RecordingNotifier):
        fail = True

        async def send(self, text, buttons=None):
            if self.fail:
                self.fail = False
                raise RuntimeError("Telegram down")
            await super().send(text, buttons)

    store, notifier = InMemoryReminderStore(), Flaky()
    await store.add("Call Pat", NOW)
    sender = ReminderSender(store, notifier, FixedClock(NOW))
    assert await sender.check() == 0
    assert await sender.check() == 1
