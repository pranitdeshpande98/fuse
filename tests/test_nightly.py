from datetime import datetime
from zoneinfo import ZoneInfo

from fuse.memory.consolidate import ConsolidationReport
from fuse.proactive.alerts import EventHeldItems
from fuse.proactive.events import InMemoryEventStore
from fuse.proactive.nightly import NightlyConsolidation, morning_note
from fuse.proactive.schedule import FixedClock

NIGHT = datetime(2026, 11, 24, 3, 0, tzinfo=ZoneInfo("America/Chicago"))


def report(saved=(), review=()):
    r = ConsolidationReport()
    r.saved, r.needs_review = list(saved), list(review)
    return r


def test_nothing_learned_means_no_note():
    assert morning_note(report()) is None


async def test_what_was_learned_shows_up_in_the_morning_brief_once():
    events = InMemoryEventStore()

    async def fake_consolidate():
        return report(
            saved=["Plays badminton on Sundays"],
            review=[("Prefers morning meetings", "Prefers afternoon meetings")],
        )

    nightly = NightlyConsolidation(fake_consolidate, events, FixedClock(NIGHT))
    await nightly.run()
    await nightly.run()  # a retry on the same night doesn't add a second note
    (item,) = await EventHeldItems(events).take_held()
    assert "Learned overnight: Plays badminton on Sundays" in item
    assert "was: Prefers afternoon meetings" in item
    assert await EventHeldItems(events).take_held() == []
