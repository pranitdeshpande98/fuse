from datetime import datetime
from zoneinfo import ZoneInfo

from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.store import InMemoryMemoryStore
from fuse.proactive.events import InMemoryEventStore
from fuse.proactive.meetings import MeetingReminders
from fuse.proactive.notify import RecordingNotifier
from fuse.proactive.schedule import FixedClock

CHICAGO = ZoneInfo("America/Chicago")
NOW = datetime(2026, 11, 23, 10, 0, tzinfo=CHICAGO)


def meeting(id, start, summary="Acme review", **extra):
    return {"id": id, "summary": summary, "start": {"dateTime": start}, **extra}


ACME = meeting(
    "e1",
    "2026-11-23T10:12:00-06:00",
    hangoutLink="https://meet.example/abc",
    attendees=[
        {"email": "me@example.com", "self": True, "responseStatus": "accepted"},
        {"email": "pat@acme.com", "displayName": "Pat Lee"},
    ],
)


class Reader:
    def __init__(self, events, messages=()):
        self.events, self.messages = events, list(messages)
        self.queries = []

    async def upcoming_events(self, days_ahead):
        return self.events

    async def search_messages(self, query, max_results):
        self.queries.append(query)
        return self.messages

    async def read_thread(self, thread_id):
        return []


def reminders(reader, clock=None, memory=None):
    notifier = RecordingNotifier()
    r = MeetingReminders(
        reader,
        InMemoryEventStore(),
        notifier,
        clock or FixedClock(NOW),
        15,
        memory,
        min_similarity=0.1,  # the hashing test embedder scores lower than the real model
    )
    return r, notifier


async def test_reminder_has_time_link_people_notes_and_recent_email():
    memory = InMemoryMemoryStore(HashingEmbedder())
    await memory.add("fact", "Pat Lee is the client contact for the Acme proposal")
    reader = Reader([ACME], messages=[{"from": "Pat Lee <pat@acme.com>", "subject": "Proposal v2"}])
    r, notifier = reminders(reader, memory=memory)
    assert await r.check() == 1
    (text,) = notifier.texts
    assert text.startswith("In 12 min: Acme review (10:12)")
    assert "https://meet.example/abc" in text
    assert "With: Pat Lee" in text and "me@example.com" not in text
    assert "client contact for the Acme proposal" in text
    assert "Proposal v2" in text
    assert reader.queries == ["from:pat@acme.com newer_than:30d"]


async def test_each_meeting_is_reminded_once():
    clock = FixedClock(NOW)
    r, notifier = reminders(Reader([ACME]), clock)
    await r.check()
    clock.advance(minutes=5)
    await r.check()
    assert len(notifier.texts) == 1


async def test_only_meetings_inside_the_window():
    later = meeting("e2", "2026-11-23T11:00:00-06:00", summary="Later")
    started = meeting("e3", "2026-11-23T09:55:00-06:00", summary="Already started")
    all_day = {"id": "e4", "summary": "Holiday", "start": {"date": "2026-11-23"}}
    r, notifier = reminders(Reader([later, started, all_day]))
    assert await r.check() == 0


async def test_declined_and_cancelled_meetings_are_skipped():
    declined = meeting(
        "e5",
        "2026-11-23T10:10:00-06:00",
        attendees=[{"email": "me@x", "self": True, "responseStatus": "declined"}],
    )
    cancelled = meeting("e6", "2026-11-23T10:10:00-06:00", status="cancelled")
    r, _ = reminders(Reader([declined, cancelled]))
    assert await r.check() == 0


async def test_rescheduled_meeting_gets_a_new_reminder():
    clock = FixedClock(NOW)
    reader = Reader([ACME])
    r, notifier = reminders(reader, clock)
    await r.check()
    reader.events = [meeting("e1", "2026-11-23T13:10:00-06:00")]  # moved to 13:10
    clock.set(datetime(2026, 11, 23, 13, 0, tzinfo=CHICAGO))
    await r.check()
    assert len(notifier.texts) == 2
