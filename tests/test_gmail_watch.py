import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httplib2
import pytest
from googleapiclient.errors import HttpError

from fuse.db import open_pool
from fuse.google.auth import GoogleNotConnected
from fuse.google.reader import GoogleApiReader, HistoryExpired
from fuse.proactive.gmail_watch import GmailWatcher
from fuse.proactive.jobs import InMemoryJobLog
from fuse.proactive.notify import RecordingNotifier
from fuse.proactive.schedule import FixedClock
from fuse.proactive.store import InMemoryCursorStore, PostgresCursorStore

DB_URL = os.environ.get("TEST_DATABASE_URL")
NOW = datetime(2026, 11, 23, 10, 0, tzinfo=ZoneInfo("America/Chicago"))


def summary(message_id: str, labels=("INBOX", "UNREAD")) -> dict:
    return {"id": message_id, "thread_id": f"t-{message_id}", "labels": list(labels)}


class FakeFeed:
    def __init__(self, latest="100"):
        self.latest = latest
        self.added: dict[str, list[str]] = {}  # history id -> messages added after it
        self.summaries: dict[str, dict] = {}
        self.error: Exception | None = None

    def deliver(self, message: dict, new_history_id: str) -> None:
        self.added.setdefault(self.latest, []).append(message["id"])
        self.summaries[message["id"]] = message
        self.latest = new_history_id

    async def latest_history_id(self):
        if self.error:
            raise self.error
        return self.latest

    async def new_message_ids(self, since):
        if self.error:
            raise self.error
        ids = [i for h, batch in self.added.items() if int(h) >= int(since) for i in batch]
        return ids, self.latest

    async def message_summary(self, message_id):
        return self.summaries.get(message_id)


def make_watcher(feed, cursors=None):
    seen, notifier = [], RecordingNotifier()

    async def sink(message):
        seen.append(message["id"])

    cursors = cursors or InMemoryCursorStore()
    watcher = GmailWatcher(feed, cursors, sink, notifier, FixedClock(NOW), InMemoryJobLog())
    return watcher, seen, notifier, cursors


async def test_first_poll_starts_from_now_instead_of_alerting_on_the_whole_inbox():
    feed = FakeFeed(latest="100")
    watcher, seen, _, cursors = make_watcher(feed)
    assert await watcher.poll() == 0
    assert await cursors.get("gmail") == "100"


async def test_new_email_is_handed_on_once():
    feed = FakeFeed()
    watcher, seen, _, cursors = make_watcher(feed)
    await watcher.poll()
    feed.deliver(summary("m1"), "101")
    feed.deliver(summary("m2"), "102")
    assert await watcher.poll() == 2
    assert await watcher.poll() == 0  # nothing new
    assert seen == ["m1", "m2"]
    assert await cursors.get("gmail") == "102"


async def test_sent_drafts_and_spam_are_skipped():
    feed = FakeFeed()
    watcher, seen, _, _ = make_watcher(feed)
    await watcher.poll()
    feed.deliver(summary("mine", labels=("SENT", "INBOX")), "101")
    feed.deliver(summary("junk", labels=("SPAM",)), "102")
    feed.deliver(summary("real"), "103")
    await watcher.poll()
    assert seen == ["real"]


async def test_cursor_does_not_move_if_handing_on_fails():
    feed = FakeFeed()
    cursors = InMemoryCursorStore()
    calls = []

    async def failing_sink(message):
        calls.append(message["id"])
        raise RuntimeError("database down")

    watcher = GmailWatcher(
        feed, cursors, failing_sink, RecordingNotifier(), FixedClock(NOW), InMemoryJobLog()
    )
    await watcher.poll()
    feed.deliver(summary("m1"), "101")
    with pytest.raises(RuntimeError):
        await watcher.poll()
    assert await cursors.get("gmail") == "100"  # m1 will be retried, not lost


async def test_expired_history_restarts_from_now():
    feed = FakeFeed()
    watcher, _, _, cursors = make_watcher(feed)
    await watcher.poll()
    feed.latest = "500"
    feed.new_message_ids = _raise(HistoryExpired("100"))  # Gmail forgot history from 100
    await watcher.poll()
    assert await cursors.get("gmail") == "500"


async def test_expired_google_sign_in_is_reported_once_a_day():
    feed = FakeFeed()
    feed.error = GoogleNotConnected("expired")
    watcher, _, notifier, _ = make_watcher(feed)
    for _ in range(5):
        await watcher.poll()
    assert len(notifier.texts) == 1
    assert "uv run python -m fuse.google.auth" in notifier.texts[0]


@pytest.mark.parametrize("kind", ["memory", "postgres"])
async def test_cursor_store(kind):
    if kind == "memory":
        cursors = InMemoryCursorStore()
        await _check_cursors(cursors)
        return
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    async with open_pool(DB_URL) as pool:
        cursors = PostgresCursorStore(pool)
        await cursors.ensure_schema()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM sync_cursors")
        await _check_cursors(cursors)


async def _check_cursors(cursors):
    assert await cursors.get("gmail") is None
    await cursors.set("gmail", "1")
    await cursors.set("gmail", "2")
    assert await cursors.get("gmail") == "2"


def _raise(exc):
    async def fn(*_):
        raise exc

    return fn


# --- The real reader, against a fake Gmail service -------------------------------------


class FakeHistoryApi:
    """Mimics gmail.users().history().list(...).execute() with two pages."""

    def __init__(self, pages=None, status=None):
        self.pages = pages or []
        self.status = status
        self.calls = []

    def users(self):
        return self

    def history(self):
        return self

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return self

    def execute(self):
        if self.status:
            raise HttpError(httplib2.Response({"status": self.status}), b"gone")
        return self.pages[len(self.calls) - 1]


def reader_with(monkeypatch, service):
    async def fake_service(self, name, version):
        return service

    monkeypatch.setattr(GoogleApiReader, "_service", fake_service)
    return GoogleApiReader(tokens=None)


async def test_history_is_read_across_pages_without_duplicates(monkeypatch):
    api = FakeHistoryApi(
        pages=[
            {
                "history": [{"messagesAdded": [{"message": {"id": "a"}}]}],
                "historyId": "110",
                "nextPageToken": "p2",
            },
            {
                "history": [
                    {"messagesAdded": [{"message": {"id": "a"}}, {"message": {"id": "b"}}]}
                ],
                "historyId": "120",
            },
        ]
    )
    ids, latest = await reader_with(monkeypatch, api).new_message_ids("100")
    assert ids == ["a", "b"]
    assert latest == "120"
    assert api.calls[0]["labelId"] == "INBOX"
    assert api.calls[1]["pageToken"] == "p2"


async def test_gmail_404_means_history_expired(monkeypatch):
    with pytest.raises(HistoryExpired):
        await reader_with(monkeypatch, FakeHistoryApi(status=404)).new_message_ids("1")
