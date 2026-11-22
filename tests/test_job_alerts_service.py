from datetime import datetime
from zoneinfo import ZoneInfo

from fuse.jobalerts.client import JobAlertsClient
from fuse.jobalerts.service import MatchWatcher, ScanRunner
from fuse.proactive.events import InMemoryEventStore
from fuse.proactive.notify import RecordingNotifier
from fuse.proactive.schedule import FixedClock
from fuse.proactive.store import InMemoryCursorStore
from tests.fake_job_alerts import FakeJobAlerts

NOW = datetime(2026, 11, 22, 17, 40, tzinfo=ZoneInfo("America/Chicago"))


def job(key: str, first_seen: str, title: str = "Software Engineer") -> dict:
    return {
        "job_key": key,
        "company": "RAND",
        "title": title,
        "location": "Remote",
        "url": f"https://rand.example/{key}",
        "first_seen": first_seen,
    }


def setup(fake):
    client = JobAlertsClient("http://127.0.0.1:5000", fake.transport())
    events, notifier, cursors = InMemoryEventStore(), RecordingNotifier(), InMemoryCursorStore()
    watcher = MatchWatcher(client, cursors, events, notifier, FixedClock(NOW))
    return client, watcher, events, notifier, cursors


async def test_first_check_starts_from_now_instead_of_sending_3000_old_jobs():
    fake = FakeJobAlerts()
    _, watcher, _, notifier, cursors = setup(fake)
    assert await watcher.check() == 0
    assert notifier.sent == []
    assert await cursors.get("job_alerts") == "2026-11-21T23:31:00+00:00"


async def test_new_matches_arrive_once_oldest_first_with_track_buttons():
    fake = FakeJobAlerts()
    _, watcher, events, notifier, _ = setup(fake)
    await watcher.check()
    fake.add_job(job("k4", "2026-11-22T23:31:00+00:00", "SAP Basis Admin"))
    fake.add_job(job("k5", "2026-11-22T23:32:00+00:00", "Data Engineer"))
    assert await watcher.check() == 2
    assert await watcher.check() == 0  # nothing new: no second message
    text, buttons = notifier.sent[0]
    assert text.startswith("2 new job matches from Job Alerts:")
    assert text.index("SAP Basis Admin") < text.index("Data Engineer")
    assert "https://rand.example/k4" in text
    assert [b.label for b in buttons] == ["Track 1", "Track 2"]
    assert buttons[0].data.startswith("jt:")
    assert [e.status for e in events.events.values()] == ["notified", "notified"]


async def test_a_long_list_is_cut_with_a_pointer_to_the_dashboard():
    fake = FakeJobAlerts()
    _, watcher, _, notifier, _ = setup(fake)
    await watcher.check()
    for i in range(12):
        fake.add_job(job(f"n{i}", f"2026-11-22T23:{i:02d}:30+00:00"))
    await watcher.check()
    text, buttons = notifier.sent[0]
    assert text.startswith("12 new job matches") and "(+4 more on the dashboard)" in text
    assert len(buttons) == 8


async def test_job_alerts_being_off_is_silent():
    fake = FakeJobAlerts(down=True)
    _, watcher, _, notifier, _ = setup(fake)
    assert await watcher.check() == 0 and notifier.sent == []


async def test_scan_reports_back_and_then_sends_the_new_matches():
    fake = FakeJobAlerts()
    client, watcher, _, notifier, _ = setup(fake)
    await watcher.check()
    fake.next_new = [job("k9", "2026-11-22T23:59:00+00:00", "Python Developer")]
    runner = ScanRunner(client, notifier, after=watcher.check)
    assert "Scanning" in runner.start()
    assert "already running" in runner.start()  # a second /scan while the first runs
    await runner.wait()
    summary, match_message = notifier.texts
    assert summary.startswith("Scan done: 2 portals, 1 new matches.")
    assert "emailed you" in summary and "1 portals failed" in summary
    assert "Python Developer" in match_message
    assert fake.scans == 1


async def test_scan_explains_when_job_alerts_is_off_or_busy():
    for fake, reason in [
        (FakeJobAlerts(down=True), "isn't running"),
        (FakeJobAlerts(busy=True), "already running"),
    ]:
        client = JobAlertsClient("http://127.0.0.1:5000", fake.transport())
        notifier = RecordingNotifier()
        runner = ScanRunner(client, notifier)
        runner.start()
        await runner.wait()
        assert notifier.texts[0].startswith("The scan didn't run:") and reason in notifier.texts[0]


async def test_the_very_first_scan_lists_only_its_own_new_jobs():
    """Fuse has never seen Job Alerts before: no bookmark yet. The first /scan must send
    exactly the jobs it found, not nothing, and not the 3 old ones."""
    fake = FakeJobAlerts()  # 3 old jobs already in Job Alerts
    client, watcher, _, notifier, _ = setup(fake)
    fake.next_new = [job("k9", "2026-11-22T23:59:00+00:00", "Python Developer")]
    runner = ScanRunner(client, notifier, after=watcher.check, before=watcher.check)
    runner.start()
    await runner.wait()
    summary, matches = notifier.texts
    assert matches.startswith("1 new job match from Job Alerts:")
    assert "Python Developer" in matches
    assert "Data Engineer" not in matches and "SAP Analyst" not in matches  # the old ones
