import pytest

from fuse.jobalerts.client import JobAlertsClient, JobAlertsDown, ScanBusy, job_line
from tests.fake_job_alerts import FakeJobAlerts


def client(fake):
    return JobAlertsClient("http://127.0.0.1:5000", fake.transport())


async def test_status_lists_only_enabled_failing_portals():
    status = await client(FakeJobAlerts()).status()
    assert (status.active, status.total, status.matches) == (2, 3, 3)
    assert status.failing == [("Duke", "Workday 500")]  # MIT is disabled: not "failing"
    assert status.schedule == "17:30 America/Chicago" and status.email_configured


async def test_matches_and_job_line():
    jobs = await client(FakeJobAlerts()).matches()
    assert [j["job_key"] for j in jobs] == ["k3", "k2", "k1"]
    assert job_line(jobs[0]) == (
        "Carnegie Mellon University: Software Engineer, Research Computing (Pittsburgh, PA)"
    )
    assert job_line(jobs[2]) == "HHMI: Data Engineer"


async def test_scan_reports_new_jobs_and_the_email():
    fake = FakeJobAlerts()
    fake.next_new = [
        {
            "job_key": "k4",
            "company": "RAND",
            "title": "Python Developer",
            "first_seen": "2026-11-22T16:00:00+00:00",
        }
    ]
    result = await client(fake).scan()
    assert [j["job_key"] for j in result.new] == ["k4"]
    assert result.emailed and result.errors == ["Duke: Workday 500"]


async def test_clear_errors_when_down_or_busy():
    with pytest.raises(JobAlertsDown, match="start.bat"):
        await client(FakeJobAlerts(down=True)).status()
    with pytest.raises(ScanBusy, match="already running"):
        await client(FakeJobAlerts(busy=True)).scan()
