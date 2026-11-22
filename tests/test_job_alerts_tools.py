from datetime import UTC, datetime, timedelta

from fuse.jobalerts.client import JobAlertsClient
from fuse.jobalerts.tools import make_job_alerts_tools
from tests.fake_job_alerts import FakeJobAlerts


def tools(fake, start_scan=None):
    client = JobAlertsClient("http://127.0.0.1:5000", fake.transport())
    return {t.name: t for t in make_job_alerts_tools(client, start_scan)}


def recent(days_ago: float, key: str, title: str) -> dict:
    seen = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    return {
        "job_key": key,
        "company": "RAND",
        "title": title,
        "url": f"https://x/{key}",
        "first_seen": seen,
    }


async def test_status_tool():
    out = await tools(FakeJobAlerts())["job_alerts_status"].ainvoke({})
    assert "2 of 3 portals active" in out and "Duke: Workday 500" in out
    assert "Dashboard: http://127.0.0.1:5000" in out


async def test_recent_matches_filters_by_days_and_keyword_and_wraps_text():
    fake = FakeJobAlerts(
        jobs=[
            recent(1, "a", "SAP Analyst"),
            recent(3, "b", "Data Engineer"),
            recent(40, "c", "SAP Architect"),
        ]
    )
    t = tools(fake)["recent_job_matches"]
    out = await t.ainvoke({"days": 7})
    assert out.startswith("2 matches:") and "<untrusted_content>" in out
    sap = await t.ainvoke({"days": 7, "keyword": "sap"})
    assert "SAP Analyst" in sap and "Data Engineer" not in sap and "Architect" not in sap
    assert "No matches" in await t.ainvoke({"days": 7, "keyword": "rust"})


async def test_tools_explain_when_job_alerts_is_off():
    out = await tools(FakeJobAlerts(down=True))["job_alerts_status"].ainvoke({})
    assert "isn't running" in out and "start.bat" in out


def test_scan_tool_only_where_scans_can_be_started():
    assert "scan_job_portals" not in tools(FakeJobAlerts())
    t = tools(FakeJobAlerts(), start_scan=lambda: "Scan started.")
    assert t["scan_job_portals"].invoke({}) == "Scan started."
