from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fuse.jobalerts.client import JobAlertsClient
from fuse.proactive.jobs import InMemoryJobLog
from fuse.proactive.store import InMemoryCursorStore
from fuse.reminders.store import InMemoryReminderStore
from fuse.status import StatusDeps, ago, status_report
from fuse.tasks.queue import InMemoryTaskStore
from fuse.tasks.worker import WORKER_HEARTBEAT
from fuse.usage import InMemoryUsageStore, today
from tests.fake_job_alerts import FakeJobAlerts

TZ = ZoneInfo("America/Chicago")


def test_ago():
    now = datetime(2026, 11, 22, 12, tzinfo=UTC)
    assert ago(now - timedelta(seconds=5), now) == "5 s ago"
    assert ago(now - timedelta(minutes=12), now) == "12 min ago"
    assert ago(now - timedelta(days=3), now) == "3 days ago"


async def test_everything_healthy():
    cursors, job_log, usage = InMemoryCursorStore(), InMemoryJobLog(), InMemoryUsageStore()
    await cursors.set(WORKER_HEARTBEAT, datetime.now(UTC).isoformat())

    async def brief():
        return None

    await job_log.run_once("morning_brief", "2026-11-22", brief)
    tasks = InMemoryTaskStore()
    await tasks.create("goal")
    reminders = InMemoryReminderStore()
    await reminders.add("Call Pat", datetime.now(UTC) + timedelta(hours=1))
    await usage.record(today(), "openai/gpt-oss-120b", 50_000)
    fake = FakeJobAlerts()
    deps = StatusDeps(
        tz=TZ,
        cursors=cursors,
        job_log=job_log,
        tasks=tasks,
        reminders=reminders,
        usage=usage,
        job_alerts=JobAlertsClient("http://127.0.0.1:5000", fake.transport()),
        quota=200_000,
    )
    report = await status_report(deps)
    assert "Worker: running" in report
    assert "Google: not set up" in report
    assert "Job Alerts: running, 2 portals active, 1 portal failing" in report
    assert "Morning brief: last sent" in report
    assert "Background tasks: 1 queued" in report
    assert "Reminders: 1 upcoming" in report
    assert "openai/gpt-oss-120b: 50,000 tokens in 1 calls (25% of 200,000)" in report


async def test_problems_say_what_to_do():
    cursors = InMemoryCursorStore()
    await cursors.set(WORKER_HEARTBEAT, (datetime.now(UTC) - timedelta(minutes=30)).isoformat())
    deps = StatusDeps(
        tz=TZ,
        cursors=cursors,
        job_log=InMemoryJobLog(),
        job_alerts=JobAlertsClient("http://127.0.0.1:5000", FakeJobAlerts(down=True).transport()),
    )
    report = await status_report(deps)
    assert "Worker: NOT RUNNING (last seen 30 min ago)" in report
    assert "start.bat" in report
    assert "Morning brief: not sent yet" in report
    never = await status_report(StatusDeps(tz=TZ, cursors=InMemoryCursorStore()))
    assert "Worker: never seen" in never


async def test_worker_writes_a_heartbeat():
    from langgraph.checkpoint.memory import InMemorySaver

    from fuse.agent.graph import build_graph
    from fuse.agent.runner import AgentRunner
    from fuse.tasks.worker import Worker
    from fuse.tools.basic import make_basic_tools
    from tests.conftest import ScriptedModel

    beats = []

    async def heartbeat():
        beats.append(1)

    runner = AgentRunner(
        build_graph(ScriptedModel(script=[]), make_basic_tools("UTC"), InMemorySaver())
    )
    worker = Worker(InMemoryTaskStore(), runner, heartbeat=heartbeat, poll_seconds=0.01)
    import asyncio

    stop = asyncio.Event()
    loop_task = asyncio.create_task(worker.run_forever(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await loop_task
    assert beats == [1]  # once at start, then at most every 30 s
