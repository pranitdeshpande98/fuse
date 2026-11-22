"""/status: one message that says whether every part of Fuse is healthy.

Each check is independent and short (a few seconds at most), and a failing check says what
to do rather than raising. This is the first thing to look at when something seems off,
and it gets more useful in the cloud, where you can't just look at the terminal.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fuse.google.auth import GoogleNotConnected, get_credentials
from fuse.jobalerts.client import JobAlertsClient, JobAlertsError
from fuse.proactive.jobs import JobLog
from fuse.proactive.store import CursorStore
from fuse.tasks.queue import FINISHED, TaskStore
from fuse.tasks.worker import WORKER_HEARTBEAT
from fuse.usage import today, usage_lines

log = logging.getLogger("fuse.status")

CHECK_TIMEOUT = 10.0
WORKER_STALE = timedelta(minutes=2)


@dataclass
class StatusDeps:
    tz: Any
    cursors: CursorStore | None = None
    job_log: JobLog | None = None
    tasks: TaskStore | None = None
    reminders: Any = None
    usage: Any = None
    tokens: Any = None  # Google token store
    job_alerts: JobAlertsClient | None = None
    quota: int = 0


def ago(then: datetime, now: datetime) -> str:
    seconds = int((now - then).total_seconds())
    if seconds < 90:
        return f"{seconds} s ago"
    if seconds < 5400:
        return f"{seconds // 60} min ago"
    if seconds < 172800:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} days ago"


async def _worker(deps: StatusDeps, now: datetime) -> str:
    if deps.cursors is None:
        return "Worker: unknown"
    beat = await deps.cursors.get(WORKER_HEARTBEAT)
    if not beat:
        return "Worker: never seen. Start Fuse with `uv run fuse` (bot + worker)."
    seen = datetime.fromisoformat(beat)
    if now - seen > WORKER_STALE:
        return f"Worker: NOT RUNNING (last seen {ago(seen, now)}). Background tasks won't run."
    return f"Worker: running (seen {ago(seen, now)})"


async def _google(deps: StatusDeps) -> str:
    if deps.tokens is None:
        return "Google: not set up (no ENCRYPTION_KEY)"
    try:
        await asyncio.wait_for(get_credentials(deps.tokens), CHECK_TIMEOUT)
    except GoogleNotConnected:
        return "Google: SIGNED OUT. Run: uv run python -m fuse.google.auth"
    except Exception as exc:
        return f"Google: couldn't check ({type(exc).__name__})"
    return "Google: signed in"


async def _job_alerts(deps: StatusDeps) -> str:
    if deps.job_alerts is None:
        return "Job Alerts: off"
    try:
        status = await asyncio.wait_for(deps.job_alerts.status(), CHECK_TIMEOUT)
    except (JobAlertsError, TimeoutError) as exc:
        return f"Job Alerts: {str(exc) or 'no answer'}"
    n = len(status.failing)
    failing = f", {n} portal{'s' if n != 1 else ''} failing" if n else ""
    return f"Job Alerts: running, {status.active} portals active{failing}"


async def _brief(deps: StatusDeps, now: datetime) -> str:
    if deps.job_log is None:
        return "Morning brief: unknown"
    last = await deps.job_log.last_done("morning_brief")
    if last is None:
        return "Morning brief: not sent yet"
    return f"Morning brief: last sent {last.astimezone(deps.tz):%a %d %b %H:%M}"


async def _tasks(deps: StatusDeps) -> str:
    if deps.tasks is None:
        return "Background tasks: off"
    active = [t for t in await deps.tasks.recent(50) if t.status not in FINISHED]
    if not active:
        return "Background tasks: none running"
    counts: dict[str, int] = {}
    for t in active:
        counts[t.status] = counts.get(t.status, 0) + 1
    return "Background tasks: " + ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in counts.items())


async def _reminders(deps: StatusDeps) -> str:
    if deps.reminders is None:
        return "Reminders: off"
    upcoming = await deps.reminders.upcoming()
    return f"Reminders: {len(upcoming)} upcoming"


async def status_report(deps: StatusDeps) -> str:
    now = datetime.now(UTC)

    async def safe(name: str, check) -> str:
        try:
            return await check
        except Exception:
            log.exception("status check %s failed", name)
            return f"{name}: couldn't check"

    lines = ["Fuse status", "Bot: running (you're talking to it)"]
    lines += await asyncio.gather(
        safe("Worker", _worker(deps, now)),
        safe("Google", _google(deps)),
        safe("Job Alerts", _job_alerts(deps)),
        safe("Morning brief", _brief(deps, now)),
        safe("Background tasks", _tasks(deps)),
        safe("Reminders", _reminders(deps)),
    )
    if deps.usage is not None:
        lines.append("Tokens today (UTC):")
        lines += usage_lines(await deps.usage.on(today()), deps.quota)
    return "\n".join(lines)
