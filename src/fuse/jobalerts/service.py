"""Job Alerts on Telegram: start a scan from your phone, and hear about new matches.

Email keeps working exactly as before, because Job Alerts sends it itself. Fuse adds the
Telegram side:

- ScanRunner: /scan (or "scan now") starts a scan in the background and replies right away.
  A scan visits 100+ portals and takes minutes; you get a summary when it's done.
- MatchWatcher: every few minutes, and right after a scan, it asks Job Alerts for matches
  newer than the last one it told you about (a cursor, like the Gmail watcher). New ones
  arrive as one message with a Track button each. Every job is stored in the events table
  under its Job Alerts key, so you never hear about the same job twice, even after a restart.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fuse.jobalerts.client import JobAlertsClient, JobAlertsDown, JobAlertsError, job_line
from fuse.proactive.events import Event, EventStore
from fuse.proactive.notify import Button, Notifier
from fuse.proactive.schedule import Clock
from fuse.proactive.store import CursorStore

log = logging.getLogger("fuse.jobalerts")

CURSOR = "job_alerts"
MAX_IN_MESSAGE = 8
TRACK_BUTTON = "jt"
APPLIED_BUTTON = "ja"


def track_buttons(events: list[Event]) -> list[list[Button]]:
    """One row per job: Track it (interested), or mark it Applied right away."""
    return [
        [
            Button(f"Track {i}", f"{TRACK_BUTTON}:{e.id}"),
            Button(f"Applied {i}", f"{APPLIED_BUTTON}:{e.id}"),
        ]
        for i, e in enumerate(events, 1)
    ]


def matches_message(events: list[Event], total: int) -> str:
    lines = [f"{total} new job match{'es' if total != 1 else ''} from Job Alerts:"]
    for i, event in enumerate(events, 1):
        job = event.payload
        lines.append(f"{i}. {job_line(job)}")
        if job.get("url"):
            lines.append(f"   {job['url']}")
    if total > len(events):
        lines.append(f"(+{total - len(events)} more on the dashboard)")
    lines.append("Track adds one to your applications; Applied also reminds you to follow up.")
    return "\n".join(lines)


class MatchWatcher:
    def __init__(
        self,
        client: JobAlertsClient,
        cursors: CursorStore,
        events: EventStore,
        notifier: Notifier,
        clock: Clock,
        on_sent: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._client, self._cursors, self._events = client, cursors, events
        self._notifier, self._clock, self._on_sent = notifier, clock, on_sent
        self._lock = asyncio.Lock()

    async def check(self) -> int:
        """Send any matches you haven't heard about. Returns how many were new."""
        async with self._lock:  # a scan finishing and the timer can't both send
            try:
                jobs = await self._client.matches()
            except JobAlertsDown:
                return 0  # Job Alerts simply isn't running right now: not worth a message
            except JobAlertsError:
                log.exception("couldn't read Job Alerts matches")
                return 0
            return await self._handle(jobs)

    async def _handle(self, jobs: list[dict[str, Any]]) -> int:
        newest = max((j.get("first_seen", "") for j in jobs), default="")
        cursor = await self._cursors.get(CURSOR)
        if cursor is None:
            # First run: start from now. 3,000 old matches on Telegram would be noise.
            await self._cursors.set(CURSOR, newest)
            return 0
        fresh = sorted(
            (j for j in jobs if j.get("first_seen", "") > cursor and j.get("job_key")),
            key=lambda j: j["first_seen"],
        )
        stored = []
        for job in fresh:
            event = await self._events.add("job", f"job:{job['job_key']}", job)
            if event is not None:
                stored.append(event)
        if stored:
            shown = stored[:MAX_IN_MESSAGE]
            text = matches_message(shown, len(stored))
            await self._notifier.send(text, track_buttons(shown))
            for event in stored:
                await self._events.decide(event.id, "notified", "new job match", self._clock.now())
            if self._on_sent is not None:
                await self._on_sent(text)
        if newest > cursor:
            await self._cursors.set(CURSOR, newest)  # moved only after the message went out
        return len(stored)


class ScanRunner:
    def __init__(
        self,
        client: JobAlertsClient,
        notifier: Notifier,
        after: Callable[[], Awaitable[Any]] | None = None,
        before: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self._client, self._notifier, self.after = client, notifier, after
        # Run before scanning: the match watcher sets its bookmark on its first ever run,
        # so it must look *before* the scan, or the scan's own new jobs would land behind
        # the bookmark and never reach Telegram.
        self.before = before
        self._task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> str:
        if self.running:
            return "A scan is already running. I'll message you when it's done."
        self._task = asyncio.create_task(self._run())
        return (
            "Scanning your Job Alerts portals now. It takes a few minutes; I'll message you "
            "when it's done, and Job Alerts will email you the new matches."
        )

    async def wait(self) -> None:
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        try:
            if self.before is not None:
                await self.before()
            result = await self._client.scan()
        except JobAlertsError as exc:
            await self._notifier.send(f"The scan didn't run: {exc}")
            return
        except Exception:
            log.exception("scan failed")
            await self._notifier.send("The scan failed on my side. Check the terminal logs.")
            return
        lines = [f"Scan done: {result.scanned} portals, {len(result.new)} new matches."]
        if result.emailed:
            lines.append("Job Alerts emailed you the new ones.")
        elif result.new:
            lines.append(f"Email: {result.email_status}.")
        if result.suppressed:
            lines.append(f"{result.suppressed} filtered out by the Job Alerts smart filter.")
        if result.errors:
            lines.append(f"{len(result.errors)} portals failed, e.g. {result.errors[0][:100]}")
        await self._notifier.send("\n".join(lines))
        if self.after is not None:
            await self.after()  # the new matches, with Track buttons, right away
