"""Time for proactive jobs: a clock you can fake in tests, quiet hours, and a scheduler.

Why a "tick" instead of firing each job at its exact time: Fuse runs on a laptop that
sleeps. A job set for 7:30 would simply be missed if the lid was closed at 7:30. Instead,
the bot calls Scheduler.tick() every 30 seconds and each job asks "am I due?". A daily job
missed while asleep still runs when the laptop wakes up, within its catch-up window, and
the job log makes sure it runs at most once per day, even across restarts.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from fuse.proactive.jobs import JobLog

log = logging.getLogger("fuse.scheduler")

JobFn = Callable[[], Awaitable[None]]


def parse_hhmm(value: str) -> time:
    """'07:30' -> time(7, 30). Raises ValueError with a readable message otherwise."""
    try:
        hours, minutes = value.strip().split(":")
        return time(int(hours), int(minutes))
    except (ValueError, TypeError):
        raise ValueError(f"expected a time like 07:30, got {value!r}") from None


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    """The real time in the user's timezone."""

    def __init__(self, timezone: str) -> None:
        self._tz = ZoneInfo(timezone)

    def now(self) -> datetime:
        return datetime.now(self._tz)


class FixedClock:
    """A clock for tests: it only moves when you move it."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def set(self, now: datetime) -> None:
        self._now = now

    def advance(self, **delta: float) -> None:
        self._now += timedelta(**delta)


def in_quiet_hours(now: datetime, start: time, end: time) -> bool:
    """True between start and end. Handles windows that cross midnight (22:00 -> 07:00)."""
    t = now.time().replace(tzinfo=None)
    if start == end:
        return False
    if start < end:
        return start <= t < end
    return t >= start or t < end


@dataclass
class DailyJob:
    name: str
    at: time
    run: JobFn
    catch_up: timedelta = timedelta(hours=4)  # still run if missed by up to this long

    def is_due(self, now: datetime) -> bool:
        scheduled = datetime.combine(now.date(), self.at, tzinfo=now.tzinfo)
        return scheduled <= now < scheduled + self.catch_up


@dataclass
class IntervalJob:
    name: str
    every: timedelta
    run: JobFn
    last_run: datetime | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_due(self, now: datetime) -> bool:
        return self.last_run is None or now - self.last_run >= self.every


class Scheduler:
    def __init__(self, clock: Clock, job_log: JobLog) -> None:
        self.clock = clock
        self._log = job_log
        self.daily: list[DailyJob] = []
        self.interval: list[IntervalJob] = []

    def add_daily(self, name: str, at: time, run: JobFn, catch_up: timedelta | None = None) -> None:
        job = DailyJob(name, at, run) if catch_up is None else DailyJob(name, at, run, catch_up)
        self.daily.append(job)

    def add_interval(self, name: str, every: timedelta, run: JobFn) -> None:
        self.interval.append(IntervalJob(name, every, run))

    async def tick(self) -> None:
        """Run whatever is due. A failing job is logged and never stops the others."""
        now = self.clock.now()
        for job in self.daily:
            if job.is_due(now):
                # The run key is the date: one run per day, remembered in Postgres.
                await self._log.run_once(job.name, now.date().isoformat(), job.run)
        for job in self.interval:
            if job.is_due(now) and not job.lock.locked():
                async with job.lock:
                    job.last_run = now
                    try:
                        await job.run()
                    except Exception:
                        log.exception("job %s failed", job.name)
