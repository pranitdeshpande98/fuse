import os
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from fuse.db import open_pool
from fuse.proactive.jobs import InMemoryJobLog, PostgresJobLog
from fuse.proactive.schedule import FixedClock, Scheduler, in_quiet_hours, parse_hhmm

CHICAGO = ZoneInfo("America/Chicago")
DB_URL = os.environ.get("TEST_DATABASE_URL")


def at(hour: int, minute: int = 0, day: int = 23) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=CHICAGO)


def test_parse_hhmm():
    assert parse_hhmm("07:30") == time(7, 30)
    with pytest.raises(ValueError, match="07:30"):
        parse_hhmm("7.30am")


def test_quiet_hours_across_midnight():
    start, end = time(22), time(7)
    assert in_quiet_hours(at(23), start, end)
    assert in_quiet_hours(at(3), start, end)
    assert not in_quiet_hours(at(7), start, end)  # end is exclusive
    assert not in_quiet_hours(at(12), start, end)


def test_quiet_hours_same_day_and_disabled():
    assert in_quiet_hours(at(13), time(12), time(14))
    assert not in_quiet_hours(at(15), time(12), time(14))
    assert not in_quiet_hours(at(3), time(0), time(0))  # start == end: no quiet hours


def test_settings_reject_bad_times():
    from fuse.config import Settings

    with pytest.raises(ValueError, match="07:30"):
        Settings(_env_file=None, brief_time="half past seven")


@pytest.fixture(params=["memory", "postgres"])
async def job_log(request):
    if request.param == "memory":
        yield InMemoryJobLog()
        return
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    async with open_pool(DB_URL) as pool:
        log = PostgresJobLog(pool)
        await log.ensure_schema()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM job_runs")
        yield log


async def test_daily_job_runs_once_per_day(job_log):
    clock = FixedClock(at(7, 0))
    runs = []
    scheduler = Scheduler(clock, job_log)
    scheduler.add_daily("brief", time(7, 30), lambda: _record(runs))

    await scheduler.tick()
    assert runs == []  # not yet 7:30
    clock.set(at(7, 30))
    await scheduler.tick()
    clock.advance(minutes=1)
    await scheduler.tick()
    assert len(runs) == 1  # once, not every tick
    clock.set(at(7, 30, day=24))
    await scheduler.tick()
    assert len(runs) == 2  # and again the next day


async def test_missed_daily_job_catches_up_but_not_forever(job_log):
    runs = []
    clock = FixedClock(at(9, 15))  # laptop was asleep at 7:30, woke at 9:15
    scheduler = Scheduler(clock, job_log)
    scheduler.add_daily("brief", time(7, 30), lambda: _record(runs), catch_up=timedelta(hours=4))
    await scheduler.tick()
    assert len(runs) == 1

    late = Scheduler(FixedClock(at(18, 0, day=24)), job_log)
    late.add_daily("brief", time(7, 30), lambda: _record(runs), catch_up=timedelta(hours=4))
    await late.tick()
    assert len(runs) == 1  # a 6 pm "morning brief" is not useful


async def test_failed_daily_job_is_retried_on_the_next_tick(job_log):
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("LLM rate limited")

    scheduler = Scheduler(FixedClock(at(7, 31)), job_log)
    scheduler.add_daily("brief", time(7, 30), flaky)
    await scheduler.tick()
    await scheduler.tick()
    await scheduler.tick()
    assert len(attempts) == 2  # failed once, succeeded once, then stopped
    assert await job_log.status("brief", "2026-11-23") == "done"


async def test_interval_job_respects_its_interval_and_survives_errors():
    clock = FixedClock(at(10))
    calls = []

    async def poll():
        calls.append(clock.now())
        raise RuntimeError("network down")  # must not break the scheduler

    scheduler = Scheduler(clock, InMemoryJobLog())
    scheduler.add_interval("gmail", timedelta(minutes=2), poll)
    await scheduler.tick()
    clock.advance(seconds=30)
    await scheduler.tick()
    clock.advance(minutes=2)
    await scheduler.tick()
    assert len(calls) == 2


async def _record(runs: list) -> None:
    runs.append(1)
