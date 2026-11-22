import asyncio
import os
from datetime import timedelta

import pytest

from fuse.db import open_pool
from fuse.tasks.queue import InMemoryTaskStore, PostgresTaskStore

DB_URL = os.environ.get("TEST_DATABASE_URL")
LEASE = timedelta(minutes=2)
EXPIRED = timedelta(seconds=-1)  # a lease that has already run out: a "crashed" worker


@pytest.fixture(params=["memory", "postgres"])
async def tasks(request):
    if request.param == "memory":
        yield InMemoryTaskStore()
        return
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    async with open_pool(DB_URL) as pool:
        store = PostgresTaskStore(pool)
        await store.ensure_schema()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM tasks")
        yield store


async def test_a_task_is_claimed_by_one_worker_only(tasks):
    task = await tasks.create("list recruiter emails")
    first = await tasks.claim("w1", LEASE)
    second = await tasks.claim("w2", LEASE)
    assert first.id == task.id and first.status == "running" and first.attempts == 1
    assert second is None
    assert first.thread_id == f"task:{task.id}"


async def test_two_workers_claiming_at_once_get_different_tasks(tasks):
    for i in range(5):
        await tasks.create(f"task {i}")
    claimed = await asyncio.gather(*(tasks.claim(f"w{i}", LEASE) for i in range(5)))
    assert len({t.id for t in claimed if t}) == 5


async def test_finish_records_the_result(tasks):
    task = await tasks.create("goal")
    await tasks.claim("w1", LEASE)
    assert await tasks.finish(task.id, "w1", "Here's your list")
    done = await tasks.get(task.id)
    assert (done.status, done.result) == ("done", "Here's your list")


async def test_a_crashed_workers_task_is_picked_up_by_another(tasks):
    task = await tasks.create("goal")
    await tasks.claim("w1", EXPIRED)  # w1 took it and then died
    again = await tasks.claim("w2", LEASE)
    assert again.id == task.id and again.attempts == 2
    # w1 wakes up and tries to write: it no longer owns the task
    assert not await tasks.finish(task.id, "w1", "stale result")
    assert await tasks.renew(task.id, "w1", LEASE) is None
    assert await tasks.finish(task.id, "w2", "fresh result")
    assert (await tasks.get(task.id)).result == "fresh result"


async def test_a_task_that_kills_every_worker_ends_up_failed(tasks):
    task = await tasks.create("goal", max_attempts=2)
    await tasks.claim("w1", EXPIRED)
    await tasks.claim("w2", EXPIRED)
    assert await tasks.claim("w3", LEASE) is None  # out of attempts
    assert await tasks.fail_abandoned() == 1
    assert (await tasks.get(task.id)).status == "failed"


async def test_retry_later_waits_before_the_next_claim(tasks):
    task = await tasks.create("goal")
    await tasks.claim("w1", LEASE)
    await tasks.retry_later(task.id, "w1", "rate limited", timedelta(minutes=5))
    assert await tasks.claim("w1", LEASE) is None  # not yet
    stored = await tasks.get(task.id)
    assert (stored.status, stored.error) == ("queued", "rate limited")


async def test_approval_round_trip(tasks):
    task = await tasks.create("goal")
    await tasks.claim("w1", LEASE)
    await tasks.wait_for_approval(task.id, "w1", {"action_key": "k1", "preview": "To: pat"})
    assert await tasks.claim("w1", LEASE) is None  # waiting for you, not for a worker

    assert not await tasks.submit_decision(task.id, "old-key", {"approved": True})
    assert await tasks.submit_decision(task.id, "k1", {"approved": True, "mode": "send"})
    assert not await tasks.submit_decision(task.id, "k1", {"approved": False})  # only once

    resumed = await tasks.claim("w2", LEASE)
    assert resumed.resume_with == {"approved": True, "mode": "send"}
    assert resumed.attempts == 1  # resuming isn't a new attempt
    again = await tasks.get(task.id)
    assert again.resume_with is None


async def test_cancel_a_queued_or_waiting_task_is_immediate(tasks):
    waiting = await tasks.create("waits for approval")
    await tasks.claim("w1", LEASE)
    await tasks.wait_for_approval(waiting.id, "w1", {"action_key": "k"})
    queued = await tasks.create("never started")
    assert await tasks.request_cancel(queued.id) == "cancelled"
    assert await tasks.request_cancel(waiting.id) == "cancelled"
    assert not await tasks.submit_decision(waiting.id, "k", {"approved": True})
    assert await tasks.request_cancel(999) is None


async def test_cancel_a_running_task_asks_its_worker(tasks):
    task = await tasks.create("goal")
    await tasks.claim("w1", LEASE)
    assert await tasks.request_cancel(task.id) == "running"
    assert await tasks.renew(task.id, "w1", LEASE) is True
    assert await tasks.cancelled(task.id, "w1")
    assert (await tasks.get(task.id)).status == "cancelled"
    assert await tasks.request_cancel(task.id) == "cancelled"  # already finished: no change
