"""A task queue in Postgres: no Redis, no Celery.

A background task is a row in `tasks`. Workers take the next one with

    SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1

which locks the row it picks and *skips* rows other workers have locked, so two workers
never take the same task and never wait for each other.

Leases handle crashes. A worker that takes a task holds it until `lease_expires_at` and
keeps renewing while it works. If the worker dies, the lease runs out and another worker
picks the task up again. Every write a worker makes checks `lease_owner`, so a worker that
lost its lease (it froze, then woke up) can't overwrite the new owner's work.

Life of a task:

    queued --> running --> done
       ^          |------> failed       (out of attempts, or a permanent error)
       |          |------> cancelled    (/cancel)
       |          '------> waiting_approval --(you tap a button)--> queued (resumes)
       '------- retry later (rate limit) <---'
"""

import itertools
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

TaskStatus = Literal["queued", "running", "waiting_approval", "done", "failed", "cancelled"]
FINISHED = ("done", "failed", "cancelled")

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id                BIGSERIAL PRIMARY KEY,
        goal              TEXT        NOT NULL,
        status            TEXT        NOT NULL DEFAULT 'queued' CHECK (status IN
                              ('queued', 'running', 'waiting_approval',
                               'done', 'failed', 'cancelled')),
        attempts          INTEGER     NOT NULL DEFAULT 0,
        max_attempts      INTEGER     NOT NULL DEFAULT 3,
        run_after         TIMESTAMPTZ NOT NULL DEFAULT now(),
        lease_owner       TEXT,
        lease_expires_at  TIMESTAMPTZ,
        cancel_requested  BOOLEAN     NOT NULL DEFAULT false,
        pending           JSONB,        -- the approval the task is waiting for
        decision          JSONB,        -- your answer, for the worker to resume with
        result            TEXT,
        error             TEXT,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at       TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS tasks_ready_idx ON tasks (run_after) WHERE status = 'queued'",
]


@dataclass
class Task:
    id: int
    goal: str
    status: TaskStatus = "queued"
    attempts: int = 0
    max_attempts: int = 3
    cancel_requested: bool = False
    pending: dict[str, Any] | None = None
    resume_with: dict[str, Any] | None = None  # set by claim() when resuming an approval
    result: str | None = None
    error: str | None = None
    created_at: datetime | None = None

    @property
    def thread_id(self) -> str:
        """Each task has its own conversation, separate from your chat."""
        return f"task:{self.id}"


class TaskStore(Protocol):
    async def create(self, goal: str, max_attempts: int = 3) -> Task: ...
    async def get(self, task_id: int) -> Task | None: ...
    async def recent(self, limit: int = 10) -> list[Task]: ...
    async def claim(self, worker: str, lease: timedelta) -> Task | None: ...
    async def renew(self, task_id: int, worker: str, lease: timedelta) -> bool | None:
        """Extend the lease. Returns whether cancellation was requested, or None if this
        worker no longer owns the task."""
        ...

    async def finish(self, task_id: int, worker: str, result: str) -> bool: ...
    async def fail(self, task_id: int, worker: str, error: str) -> bool: ...
    async def retry_later(
        self, task_id: int, worker: str, error: str, delay: timedelta
    ) -> bool: ...
    async def cancelled(self, task_id: int, worker: str) -> bool: ...
    async def wait_for_approval(
        self, task_id: int, worker: str, pending: dict[str, Any]
    ) -> bool: ...
    async def submit_decision(
        self, task_id: int, action_key: str, decision: dict[str, Any]
    ) -> bool:
        """Your Approve/Reject/Send for a waiting task. Only accepted for the approval it is
        actually waiting for, so an old button does nothing."""
        ...

    async def request_cancel(self, task_id: int) -> TaskStatus | None: ...
    async def fail_abandoned(self) -> int:
        """Tasks whose worker died on every attempt: mark them failed instead of retrying."""
        ...


COLUMNS = (
    "id, goal, status, attempts, max_attempts, cancel_requested, pending, result, error, created_at"
)


def _task(row: dict[str, Any]) -> Task:
    return Task(
        id=row["id"],
        goal=row["goal"],
        status=row["status"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        cancel_requested=row["cancel_requested"],
        pending=row["pending"],
        resume_with=row.get("resume_with"),
        result=row["result"],
        error=row["error"],
        created_at=row["created_at"],
    )


class PostgresTaskStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def _one(self, sql: str, params: tuple) -> dict[str, Any] | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchone()

    async def create(self, goal: str, max_attempts: int = 3) -> Task:
        row = await self._one(
            f"INSERT INTO tasks (goal, max_attempts) VALUES (%s, %s) RETURNING {COLUMNS}",
            (goal, max_attempts),
        )
        assert row is not None
        return _task(row)

    async def get(self, task_id: int) -> Task | None:
        row = await self._one(f"SELECT {COLUMNS} FROM tasks WHERE id = %s", (task_id,))
        return _task(row) if row else None

    async def recent(self, limit: int = 10) -> list[Task]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {COLUMNS} FROM tasks ORDER BY id DESC LIMIT %s", (limit,)
            )
            return [_task(r) for r in await cur.fetchall()]

    async def claim(self, worker: str, lease: timedelta) -> Task | None:
        row = await self._one(
            f"""
            WITH picked AS (
                SELECT id, decision FROM tasks
                WHERE (status = 'queued' AND run_after <= now())
                   OR (status = 'running' AND lease_expires_at < now()
                       AND attempts < max_attempts)
                ORDER BY id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE tasks t SET
                status = 'running',
                lease_owner = %s,
                lease_expires_at = now() + %s,
                attempts = t.attempts + CASE WHEN picked.decision IS NULL THEN 1 ELSE 0 END,
                decision = NULL,
                updated_at = now()
            FROM picked WHERE t.id = picked.id
            RETURNING {", ".join("t." + c.strip() for c in COLUMNS.split(","))},
                      picked.decision AS resume_with
            """,
            (worker, lease),
        )
        return _task(row) if row else None

    async def renew(self, task_id: int, worker: str, lease: timedelta) -> bool | None:
        row = await self._one(
            """
            UPDATE tasks SET lease_expires_at = now() + %s, updated_at = now()
            WHERE id = %s AND lease_owner = %s AND status = 'running'
            RETURNING cancel_requested
            """,
            (lease, task_id, worker),
        )
        return row["cancel_requested"] if row else None

    async def _end(self, task_id: int, worker: str, sets: str, params: tuple) -> bool:
        row = await self._one(
            f"""
            UPDATE tasks SET {sets}, lease_owner = NULL, lease_expires_at = NULL,
                             updated_at = now()
            WHERE id = %s AND lease_owner = %s AND status = 'running'
            RETURNING id
            """,
            (*params, task_id, worker),
        )
        return row is not None

    async def finish(self, task_id: int, worker: str, result: str) -> bool:
        return await self._end(
            task_id, worker, "status = 'done', result = %s, finished_at = now()", (result,)
        )

    async def fail(self, task_id: int, worker: str, error: str) -> bool:
        return await self._end(
            task_id, worker, "status = 'failed', error = %s, finished_at = now()", (error,)
        )

    async def retry_later(self, task_id: int, worker: str, error: str, delay: timedelta) -> bool:
        return await self._end(
            task_id,
            worker,
            "status = 'queued', error = %s, run_after = now() + %s",
            (error, delay),
        )

    async def cancelled(self, task_id: int, worker: str) -> bool:
        return await self._end(task_id, worker, "status = 'cancelled', finished_at = now()", ())

    async def wait_for_approval(self, task_id: int, worker: str, pending: dict[str, Any]) -> bool:
        return await self._end(
            task_id, worker, "status = 'waiting_approval', pending = %s", (Jsonb(pending),)
        )

    async def submit_decision(
        self, task_id: int, action_key: str, decision: dict[str, Any]
    ) -> bool:
        row = await self._one(
            """
            UPDATE tasks SET status = 'queued', decision = %s, pending = NULL,
                             run_after = now(), updated_at = now()
            WHERE id = %s AND status = 'waiting_approval'
              AND pending ->> 'action_key' = %s
            RETURNING id
            """,
            (Jsonb(decision), task_id, action_key),
        )
        return row is not None

    async def request_cancel(self, task_id: int) -> TaskStatus | None:
        # Not running: cancel straight away. Running: ask the worker to stop.
        row = await self._one(
            """
            UPDATE tasks SET
                status = CASE WHEN status IN ('queued', 'waiting_approval')
                              THEN 'cancelled' ELSE status END,
                finished_at = CASE WHEN status IN ('queued', 'waiting_approval')
                                   THEN now() ELSE finished_at END,
                cancel_requested = (status = 'running'),
                pending = NULL,
                updated_at = now()
            WHERE id = %s AND status NOT IN ('done', 'failed', 'cancelled')
            RETURNING status
            """,
            (task_id,),
        )
        if row:
            return row["status"]
        task = await self.get(task_id)
        return task.status if task else None

    async def fail_abandoned(self) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE tasks SET status = 'failed', finished_at = now(), updated_at = now(),
                    error = 'The worker stopped while running this task, on every attempt.',
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE status = 'running' AND lease_expires_at < now()
                  AND attempts >= max_attempts
                """
            )
            return cur.rowcount


@dataclass
class _Row:
    task: Task
    run_after: datetime
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    decision: dict[str, Any] | None = None


@dataclass
class InMemoryTaskStore:
    """Same rules, in memory. `now` can be replaced to test leases without waiting."""

    rows: dict[int, _Row] = field(default_factory=dict)
    now: Any = field(default=lambda: datetime.now(UTC))
    _ids: Any = field(default_factory=lambda: itertools.count(1))

    def _owned(self, task_id: int, worker: str) -> _Row | None:
        row = self.rows.get(task_id)
        if row and row.lease_owner == worker and row.task.status == "running":
            return row
        return None

    async def create(self, goal: str, max_attempts: int = 3) -> Task:
        task = Task(next(self._ids), goal, max_attempts=max_attempts, created_at=self.now())
        self.rows[task.id] = _Row(task, run_after=self.now())
        return replace(task)

    async def get(self, task_id: int) -> Task | None:
        row = self.rows.get(task_id)
        return replace(row.task) if row else None

    async def recent(self, limit: int = 10) -> list[Task]:
        return [replace(r.task) for r in sorted(self.rows.values(), key=lambda r: -r.task.id)][
            :limit
        ]

    async def claim(self, worker: str, lease: timedelta) -> Task | None:
        now = self.now()
        for row in sorted(self.rows.values(), key=lambda r: r.task.id):
            t = row.task
            ready = t.status == "queued" and row.run_after <= now
            abandoned = (
                t.status == "running"
                and row.lease_expires_at is not None
                and row.lease_expires_at < now
                and t.attempts < t.max_attempts
            )
            if ready or abandoned:
                resume_with, row.decision = row.decision, None
                if resume_with is None:
                    t.attempts += 1
                t.status = "running"
                row.lease_owner, row.lease_expires_at = worker, now + lease
                return replace(t, resume_with=resume_with)
        return None

    async def renew(self, task_id: int, worker: str, lease: timedelta) -> bool | None:
        row = self._owned(task_id, worker)
        if row is None:
            return None
        row.lease_expires_at = self.now() + lease
        return row.task.cancel_requested

    def _end(self, row: _Row | None, **changes: Any) -> bool:
        if row is None:
            return False
        for key, value in changes.items():
            setattr(row.task, key, value)
        row.lease_owner = row.lease_expires_at = None
        return True

    async def finish(self, task_id: int, worker: str, result: str) -> bool:
        return self._end(self._owned(task_id, worker), status="done", result=result)

    async def fail(self, task_id: int, worker: str, error: str) -> bool:
        return self._end(self._owned(task_id, worker), status="failed", error=error)

    async def retry_later(self, task_id: int, worker: str, error: str, delay: timedelta) -> bool:
        row = self._owned(task_id, worker)
        if row is not None:
            row.run_after = self.now() + delay
        return self._end(row, status="queued", error=error)

    async def cancelled(self, task_id: int, worker: str) -> bool:
        return self._end(self._owned(task_id, worker), status="cancelled")

    async def wait_for_approval(self, task_id: int, worker: str, pending: dict[str, Any]) -> bool:
        return self._end(self._owned(task_id, worker), status="waiting_approval", pending=pending)

    async def submit_decision(
        self, task_id: int, action_key: str, decision: dict[str, Any]
    ) -> bool:
        row = self.rows.get(task_id)
        if not row or row.task.status != "waiting_approval":
            return False
        if (row.task.pending or {}).get("action_key") != action_key:
            return False
        row.task.status, row.task.pending = "queued", None
        row.decision, row.run_after = decision, self.now()
        return True

    async def request_cancel(self, task_id: int) -> TaskStatus | None:
        row = self.rows.get(task_id)
        if row is None:
            return None
        t = row.task
        if t.status in ("queued", "waiting_approval"):
            t.status, t.pending = "cancelled", None
        elif t.status == "running":
            t.cancel_requested = True
        return t.status

    async def fail_abandoned(self) -> int:
        now, count = self.now(), 0
        for row in self.rows.values():
            t = row.task
            if (
                t.status == "running"
                and row.lease_expires_at is not None
                and row.lease_expires_at < now
                and t.attempts >= t.max_attempts
            ):
                t.status, t.error = "failed", "The worker stopped on every attempt."
                row.lease_owner = row.lease_expires_at = None
                count += 1
        return count
