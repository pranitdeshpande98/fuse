"""A log of scheduled job runs, so a daily job runs at most once per day, even if the bot
restarts, and a failed run is retried on the next tick."""

import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger("fuse.jobs")

# A run still marked "running" after this long is assumed to have died with the bot.
STALE_MINUTES = 10

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS job_runs (
        job          TEXT        NOT NULL,
        run_key      TEXT        NOT NULL,
        status       TEXT        NOT NULL CHECK (status IN ('running', 'done', 'failed')),
        detail       TEXT,
        started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at  TIMESTAMPTZ,
        PRIMARY KEY (job, run_key)
    )
    """,
]


class JobLog(Protocol):
    async def run_once(self, job: str, run_key: str, fn: Callable[[], Awaitable[None]]) -> bool:
        """Run fn unless (job, run_key) already ran or is running. True if it ran and
        succeeded."""
        ...

    async def status(self, job: str, run_key: str) -> str | None: ...


class PostgresJobLog:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def _claim(self, job: str, run_key: str) -> bool:
        """Atomically take the run. Only one caller wins, even with two bots running."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                INSERT INTO job_runs (job, run_key, status) VALUES (%s, %s, 'running')
                ON CONFLICT (job, run_key) DO UPDATE
                    SET status = 'running', started_at = now(), detail = NULL
                    WHERE job_runs.status = 'failed'
                       OR (job_runs.status = 'running'
                           AND job_runs.started_at < now() - interval '{STALE_MINUTES} minutes')
                RETURNING job
                """,
                (job, run_key),
            )
            return await cur.fetchone() is not None

    async def _finish(self, job: str, run_key: str, status: str, detail: str | None) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE job_runs SET status = %s, detail = %s, finished_at = now() "
                "WHERE job = %s AND run_key = %s",
                (status, detail, job, run_key),
            )

    async def run_once(self, job: str, run_key: str, fn: Callable[[], Awaitable[None]]) -> bool:
        if not await self._claim(job, run_key):
            return False
        try:
            await fn()
        except Exception as exc:
            log.exception("job %s (%s) failed; will retry", job, run_key)
            await self._finish(job, run_key, "failed", f"{type(exc).__name__}: {exc}"[:500])
            return False
        await self._finish(job, run_key, "done", None)
        return True

    async def status(self, job: str, run_key: str) -> str | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT status FROM job_runs WHERE job = %s AND run_key = %s", (job, run_key)
            )
            row = await cur.fetchone()
        return row["status"] if row else None


class InMemoryJobLog:
    """Same behaviour, for tests."""

    def __init__(self) -> None:
        self.runs: dict[tuple[str, str], str] = {}

    async def run_once(self, job: str, run_key: str, fn: Callable[[], Awaitable[None]]) -> bool:
        if self.runs.get((job, run_key)) in ("running", "done"):
            return False
        self.runs[(job, run_key)] = "running"
        try:
            await fn()
        except Exception:
            log.exception("job %s (%s) failed; will retry", job, run_key)
            self.runs[(job, run_key)] = "failed"
            return False
        self.runs[(job, run_key)] = "done"
        return True

    async def status(self, job: str, run_key: str) -> str | None:
        return self.runs.get((job, run_key))
