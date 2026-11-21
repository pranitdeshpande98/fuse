"""The actions log: every side effect Fuse takes is recorded here, and runs at most once.

Each action gets an idempotency key. Before running an action we check whether that key
already succeeded, so a retry after a crash can never send the same email twice.
"""

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

# Each statement runs separately: the pool prepares statements, and a prepared statement
# can hold only one command.
SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS actions (
        idempotency_key TEXT PRIMARY KEY,
        tool            TEXT        NOT NULL,
        args            JSONB       NOT NULL,
        status          TEXT        NOT NULL CHECK (status IN
                            ('pending', 'approved', 'rejected', 'executed', 'failed')),
        result          TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS actions_status_idx ON actions (status)",
]


def idempotency_key(thread_id: str, tool_call_id: str) -> str:
    """Stable for one tool call: the same call, retried or resumed, gets the same key."""
    return hashlib.sha256(f"{thread_id}:{tool_call_id}".encode()).hexdigest()[:32]


@dataclass
class ActionRecord:
    key: str
    tool: str
    args: dict[str, Any]
    status: str  # pending | approved | rejected | executed | failed
    result: str | None = None


class ActionStore(Protocol):
    """Anything that can record actions. Two implementations below: memory and Postgres."""

    async def record(self, key: str, tool: str, args: dict[str, Any]) -> ActionRecord: ...
    async def get(self, key: str) -> ActionRecord | None: ...
    async def set_status(self, key: str, status: str, result: str | None = None) -> None: ...


async def execute_once(store: ActionStore, key: str, run: Callable[[], Awaitable[str]]) -> str:
    """Run the action unless this key already succeeded. Records the outcome either way."""
    existing = await store.get(key)
    if existing and existing.status == "executed":
        return existing.result or "Already done."
    try:
        result = await run()
    except Exception as exc:
        await store.set_status(key, "failed", f"{type(exc).__name__}: {exc}")
        raise
    await store.set_status(key, "executed", result)
    return result


class InMemoryActionStore:
    """For tests. Forgets everything when the process stops."""

    def __init__(self) -> None:
        self._rows: dict[str, ActionRecord] = {}

    async def record(self, key: str, tool: str, args: dict[str, Any]) -> ActionRecord:
        if key not in self._rows:  # recording the same key again changes nothing
            self._rows[key] = ActionRecord(key=key, tool=tool, args=args, status="pending")
        return self._rows[key]

    async def get(self, key: str) -> ActionRecord | None:
        return self._rows.get(key)

    async def set_status(self, key: str, status: str, result: str | None = None) -> None:
        row = self._rows[key]
        row.status = status
        if result is not None:
            row.result = result


class PostgresActionStore:
    """Stores actions in the `actions` table."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        """Create the table if it doesn't exist yet. Safe to run on every startup."""
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def record(self, key: str, tool: str, args: dict[str, Any]) -> ActionRecord:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO actions (idempotency_key, tool, args, status)
                VALUES (%s, %s, %s, 'pending')
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (key, tool, Jsonb(args)),
            )
        row = await self.get(key)
        assert row is not None
        return row

    async def get(self, key: str) -> ActionRecord | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT tool, args, status, result FROM actions WHERE idempotency_key = %s",
                (key,),
            )
            row = await cur.fetchone()  # a dict, because the pool uses dict_row
        if row is None:
            return None
        return ActionRecord(
            key=key, tool=row["tool"], args=row["args"], status=row["status"], result=row["result"]
        )

    async def set_status(self, key: str, status: str, result: str | None = None) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                UPDATE actions
                SET status = %s, result = COALESCE(%s, result), updated_at = now()
                WHERE idempotency_key = %s
                """,
                (status, result, key),
            )
