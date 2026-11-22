"""Small Postgres tables the proactive side needs, each with an in-memory twin for tests."""

from typing import Protocol

from psycopg_pool import AsyncConnectionPool

CURSOR_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sync_cursors (
        name        TEXT PRIMARY KEY,
        value       TEXT        NOT NULL,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
]


class CursorStore(Protocol):
    """Where each watcher got up to, e.g. the last Gmail history ID it processed."""

    async def get(self, name: str) -> str | None: ...
    async def set(self, name: str, value: str) -> None: ...


class PostgresCursorStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in CURSOR_SCHEMA:
                await conn.execute(statement)

    async def get(self, name: str) -> str | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT value FROM sync_cursors WHERE name = %s", (name,))
            row = await cur.fetchone()
        return row["value"] if row else None

    async def set(self, name: str, value: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO sync_cursors (name, value) VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (name, value),
            )


class InMemoryCursorStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, name: str) -> str | None:
        return self.values.get(name)

    async def set(self, name: str, value: str) -> None:
        self.values[name] = value
