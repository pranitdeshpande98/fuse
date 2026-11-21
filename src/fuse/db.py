"""Database helpers: a connection check, and the shared async connection pool."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from fuse.config import get_settings


def check_database(url: str) -> dict[str, str]:
    """Connect, and report the Postgres version and whether pgvector is installed."""
    with psycopg.connect(url, connect_timeout=5) as conn:
        server = conn.execute("SHOW server_version").fetchone()[0]
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return {"postgres": server, "pgvector": row[0] if row else "not installed"}


@asynccontextmanager
async def open_pool(url: str, max_size: int = 10) -> AsyncIterator[AsyncConnectionPool]:
    """A pool of reusable async connections, closed cleanly on exit.

    LangGraph's Postgres checkpointer needs these exact connection settings.
    """
    pool = AsyncConnectionPool(
        conninfo=url,
        min_size=1,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "connect_timeout": 5,
        },
        open=False,
    )
    await pool.open(wait=True, timeout=10)
    try:
        yield pool
    finally:
        await pool.close()


def main() -> None:
    info = check_database(get_settings().database_url.get_secret_value())
    print(f"Postgres {info['postgres']}, pgvector {info['pgvector']}")


if __name__ == "__main__":
    main()
