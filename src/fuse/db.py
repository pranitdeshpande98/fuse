"""Database helpers. For now: a quick check that Postgres and pgvector are reachable."""

import psycopg

from fuse.config import get_settings


def check_database(url: str) -> dict[str, str]:
    """Connect, and report the Postgres version and whether pgvector is installed."""
    with psycopg.connect(url, connect_timeout=5) as conn:
        server = conn.execute("SHOW server_version").fetchone()[0]
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return {"postgres": server, "pgvector": row[0] if row else "not installed"}


def main() -> None:
    info = check_database(get_settings().database_url.get_secret_value())
    print(f"Postgres {info['postgres']}, pgvector {info['pgvector']}")


if __name__ == "__main__":
    main()
