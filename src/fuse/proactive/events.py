"""Everything that might deserve your attention becomes one row in `events`.

New email and upcoming meetings come from different places, but after this point they
look the same: a kind, a payload, and a dedupe key. The decision engine reads new events
and decides what happens to each, and the row records that decision and why.

Life of an event:

    new --> ignored                  not worth your attention
        --> held --> delivered       waits for the morning brief (quiet hours, digest)
        --> notified / suggested     sent to you now (suggested = with a draft offer)
                 --> dismissed / muted   what you did with the alert

The dedupe key (e.g. "gmail:<message id>") is UNIQUE, so seeing the same email twice,
after a crash or an overlapping poll, can never create a second alert.
"""

import itertools
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Protocol

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

EventKind = Literal["email", "meeting"]
Status = Literal[
    "new", "ignored", "held", "delivered", "notified", "suggested", "dismissed", "muted"
]

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS events (
        id          BIGSERIAL PRIMARY KEY,
        kind        TEXT        NOT NULL,
        dedupe_key  TEXT        NOT NULL UNIQUE,
        payload     JSONB       NOT NULL,
        status      TEXT        NOT NULL DEFAULT 'new',
        reason      TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        decided_at  TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS events_status_idx ON events (status, id)",
]


@dataclass
class Event:
    id: int
    kind: EventKind
    dedupe_key: str
    payload: dict[str, Any]
    status: Status = "new"
    reason: str | None = None
    decided_at: datetime | None = None


class EventStore(Protocol):
    async def add(self, kind: EventKind, dedupe_key: str, payload: dict[str, Any]) -> Event | None:
        """The new event, or None if one with this dedupe key already exists."""
        ...

    async def get(self, event_id: int) -> Event | None: ...
    async def pending(self, limit: int = 20) -> list[Event]: ...
    async def decide(self, event_id: int, status: Status, reason: str, at: datetime) -> None: ...
    async def set_status(self, event_id: int, status: Status) -> None: ...
    async def sent_since(self, since: datetime) -> int:
        """How many alerts went out since then: the hourly cap counts these."""
        ...

    async def take_held(self) -> list[Event]:
        """Held events, marked delivered in the same step so the brief shows each once."""
        ...


def _event(row: dict[str, Any]) -> Event:
    return Event(
        id=row["id"],
        kind=row["kind"],
        dedupe_key=row["dedupe_key"],
        payload=row["payload"],
        status=row["status"],
        reason=row["reason"],
        decided_at=row["decided_at"],
    )


COLUMNS = "id, kind, dedupe_key, payload, status, reason, decided_at"


class PostgresEventStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def add(self, kind: EventKind, dedupe_key: str, payload: dict[str, Any]) -> Event | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                INSERT INTO events (kind, dedupe_key, payload) VALUES (%s, %s, %s)
                ON CONFLICT (dedupe_key) DO NOTHING
                RETURNING {COLUMNS}
                """,
                (kind, dedupe_key, Jsonb(payload)),
            )
            row = await cur.fetchone()
        return _event(row) if row else None

    async def get(self, event_id: int) -> Event | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(f"SELECT {COLUMNS} FROM events WHERE id = %s", (event_id,))
            row = await cur.fetchone()
        return _event(row) if row else None

    async def pending(self, limit: int = 20) -> list[Event]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {COLUMNS} FROM events WHERE status = 'new' ORDER BY id LIMIT %s",
                (limit,),
            )
            return [_event(r) for r in await cur.fetchall()]

    async def decide(self, event_id: int, status: Status, reason: str, at: datetime) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE events SET status = %s, reason = %s, decided_at = %s WHERE id = %s",
                (status, reason, at, event_id),
            )

    async def set_status(self, event_id: int, status: Status) -> None:
        async with self._pool.connection() as conn:
            await conn.execute("UPDATE events SET status = %s WHERE id = %s", (status, event_id))

    async def sent_since(self, since: datetime) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM events "
                "WHERE status IN ('notified', 'suggested', 'dismissed', 'muted') "
                "AND decided_at >= %s",
                (since,),
            )
            row = await cur.fetchone()
        return row["n"]

    async def take_held(self) -> list[Event]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                UPDATE events SET status = 'delivered' WHERE status = 'held'
                RETURNING {COLUMNS}
                """
            )
            rows = await cur.fetchall()
        return sorted((_event(r) for r in rows), key=lambda e: e.id)


@dataclass
class InMemoryEventStore:
    events: dict[int, Event] = field(default_factory=dict)
    _ids: Any = field(default_factory=lambda: itertools.count(1))

    async def add(self, kind: EventKind, dedupe_key: str, payload: dict[str, Any]) -> Event | None:
        if any(e.dedupe_key == dedupe_key for e in self.events.values()):
            return None
        event = Event(next(self._ids), kind, dedupe_key, dict(payload))
        self.events[event.id] = event
        return replace(event)

    async def get(self, event_id: int) -> Event | None:
        event = self.events.get(event_id)
        return replace(event) if event else None

    async def pending(self, limit: int = 20) -> list[Event]:
        return [replace(e) for e in self.events.values() if e.status == "new"][:limit]

    async def decide(self, event_id: int, status: Status, reason: str, at: datetime) -> None:
        event = self.events[event_id]
        event.status, event.reason, event.decided_at = status, reason, at

    async def set_status(self, event_id: int, status: Status) -> None:
        self.events[event_id].status = status

    async def sent_since(self, since: datetime) -> int:
        sent = ("notified", "suggested", "dismissed", "muted")
        return sum(
            1
            for e in self.events.values()
            if e.status in sent and e.decided_at is not None and e.decided_at >= since
        )

    async def take_held(self) -> list[Event]:
        held = [e for e in self.events.values() if e.status == "held"]
        for e in held:
            e.status = "delivered"
        return [replace(e) for e in held]
