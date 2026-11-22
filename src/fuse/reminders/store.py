"""Reminders: "remind me tomorrow at 9 to call Pat".

A reminder is a row with a due time. The scheduler checks every 30 seconds and sends any
that are due, with Done / Snooze 1 h / Tomorrow 9:00 buttons. Sending is marked first, then
done, the same way as alerts: a reminder can't arrive twice, and if Telegram fails it goes
back to pending and is tried again on the next check.
"""

import itertools
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Literal, Protocol

from psycopg_pool import AsyncConnectionPool

Status = Literal["pending", "sent", "done", "cancelled"]

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS reminders (
        id          BIGSERIAL PRIMARY KEY,
        text        TEXT        NOT NULL,
        due_at      TIMESTAMPTZ NOT NULL,
        status      TEXT        NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'sent', 'done', 'cancelled')),
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS reminders_due_idx ON reminders (due_at) WHERE status = 'pending'",
]


@dataclass
class Reminder:
    id: int
    text: str
    due_at: datetime
    status: Status = "pending"


class ReminderStore(Protocol):
    async def add(self, text: str, due_at: datetime) -> Reminder: ...
    async def get(self, reminder_id: int) -> Reminder | None: ...
    async def upcoming(self, limit: int = 20) -> list[Reminder]: ...
    async def take_due(self, now: datetime) -> list[Reminder]:
        """Due reminders, marked "sent" in the same step so no two checks send one twice."""
        ...

    async def set(self, reminder_id: int, status: Status, due_at: datetime | None = None) -> bool:
        """Change status (and optionally the due time, for snoozing). False if not found or
        already cancelled."""
        ...


def _reminder(row: dict[str, Any]) -> Reminder:
    return Reminder(row["id"], row["text"], row["due_at"], row["status"])


class PostgresReminderStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def add(self, text: str, due_at: datetime) -> Reminder:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO reminders (text, due_at) VALUES (%s, %s) "
                "RETURNING id, text, due_at, status",
                (text, due_at),
            )
            return _reminder(await cur.fetchone())

    async def get(self, reminder_id: int) -> Reminder | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, text, due_at, status FROM reminders WHERE id = %s", (reminder_id,)
            )
            row = await cur.fetchone()
        return _reminder(row) if row else None

    async def upcoming(self, limit: int = 20) -> list[Reminder]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, text, due_at, status FROM reminders WHERE status = 'pending' "
                "ORDER BY due_at LIMIT %s",
                (limit,),
            )
            return [_reminder(r) for r in await cur.fetchall()]

    async def take_due(self, now: datetime) -> list[Reminder]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE reminders SET status = 'sent'
                WHERE id IN (
                    SELECT id FROM reminders WHERE status = 'pending' AND due_at <= %s
                    ORDER BY due_at FOR UPDATE SKIP LOCKED
                )
                RETURNING id, text, due_at, status
                """,
                (now,),
            )
            rows = await cur.fetchall()
        return sorted((_reminder(r) for r in rows), key=lambda r: r.due_at)

    async def set(self, reminder_id: int, status: Status, due_at: datetime | None = None) -> bool:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE reminders SET status = %s, due_at = COALESCE(%s, due_at) "
                "WHERE id = %s AND status <> 'cancelled' RETURNING id",
                (status, due_at, reminder_id),
            )
            return await cur.fetchone() is not None


class InMemoryReminderStore:
    def __init__(self) -> None:
        self.reminders: dict[int, Reminder] = {}
        self._ids = itertools.count(1)

    async def add(self, text: str, due_at: datetime) -> Reminder:
        reminder = Reminder(next(self._ids), text, due_at)
        self.reminders[reminder.id] = reminder
        return replace(reminder)

    async def get(self, reminder_id: int) -> Reminder | None:
        r = self.reminders.get(reminder_id)
        return replace(r) if r else None

    async def upcoming(self, limit: int = 20) -> list[Reminder]:
        pending = [r for r in self.reminders.values() if r.status == "pending"]
        return [replace(r) for r in sorted(pending, key=lambda r: r.due_at)][:limit]

    async def take_due(self, now: datetime) -> list[Reminder]:
        due = [r for r in self.reminders.values() if r.status == "pending" and r.due_at <= now]
        for r in due:
            r.status = "sent"
        return [replace(r) for r in sorted(due, key=lambda r: r.due_at)]

    async def set(self, reminder_id: int, status: Status, due_at: datetime | None = None) -> bool:
        r = self.reminders.get(reminder_id)
        if r is None or r.status == "cancelled":
            return False
        r.status = status
        if due_at is not None:
            r.due_at = due_at
        return True
