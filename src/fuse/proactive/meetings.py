"""Meeting reminders: shortly before a meeting, who's in it and what you know about them.

No LLM here on purpose: a reminder must be fast, free and predictable, and a template
does that better. The context comes from memory (what you've told Fuse about these people)
and Gmail (your latest emails with them).

Reminders are sent even in quiet hours: a 6:45 meeting is one you chose to have. Each
meeting is reminded once, using the events table's dedupe key (meeting id + start time,
so a rescheduled meeting gets a new reminder).
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from fuse.google.auth import GoogleNotConnected
from fuse.google.reader import GoogleReader
from fuse.memory.context import RECALL_MIN_SIMILARITY
from fuse.memory.store import MemoryStore
from fuse.proactive.events import EventStore
from fuse.proactive.notify import Notifier
from fuse.proactive.schedule import Clock

log = logging.getLogger("fuse.meetings")

MAX_PEOPLE = 3  # context for the first few attendees keeps the message short


def _start(event: dict[str, Any]) -> datetime | None:
    value = event.get("start", {}).get("dateTime")  # all-day events have only "date"
    return datetime.fromisoformat(value) if value else None


def _declined(event: dict[str, Any]) -> bool:
    return any(
        a.get("self") and a.get("responseStatus") == "declined" for a in event.get("attendees", [])
    )


def _others(event: dict[str, Any]) -> list[dict[str, str]]:
    return [a for a in event.get("attendees", []) if not a.get("self") and a.get("email")]


class MeetingReminders:
    def __init__(
        self,
        reader: GoogleReader,
        events: EventStore,
        notifier: Notifier,
        clock: Clock,
        minutes_before: int = 15,
        memory: MemoryStore | None = None,
        on_sent: Callable[[str], Awaitable[None]] | None = None,
        min_similarity: float = RECALL_MIN_SIMILARITY,
    ) -> None:
        self._reader, self._events, self._notifier = reader, events, notifier
        self._min_similarity = min_similarity
        self._clock, self._window = clock, timedelta(minutes=minutes_before)
        self._memory, self._on_sent = memory, on_sent

    def due(self, items: list[dict[str, Any]], now: datetime) -> list[tuple[dict, datetime]]:
        result = []
        for event in items:
            start = _start(event)
            if start is None or event.get("status") == "cancelled" or _declined(event):
                continue
            if now < start <= now + self._window:
                result.append((event, start))
        return result

    async def check(self) -> int:
        """Send a reminder for every meeting starting soon. Returns how many were sent."""
        try:
            items = await self._reader.upcoming_events(days_ahead=1)
        except GoogleNotConnected:
            return 0  # the Gmail watcher already tells you about expired sign-in
        now = self._clock.now()
        sent = 0
        for event, start in self.due(items, now):
            key = f"meeting:{event.get('id', event.get('summary'))}:{start.isoformat()}"
            stored = await self._events.add("meeting", key, {"summary": event.get("summary")})
            if stored is None:
                continue  # already reminded
            text = await self.compose(event, start, now)
            await self._events.decide(stored.id, "notified", "meeting starting soon", now)
            await self._notifier.send(text)
            if self._on_sent is not None:
                await self._on_sent(text)
            sent += 1
        return sent

    async def compose(self, event: dict[str, Any], start: datetime, now: datetime) -> str:
        minutes = max(1, round((start - now).total_seconds() / 60))
        local = start.astimezone(now.tzinfo)
        lines = [f"In {minutes} min: {event.get('summary', '(no title)')} ({local:%H:%M})"]
        where = event.get("hangoutLink") or event.get("location")
        if where:
            lines.append(f"Where: {where}")

        people = _others(event)
        if people:
            names = [p.get("displayName") or p["email"] for p in people]
            lines.append("With: " + ", ".join(names))

        notes = await self._notes(event, people[:MAX_PEOPLE])
        if notes:
            lines.append("\nWhat I know:")
            lines += [f"- {n}" for n in notes]

        emails = await self._recent_emails(people[:MAX_PEOPLE])
        if emails:
            lines.append("\nRecent email:")
            lines += [f"- {m['from']}: {m['subject']}" for m in emails]
        return "\n".join(lines)

    async def _notes(self, event: dict[str, Any], people: list[dict[str, str]]) -> list[str]:
        if self._memory is None:
            return []
        queries = [event.get("summary", "")] + [p.get("displayName") or p["email"] for p in people]
        found: dict[int, str] = {}
        for query in filter(None, queries):
            for m in await self._memory.search(query, limit=2, min_similarity=self._min_similarity):
                found[m.id] = m.content
        return list(found.values())[:5]

    async def _recent_emails(self, people: list[dict[str, str]]) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        for person in people:
            try:
                found += await self._reader.search_messages(
                    f"from:{person['email']} newer_than:30d", 2
                )
            except Exception:
                log.exception("couldn't search email for meeting context")
        return found[:4]
