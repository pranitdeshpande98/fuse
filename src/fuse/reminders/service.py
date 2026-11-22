"""Parsing "when", the reminder tools, sending due reminders, and their buttons."""

import logging
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, tool

from fuse.proactive.notify import Button, Notifier
from fuse.proactive.schedule import Clock
from fuse.reminders.store import Reminder, ReminderStore

log = logging.getLogger("fuse.reminders")

BUTTON = "rm"
TOMORROW_AT = time(9, 0)


def parse_when(text: str, tz: ZoneInfo) -> datetime:
    """'2026-11-23 09:00' (the user's local time) -> an aware datetime."""
    text = text.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d":
            parsed = datetime.combine(parsed.date(), TOMORROW_AT)
        return parsed.replace(tzinfo=tz)
    raise ValueError(f"use a time like 2026-11-23 09:00, not {text!r}")


def when_text(due: datetime, tz: ZoneInfo) -> str:
    return due.astimezone(tz).strftime("%a %d %b %H:%M")


def reminder_buttons(reminder: Reminder) -> list[Button]:
    return [
        Button("Done", f"{BUTTON}:done:{reminder.id}"),
        Button("Snooze 1 h", f"{BUTTON}:snooze:{reminder.id}"),
        Button("Tomorrow 9:00", f"{BUTTON}:tomorrow:{reminder.id}"),
    ]


def make_reminder_tools(store: ReminderStore, clock: Clock) -> list[BaseTool]:
    @tool
    async def set_reminder(text: str, when: str) -> str:
        """Remind the user of something at a time. `when` is the user's local time as
        YYYY-MM-DD HH:MM (check the current time first if the user said "tomorrow" or "in
        2 hours"). `text` is what to remind them of, e.g. "Call Pat about the proposal"."""
        now = clock.now()
        tz: ZoneInfo = now.tzinfo  # type: ignore[assignment]
        try:
            due = parse_when(when, tz)
        except ValueError as exc:
            return f"Not set: {exc}."
        if due < now - timedelta(minutes=1):
            return f"Not set: {when_text(due, tz)} has already passed."
        if not text.strip():
            return "Not set: what should I remind you of?"
        reminder = await store.add(text.strip()[:500], max(due, now))
        return f"Reminder #{reminder.id} set for {when_text(reminder.due_at, tz)}: {reminder.text}"

    @tool
    async def list_reminders() -> str:
        """The user's upcoming reminders."""
        return await reminders_text(store, clock)

    @tool
    async def cancel_reminder(reminder_id: int) -> str:
        """Cancel one of the user's reminders by its number."""
        reminder = await store.get(reminder_id)
        if reminder is None or reminder.status != "pending":
            return f"There's no upcoming reminder #{reminder_id}."
        await store.set(reminder_id, "cancelled")
        return f"Cancelled reminder #{reminder_id}: {reminder.text}"

    return [set_reminder, list_reminders, cancel_reminder]


async def reminders_text(store: ReminderStore, clock: Clock) -> str:
    upcoming = await store.upcoming()
    if not upcoming:
        return 'No upcoming reminders. Try: "remind me tomorrow at 9 to call Pat".'
    tz = clock.now().tzinfo
    lines = ["Upcoming reminders:"]
    lines += [f"#{r.id} {when_text(r.due_at, tz)}: {r.text}" for r in upcoming]  # type: ignore[arg-type]
    lines.append('Cancel one by asking, e.g. "cancel reminder 3".')
    return "\n".join(lines)


class ReminderSender:
    """Runs on the scheduler: sends every reminder that's due."""

    def __init__(self, store: ReminderStore, notifier: Notifier, clock: Clock) -> None:
        self._store, self._notifier, self._clock = store, notifier, clock

    async def check(self) -> int:
        sent = 0
        for reminder in await self._store.take_due(self._clock.now()):
            try:
                await self._notifier.send(f"Reminder: {reminder.text}", reminder_buttons(reminder))
                sent += 1
            except Exception:
                log.exception("couldn't send reminder %s; will retry", reminder.id)
                await self._store.set(reminder.id, "pending")  # tried again on the next check
        return sent


async def handle_reminder_button(store: ReminderStore, clock: Clock, data: str) -> str:
    """Done / Snooze 1 h / Tomorrow 9:00 on a reminder. Returns the confirmation."""
    _, action, raw_id = (data.split(":") + ["", "", ""])[:3]
    if not raw_id.isdigit():
        return "That button isn't valid any more."
    reminder_id, now = int(raw_id), clock.now()
    if action == "done":
        ok = await store.set(reminder_id, "done")
        return "Done." if ok else "I can't find that reminder any more."
    if action == "snooze":
        until = now + timedelta(hours=1)
    elif action == "tomorrow":
        until = datetime.combine(now.date() + timedelta(days=1), TOMORROW_AT, tzinfo=now.tzinfo)
    else:
        return "That button isn't valid any more."
    if not await store.set(reminder_id, "pending", until):
        return "I can't find that reminder any more."
    return f"I'll remind you again {when_text(until, now.tzinfo)}."  # type: ignore[arg-type]
