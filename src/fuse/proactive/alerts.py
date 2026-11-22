"""Turning decided events into Telegram alerts, and handling the buttons on them."""

import logging
from collections.abc import Awaitable, Callable

from fuse.agent.runner import AgentReply, AgentRunner
from fuse.proactive.decide import sender_address
from fuse.proactive.events import Event, EventStore, Status
from fuse.proactive.notify import Button, Notifier
from fuse.proactive.store import MuteStore

log = logging.getLogger("fuse.alerts")

BUTTON_PREFIX = "ev"

# What tapping "Draft reply" asks the agent. Only the thread ID goes in, never the subject
# or sender: those come from the email, and putting them in *your* message would let an
# email speak with your voice. The agent reads the thread through its tool, which wraps
# the content as untrusted, and the draft still needs your Approve.
DRAFT_REQUEST = (
    "Please draft a reply to the email thread with ID {thread_id}. "
    "Read the thread first, then propose the draft."
)


def alert_text(event: Event, status: Status, reason: str) -> str:
    email = event.payload
    lines = [
        f"New email from {email.get('from', '?')}",
        f"Subject: {email.get('subject', '(no subject)')}",
        "",
        f"Why you're seeing this: {reason}",
    ]
    if status == "suggested":
        lines.append("A short reply looks expected. Want me to draft one?")
    return "\n".join(lines)


def alert_buttons(event: Event) -> list[Button]:
    return [
        Button("Draft reply", f"{BUTTON_PREFIX}:draft:{event.id}"),
        Button("Dismiss", f"{BUTTON_PREFIX}:dismiss:{event.id}"),
        Button("Mute sender", f"{BUTTON_PREFIX}:mute:{event.id}"),
    ]


class AlertSender:
    """Passed to the decision engine as on_alert."""

    def __init__(
        self, notifier: Notifier, on_sent: Callable[[str], Awaitable[None]] | None = None
    ) -> None:
        self._notifier = notifier
        self._on_sent = on_sent

    async def __call__(self, event: Event, status: Status, reason: str) -> None:
        text = alert_text(event, status, reason)
        await self._notifier.send(text, alert_buttons(event))
        if self._on_sent is not None:
            # In the conversation too, so "draft a reply to that" works without the button.
            await self._on_sent(f"{text}\n(email thread ID: {event.payload.get('thread_id')})")


class EventHeldItems:
    """Feeds held events into the morning brief (each one once)."""

    def __init__(self, events: EventStore) -> None:
        self._events = events

    async def take_held(self) -> list[str]:
        return [
            f"From {e.payload.get('from', '?')}: {e.payload.get('subject', '')} ({e.reason})"
            for e in await self._events.take_held()
            if e.kind == "email"
        ]


class AlertActions:
    """What the alert buttons do."""

    def __init__(self, events: EventStore, mutes: MuteStore, runner: AgentRunner) -> None:
        self._events, self._mutes, self._runner = events, mutes, runner

    async def handle(self, data: str, thread_id: str) -> AgentReply:
        prefix, action, raw_id = (data.split(":") + ["", "", ""])[:3]
        if prefix != BUTTON_PREFIX or not raw_id.isdigit():
            return AgentReply("text", "That button isn't valid any more.")
        event = await self._events.get(int(raw_id))
        if event is None or event.kind != "email":
            return AgentReply("text", "I can't find that email any more.")

        if action == "dismiss":
            await self._events.set_status(event.id, "dismissed")
            return AgentReply("text", "Dismissed.")
        if action == "mute":
            address = sender_address(event.payload.get("from", ""))
            if not address:
                return AgentReply("text", "I couldn't tell who sent that one.")
            await self._mutes.mute(address)
            await self._events.set_status(event.id, "muted")
            return AgentReply(
                "text", f"Muted {address}. I won't alert you about them. Undo: /unmute {address}"
            )
        if action == "draft":
            request = DRAFT_REQUEST.format(thread_id=event.payload.get("thread_id", ""))
            return await self._runner.send(thread_id, request)
        return AgentReply("text", "I don't know that button.")
