"""Everything a chat channel does, without knowing which channel it is.

A channel adapter (Telegram today, WhatsApp in Phase 6) only translates: it turns an
incoming message, button tap or command into a call here, and turns what comes back into
that platform's messages and buttons. The rules (who can approve what, which buttons a
draft gets, what /tasks shows) live here once, so every channel behaves the same.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from fuse.agent.runner import AgentReply, AgentRunner
from fuse.approval_ui import after_allowing, approval_buttons, approval_text
from fuse.contacts import ContactStore, contacts_text
from fuse.memory.commands import forget_text, list_memories_text
from fuse.memory.store import MemoryStore
from fuse.proactive.alerts import AlertActions
from fuse.proactive.notify import Button, split_message
from fuse.proactive.store import MuteStore
from fuse.tasks.queue import TaskStore
from fuse.tasks.report import (
    TASK_BUTTON,
    allow_task_recipients,
    cancel_text,
    decide_task_approval,
    tasks_text,
)
from fuse.tasks.tools import start_task

log = logging.getLogger("fuse.core")

ERROR_TEXT = "Something went wrong on my side. Check the terminal logs."
STALE = "This is no longer waiting for approval."

Rows = list[list[Button]]


@dataclass
class Outgoing:
    text: str
    buttons: Rows = field(default_factory=list)


@dataclass
class ButtonResult:
    toast: str = ""  # a short confirmation shown where the button was
    buttons: Rows | None = None  # new buttons for that message; None removes them
    messages: list[Outgoing] = field(default_factory=list)


@dataclass
class Command:
    help: str
    run: Callable[[str, str], Awaitable[list[Outgoing]]]  # (thread_id, args) -> messages


class ChatCore:
    def __init__(
        self,
        runner: AgentRunner,
        memory: MemoryStore | None = None,
        tasks: TaskStore | None = None,
        contacts: ContactStore | None = None,
        mutes: MuteStore | None = None,
        alert_actions: AlertActions | None = None,
        brief: Callable[[], Awaitable[None]] | None = None,
        task_max_attempts: int = 3,
    ) -> None:
        self.runner, self.memory, self.tasks = runner, memory, tasks
        self.contacts, self.mutes, self.alert_actions = contacts, mutes, alert_actions
        self.brief, self.task_max_attempts = brief, task_max_attempts
        self.commands: dict[str, Command] = {
            "brief": Command("today's brief, now", self._brief),
            "task": Command("start a background task: /task <goal>", self._task),
            "tasks": Command("recent background tasks", self._tasks),
            "cancel": Command("stop a task: /cancel <id>", self._cancel),
            "contacts": Command("who Fuse may email: /contacts [add|remove]", self._contacts),
            "memories": Command("everything Fuse remembers about you", self._memories),
            "forget": Command("delete a memory: /forget <id>", self._forget),
            "unmute": Command("alerts from a sender again: /unmute [address]", self._unmute),
        }

    # --- rendering ---------------------------------------------------------------------

    def render(self, reply: AgentReply) -> list[Outgoing]:
        """An agent reply as messages: plain text (split if long), or a preview with the
        buttons that fit it (Send / Save as draft / Reject for an email)."""
        if reply.kind == "approval":
            payload = reply.approval
            key = payload["action_key"]
            buttons = approval_buttons(payload, lambda verb: f"{verb}:{key}")
            return [Outgoing(f"{reply.text}\n\n{approval_text(payload)}", buttons)]
        return [Outgoing(part) for part in split_message(reply.text)]

    # --- messages ----------------------------------------------------------------------

    async def message(self, thread_id: str, text: str) -> list[Outgoing]:
        log.info("message received (%d chars)", len(text))
        try:
            return self.render(await self.runner.send(thread_id, text))
        except Exception:
            log.exception("agent failed")
            return [Outgoing(ERROR_TEXT)]

    # --- buttons -----------------------------------------------------------------------

    async def button(self, thread_id: str, data: str) -> ButtonResult:
        try:
            if data.startswith("ev:"):
                return await self._alert_button(thread_id, data)
            if data.startswith(f"{TASK_BUTTON}:"):
                return await self._task_button(data)
            return await self._approval_button(thread_id, data)
        except Exception:
            log.exception("button failed")
            return ButtonResult(messages=[Outgoing(ERROR_TEXT)])

    async def _approval_button(self, thread_id: str, data: str) -> ButtonResult:
        verb, _, key = data.partition(":")
        # Only act if this exact action is still the one waiting (old buttons do nothing).
        pending = await self.runner.pending_approval(thread_id)
        if not pending or pending.get("action_key") != key:
            return ButtonResult(toast=STALE)
        if verb == "allow":
            if self.contacts is None:
                return ButtonResult(toast="Contacts aren't available.", buttons=None)
            for address in pending.get("blocked") or []:
                await self.contacts.add(address, source="always allow")
            buttons = approval_buttons(after_allowing(pending), lambda v: f"{v}:{key}")
            return ButtonResult(toast="Added to your contacts", buttons=buttons)
        if verb not in ("approve", "send", "reject"):
            return ButtonResult(toast="That button isn't valid any more.")
        labels = {"send": "Sending", "approve": "Approved", "reject": "Rejected"}
        log.info("action %s in %s", verb, thread_id)
        reply = await self.runner.resume(
            thread_id, approved=verb != "reject", mode="send" if verb == "send" else None
        )
        return ButtonResult(toast=labels[verb], messages=self.render(reply))

    async def _alert_button(self, thread_id: str, data: str) -> ButtonResult:
        if self.alert_actions is None:
            return ButtonResult(toast="Alerts aren't available.")
        reply = await self.alert_actions.handle(data, thread_id)
        return ButtonResult(messages=self.render(reply))  # each alert is handled once

    async def _task_button(self, data: str) -> ButtonResult:
        if self.tasks is None:
            return ButtonResult(toast="Tasks aren't available.")
        if data.startswith(f"{TASK_BUTTON}:allow:") and self.contacts is not None:
            text, buttons = await allow_task_recipients(self.tasks, self.contacts, data)
            return ButtonResult(toast=text[:200], buttons=buttons)
        text = await decide_task_approval(self.tasks, data)
        return ButtonResult(toast=text[:200], messages=[Outgoing(text)])

    # --- commands ----------------------------------------------------------------------

    async def command(self, thread_id: str, name: str, args: str = "") -> list[Outgoing]:
        command = self.commands.get(name)
        if command is None:
            return [Outgoing(f"I don't know /{name}.")]
        try:
            return await command.run(thread_id, args.strip())
        except Exception:
            log.exception("command /%s failed", name)
            return [Outgoing(ERROR_TEXT)]

    @staticmethod
    def _one(text: str) -> list[Outgoing]:
        return [Outgoing(part) for part in split_message(text)]

    async def _brief(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.brief is None:
            return self._one("The brief isn't set up.")
        await self.brief()  # the brief sends itself, and joins the conversation
        return []

    async def _task(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.tasks is None:
            return self._one("Background tasks aren't set up.")
        return self._one(await start_task(self.tasks, args, self.task_max_attempts))

    async def _tasks(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.tasks is None:
            return self._one("Background tasks aren't set up.")
        return self._one(await tasks_text(self.tasks))

    async def _cancel(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.tasks is None:
            return self._one("Background tasks aren't set up.")
        return self._one(await cancel_text(self.tasks, args))

    async def _contacts(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.contacts is None:
            return self._one("Contacts aren't set up.")
        return self._one(await contacts_text(self.contacts, args))

    async def _memories(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.memory is None:
            return self._one("Memory isn't set up.")
        return self._one(await list_memories_text(self.memory))

    async def _forget(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.memory is None:
            return self._one("Memory isn't set up.")
        return self._one(await forget_text(self.memory, args))

    async def _unmute(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.mutes is None:
            return self._one("Alerts aren't set up.")
        if not args:
            muted = sorted(await self.mutes.all())
            if not muted:
                return self._one("You haven't muted anyone.")
            return self._one("Muted: " + ", ".join(muted) + "\nUsage: /unmute name@example.com")
        if await self.mutes.unmute(args):
            return self._one(f"Unmuted {args}.")
        return self._one(f"{args} wasn't muted.")


COMMAND_NAMES = (
    "brief",
    "task",
    "tasks",
    "cancel",
    "contacts",
    "memories",
    "forget",
    "unmute",
)
