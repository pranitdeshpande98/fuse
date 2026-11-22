"""Everything a chat channel does, without knowing which channel it is.

A channel adapter (Telegram today, WhatsApp in Phase 6) only translates: it turns an
incoming message, button tap or command into a call here, and turns what comes back into
that platform's messages and buttons. The rules (who can approve what, which buttons a
draft gets, what /tasks shows) live here once, so every channel behaves the same.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

from fuse.agent.runner import AgentReply, AgentRunner
from fuse.approval_ui import after_allowing, approval_buttons, approval_text
from fuse.contacts import ContactStore, contacts_text
from fuse.jobalerts.client import JobAlertsClient, JobAlertsError
from fuse.jobalerts.service import APPLIED_BUTTON, TRACK_BUTTON, ScanRunner
from fuse.jobalerts.tools import status_text
from fuse.jobs.store import JobStore, jobs_text
from fuse.memory.commands import forget_text, list_memories_text
from fuse.memory.store import MemoryStore
from fuse.proactive.alerts import AlertActions
from fuse.proactive.events import EventStore
from fuse.proactive.notify import Button, split_message
from fuse.proactive.schedule import Clock
from fuse.proactive.store import MuteStore
from fuse.reminders.service import BUTTON as REMINDER_BUTTON
from fuse.reminders.service import handle_reminder_button, reminders_text
from fuse.reminders.store import ReminderStore
from fuse.skills.loader import Skill
from fuse.skills.tools import skills_text, start_skill
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

FOLLOW_UP_DAYS = 10
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
    only_this_button: bool = False  # remove just the tapped button, keep the others
    only_this_row: bool = False  # remove the tapped button's row (e.g. one job's buttons)
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
        skills: list[Skill] | None = None,
        jobs: JobStore | None = None,
        job_alerts: JobAlertsClient | None = None,
        scan_runner: ScanRunner | None = None,
        events: EventStore | None = None,
        reminders: ReminderStore | None = None,
        clock: Clock | None = None,
        status: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self.runner, self.memory, self.tasks = runner, memory, tasks
        self.contacts, self.mutes, self.alert_actions = contacts, mutes, alert_actions
        self.brief, self.task_max_attempts = brief, task_max_attempts
        self.skills, self.jobs = skills or [], jobs
        self.job_alerts, self.scan_runner, self.events = job_alerts, scan_runner, events
        self.reminders, self.clock, self.status = reminders, clock, status
        self.commands: dict[str, Command] = {
            "brief": Command("today's brief, now", self._brief),
            "task": Command("start a background task: /task <goal>", self._task),
            "tasks": Command("recent background tasks", self._tasks),
            "cancel": Command("stop a task: /cancel <id>", self._cancel),
            "contacts": Command("who Fuse may email: /contacts [add|remove]", self._contacts),
            "memories": Command("everything Fuse remembers about you", self._memories),
            "forget": Command("delete a memory: /forget <id>", self._forget),
            "unmute": Command("alerts from a sender again: /unmute [address]", self._unmute),
            "skills": Command("your skills and when they run", self._skills),
            "skill": Command("run a skill now: /skill <name>", self._skill),
            "jobs": Command("your job applications", self._jobs),
            "scan": Command("scan your Job Alerts portals now", self._scan),
            "dashboard": Command("Job Alerts status and failing portals", self._dashboard),
            "reminders": Command("your upcoming reminders", self._reminders),
            "status": Command("is every part of Fuse healthy?", self._status),
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
            if data.startswith(f"{TRACK_BUTTON}:"):
                return await self._track_button(data)
            if data.startswith(f"{APPLIED_BUTTON}:"):
                return await self._applied_button(data)
            if data.startswith(f"{REMINDER_BUTTON}:"):
                if self.reminders is None or self.clock is None:
                    return ButtonResult(toast="Reminders aren't set up.")
                text = await handle_reminder_button(self.reminders, self.clock, data)
                return ButtonResult(toast=text, messages=[Outgoing(text)])
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

    async def _job_from_button(self, data: str) -> dict | None:
        raw_id = data.partition(":")[2]
        if self.events is None or self.jobs is None or not raw_id.isdigit():
            return None
        event = await self.events.get(int(raw_id))
        return event.payload if event is not None and event.kind == "job" else None

    async def _existing(self, company: str, role: str) -> Any:
        assert self.jobs is not None
        for app in await self.jobs.list():
            if (app.company.lower(), app.role.lower()) == (company.lower(), role.lower()):
                return app
        return None

    async def _track_button(self, data: str) -> ButtonResult:
        """Track on a Job Alerts match: add it to your applications as "interested"."""
        job = await self._job_from_button(data)
        if job is None or self.jobs is None:
            return ButtonResult(toast="I can't find that job any more.")
        company, role = job.get("company") or "?", job.get("title") or "?"
        existing = await self._existing(company, role)
        if existing is not None:  # never move an application backwards (applied -> interested)
            text = f"Already tracked: {existing.company}, {existing.role} [{existing.stage}]."
            return ButtonResult(
                toast="Already tracked", messages=[Outgoing(text)], only_this_button=True
            )
        app, _ = await self.jobs.save(
            company,
            role,
            "interested",
            notes=f"From Job Alerts ({job.get('location') or 'location not listed'})",
            url=job.get("url", ""),
        )
        text = (
            f"Added to your applications: {app.company}, {app.role} [interested].\n"
            "Tap Applied when you've applied, or just tell me. /jobs shows them all."
        )
        return ButtonResult(toast="Tracked", messages=[Outgoing(text)], only_this_button=True)

    async def _applied_button(self, data: str) -> ButtonResult:
        """Applied on a Job Alerts match: stage applied, and a follow-up reminder."""
        job = await self._job_from_button(data)
        if job is None or self.jobs is None or self.clock is None:
            return ButtonResult(toast="I can't find that job any more.")
        now = self.clock.now()
        follow_up = datetime.combine(
            now.date() + timedelta(days=FOLLOW_UP_DAYS), time(9, 0), tzinfo=now.tzinfo
        )
        company, role = job.get("company") or "?", job.get("title") or "?"
        app, _ = await self.jobs.save(
            company,
            role,
            "applied",
            last_contact=now.date(),
            next_step=f"Follow up around {follow_up:%a %d %b} if there's no reply",
            notes=f"From Job Alerts ({job.get('location') or 'location not listed'})",
            url=job.get("url", ""),
        )
        lines = [f"Marked applied: {app.company}, {app.role}."]
        if self.reminders is not None:
            await self.reminders.add(
                f"Follow up on your {app.company} application ({app.role}), applied "
                f"{now:%d %b}. Ask me to draft the follow-up.",
                follow_up,
            )
            lines.append(f"I'll remind you to follow up on {follow_up:%a %d %b} at 9:00.")
        return ButtonResult(
            toast="Marked applied", messages=[Outgoing("\n".join(lines))], only_this_row=True
        )

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

    async def _skills(self, thread_id: str, args: str) -> list[Outgoing]:
        return self._one(skills_text(self.skills))

    async def _skill(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.tasks is None:
            return self._one("Background tasks aren't set up.")
        return self._one(await start_skill(self.tasks, self.skills, args, self.task_max_attempts))

    async def _jobs(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.jobs is None:
            return self._one("The job tracker isn't set up.")
        return self._one(await jobs_text(self.jobs))

    async def _status(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.status is None:
            return self._one("Status isn't available.")
        return self._one(await self.status())

    async def _reminders(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.reminders is None or self.clock is None:
            return self._one("Reminders aren't set up.")
        return self._one(await reminders_text(self.reminders, self.clock))

    async def _scan(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.scan_runner is None:
            return self._one("Job Alerts isn't set up (JOB_ALERTS_URL is empty).")
        return self._one(self.scan_runner.start())

    async def _dashboard(self, thread_id: str, args: str) -> list[Outgoing]:
        if self.job_alerts is None:
            return self._one("Job Alerts isn't set up (JOB_ALERTS_URL is empty).")
        try:
            status = await self.job_alerts.status()
        except JobAlertsError as exc:
            return self._one(str(exc))
        return self._one(status_text(status, self.job_alerts.base_url))

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
    "status",
    "reminders",
    "scan",
    "dashboard",
    "skills",
    "skill",
    "jobs",
    "brief",
    "task",
    "tasks",
    "cancel",
    "contacts",
    "memories",
    "forget",
    "unmute",
)
