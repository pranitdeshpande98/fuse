"""What background tasks tell you on Telegram, and the /tasks and /cancel commands."""

from datetime import timedelta

from fuse.approval_ui import after_allowing, approval_buttons, approval_text
from fuse.contacts import ContactStore
from fuse.proactive.notify import Button, Notifier, split_message
from fuse.tasks.queue import Task, TaskStore

TASK_BUTTON = "tk"
KEY_CHARS = 16  # enough of the action key to match it; Telegram allows 64 bytes per button


def task_buttons(task_id: int, approval: dict) -> list[list[Button]]:
    key = approval["action_key"][:KEY_CHARS]
    return approval_buttons(approval, lambda verb: f"{TASK_BUTTON}:{verb}:{task_id}:{key}")


def parse_task_button(data: str) -> tuple[str, int, str] | None:
    """'tk:approve:7:ab12…' -> ('approve', 7, 'ab12…'), or None if it isn't one."""
    parts = data.split(":")
    if len(parts) != 4 or parts[0] != TASK_BUTTON or not parts[2].isdigit():
        return None
    return parts[1], int(parts[2]), parts[3]


async def decide_task_approval(tasks: TaskStore, data: str) -> str:
    """Your tap on a task's approval button. The worker picks the task up again within
    a couple of seconds and continues from where it paused."""
    parsed = parse_task_button(data)
    if parsed is None:
        return "That button isn't valid any more."
    choice, task_id, key_prefix = parsed
    task = await tasks.get(task_id)
    pending = (task.pending or {}) if task else {}
    full_key = pending.get("action_key", "")
    if not task or task.status != "waiting_approval" or not full_key.startswith(key_prefix):
        return "This is no longer waiting for approval."
    if choice not in ("approve", "send", "reject"):
        return "That button isn't valid any more."
    approved = choice in ("approve", "send")
    decision: dict = {"approved": approved}
    if choice == "send":
        decision["mode"] = "send"
    if not await tasks.submit_decision(task_id, full_key, decision):
        return "This is no longer waiting for approval."
    verb = "Sending" if choice == "send" else ("Approved" if approved else "Rejected")
    return f"{verb}. Task #{task_id} continues."


async def allow_task_recipients(
    tasks: TaskStore, contacts: ContactStore, data: str
) -> tuple[str, list[list[Button]] | None]:
    """ "Always allow" on a task's email: add the blocked recipients to your contacts and
    return the new buttons, now with Send."""
    parsed = parse_task_button(data)
    task = await tasks.get(parsed[1]) if parsed else None
    pending = (task.pending or {}) if task else {}
    if (
        not parsed
        or task is None
        or task.status != "waiting_approval"
        or not pending.get("action_key", "").startswith(parsed[2])
    ):
        return "This is no longer waiting for approval.", None
    for address in pending.get("blocked") or []:
        await contacts.add(address, source="always allow")
    allowed = ", ".join(pending.get("blocked") or [])
    return f"Added {allowed} to your contacts.", task_buttons(task.id, after_allowing(pending))


STATUS_LABEL = {
    "queued": "queued",
    "running": "running",
    "waiting_approval": "waiting for your approval",
    "done": "done",
    "failed": "failed",
    "cancelled": "cancelled",
}


def short(goal: str, limit: int = 80) -> str:
    goal = " ".join(goal.split())
    return goal if len(goal) <= limit else goal[: limit - 1] + "…"


class TelegramTaskReporter:
    def __init__(self, notifier: Notifier) -> None:
        self._notifier = notifier

    async def started(self, task: Task) -> None:
        await self._notifier.send(f"Task #{task.id} started: {short(task.goal)}")

    async def progress(self, task_id: int, text: str) -> None:
        await self._notifier.send(f"Task #{task_id}: {text}")

    async def needs_approval(self, task: Task, approval: dict) -> None:
        text = f"Task #{task.id} would like to do this:\n\n{approval_text(approval)}"
        await self._notifier.send(text, task_buttons(task.id, approval))

    async def finished(self, task: Task, result: str) -> None:
        parts = split_message(f"Task #{task.id} done: {short(task.goal)}\n\n{result}")
        for part in parts:
            await self._notifier.send(part)

    async def failed(self, task: Task, error: str) -> None:
        await self._notifier.send(f"Task #{task.id} failed: {error}")

    async def retrying(self, task: Task, error: str, delay: timedelta) -> None:
        seconds = int(delay.total_seconds())
        await self._notifier.send(
            f"Task #{task.id} hit a temporary problem; trying again in {seconds} s."
        )

    async def cancelled(self, task: Task) -> None:
        await self._notifier.send(f"Task #{task.id} cancelled.")


async def tasks_text(tasks: TaskStore, limit: int = 10) -> str:
    recent = await tasks.recent(limit)
    if not recent:
        return "No background tasks yet. Start one with /task <what to do>."
    lines = ["Recent tasks:"]
    for t in recent:
        lines.append(f"#{t.id} [{STATUS_LABEL[t.status]}] {short(t.goal, 60)}")
    lines.append('\nDetails: ask me "what did task 7 find?" · Stop one: /cancel <id>')
    return "\n".join(lines)


async def cancel_text(tasks: TaskStore, argument: str) -> str:
    argument = argument.strip().lstrip("#")
    if not argument.isdigit():
        return "Usage: /cancel <task id>. See your tasks with /tasks."
    status = await tasks.request_cancel(int(argument))
    if status is None:
        return f"There's no task #{argument}."
    if status == "running":
        return f"Stopping task #{argument}; it ends after its current step."
    if status == "cancelled":
        return f"Task #{argument} is cancelled."
    return f"Task #{argument} had already finished ({STATUS_LABEL[status]})."
