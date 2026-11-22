"""What background tasks tell you on Telegram, and the /tasks and /cancel commands."""

from datetime import timedelta

from fuse.proactive.notify import Button, Notifier, split_message
from fuse.tasks.queue import Task, TaskStore

TASK_BUTTON = "tk"
KEY_CHARS = 16  # enough of the action key to match it; Telegram allows 64 bytes per button


def task_buttons(task_id: int, approval: dict) -> list[Button]:
    key = approval["action_key"][:KEY_CHARS]
    return [
        Button("Approve", f"{TASK_BUTTON}:approve:{task_id}:{key}"),
        Button("Reject", f"{TASK_BUTTON}:reject:{task_id}:{key}"),
    ]


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
    approved = choice in ("approve", "send", "draft")
    decision: dict = {"approved": approved}
    if choice in ("send", "draft"):
        decision["mode"] = choice
    if not await tasks.submit_decision(task_id, full_key, decision):
        return "This is no longer waiting for approval."
    return f"{'Approved' if approved else 'Rejected'}. Task #{task_id} continues."


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
        text = f"Task #{task.id} would like to do this:\n\n{approval['preview']}"
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
