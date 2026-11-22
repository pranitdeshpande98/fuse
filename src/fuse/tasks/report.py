"""What background tasks tell you on Telegram, and the /tasks and /cancel commands."""

from datetime import timedelta

from fuse.proactive.notify import Notifier, split_message
from fuse.tasks.queue import Task, TaskStore

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
        # Replaced in the next step with Approve / Reject buttons.
        await self._notifier.send(f"Task #{task.id} needs your approval:\n\n{approval['preview']}")

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
