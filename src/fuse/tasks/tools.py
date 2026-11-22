"""Agent tools for background tasks.

`start_background_task` is for your chat: the agent hands off work that would take many
steps. `report_progress` is for the task's own agent, to send you a one-line update.
"""

from collections.abc import Awaitable, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from fuse.tasks.queue import FINISHED, TaskStore

MAX_ACTIVE_TASKS = 3  # a runaway model can't fill the queue
MAX_PROGRESS_UPDATES = 5


async def start_task(tasks: TaskStore, goal: str, max_attempts: int = 3) -> str:
    """Shared by the tool and the /task command."""
    goal = goal.strip()
    if not goal:
        return "Tell me what the task should do, e.g. /task list every recruiter email."
    active = [t for t in await tasks.recent(50) if t.status not in FINISHED]
    if len(active) >= MAX_ACTIVE_TASKS:
        ids = ", ".join(f"#{t.id}" for t in active)
        return (
            f"There are already {len(active)} tasks running or waiting ({ids}). "
            "Wait for one to finish, or stop one with /cancel <id>."
        )
    task = await tasks.create(goal, max_attempts=max_attempts)
    return f"Started background task #{task.id}. I'll message you when it's done."


def make_task_tools(tasks: TaskStore, max_attempts: int = 3) -> list[BaseTool]:
    @tool
    async def start_background_task(goal: str) -> str:
        """Hand off work needing many steps (many emails, several web searches, a
        report) so the chat stays free; not for quick questions. Write the goal as a
        complete instruction: the task can't see this chat. The result arrives later."""
        return await start_task(tasks, goal, max_attempts)

    @tool
    async def check_task(task_id: int) -> str:
        """The status and result of a background task, e.g. to follow up on "task 7"."""
        task = await tasks.get(task_id)
        if task is None:
            return f"There is no task #{task_id}."
        lines = [f"Task #{task.id}: {task.status}", f"Goal: {task.goal}"]
        if task.result:
            lines.append(f"Result:\n{task.result}")
        if task.error:
            lines.append(f"Problem: {task.error}")
        return "\n".join(lines)

    return [start_background_task, check_task]


def make_progress_tool(
    send: Callable[[int, str], Awaitable[None]],
) -> BaseTool:
    """`send(task_id, text)` delivers the update; the task ID comes from the thread."""
    sent: dict[str, int] = {}

    @tool
    async def report_progress(message: str, config: RunnableConfig) -> str:
        """Send the user a one-line progress update about this background task."""
        thread_id = str(config["configurable"]["thread_id"])
        if not thread_id.startswith("task:"):
            return "Progress updates are only for background tasks."
        if sent.get(thread_id, 0) >= MAX_PROGRESS_UPDATES:
            return "Enough updates for this task; just finish it."
        sent[thread_id] = sent.get(thread_id, 0) + 1
        await send(int(thread_id.removeprefix("task:")), message.strip()[:300])
        return "Update sent."

    return report_progress
