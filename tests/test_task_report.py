from datetime import timedelta

from fuse.proactive.notify import RecordingNotifier, split_message
from fuse.tasks.queue import InMemoryTaskStore
from fuse.tasks.report import TelegramTaskReporter, cancel_text, tasks_text
from fuse.tasks.tools import make_task_tools


def test_long_messages_split_at_paragraphs():
    text = "\n\n".join(f"Paragraph {i}: " + "x" * 900 for i in range(10))
    parts = split_message(text, 4000)
    assert all(len(p) <= 4000 for p in parts) and len(parts) == 3
    assert parts[1].startswith("Paragraph")
    assert split_message("short") == ["short"]


async def test_a_long_result_arrives_in_several_messages():
    tasks, notifier = InMemoryTaskStore(), RecordingNotifier()
    task = await tasks.create("big report")
    await TelegramTaskReporter(notifier).finished(task, "line\n" * 2000)
    assert len(notifier.texts) == 3
    assert notifier.texts[0].startswith(f"Task #{task.id} done: big report")


async def test_tasks_and_cancel_commands():
    tasks = InMemoryTaskStore()
    assert "No background tasks" in await tasks_text(tasks)
    a = await tasks.create("list recruiter emails")
    b = await tasks.create("prepare tomorrow")
    await tasks.claim("w1", timedelta(minutes=2))  # a is running
    text = await tasks_text(tasks)
    assert f"#{a.id} [running]" in text and f"#{b.id} [queued]" in text

    assert "cancelled" in await cancel_text(tasks, str(b.id))
    assert "Stopping task" in await cancel_text(tasks, f"#{a.id}")
    assert "no task #99" in await cancel_text(tasks, "99")
    assert "Usage" in await cancel_text(tasks, "abc")


async def test_chat_can_read_a_finished_tasks_result():
    tasks = InMemoryTaskStore()
    task = await tasks.create("recruiter list")
    await tasks.claim("w1", timedelta(minutes=2))
    await tasks.finish(task.id, "w1", "Acme: SWE II, interview Tuesday")
    check = {t.name: t for t in make_task_tools(tasks)}["check_task"]
    out = await check.ainvoke({"task_id": task.id})
    assert "done" in out and "Acme: SWE II" in out
