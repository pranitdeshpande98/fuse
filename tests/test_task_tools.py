from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.tasks.queue import InMemoryTaskStore
from fuse.tasks.tools import MAX_PROGRESS_UPDATES, make_progress_tool, make_task_tools, start_task
from tests.conftest import ScriptedModel, tool_call


async def test_chat_agent_hands_off_a_big_job():
    tasks = InMemoryTaskStore()
    goal = "Go through last month's recruiter emails and list company, role and status"
    model = ScriptedModel(
        script=[tool_call("start_background_task", {"goal": goal}), AIMessage("Started #1.")]
    )
    runner = AgentRunner(build_graph(model, make_task_tools(tasks), InMemorySaver()))
    reply = await runner.send("telegram:42", "make me a list of all my recruiter emails")
    assert reply.text == "Started #1."
    (task,) = await tasks.recent()
    assert (task.goal, task.status) == (goal, "queued")
    assert "Started background task #1" in model.calls[1][-1].content


async def test_too_many_active_tasks_are_refused():
    tasks = InMemoryTaskStore()
    for i in range(3):
        await start_task(tasks, f"task {i}")
    text = await start_task(tasks, "one more")
    assert "already 3 tasks" in text and "/cancel" in text
    assert len(await tasks.recent()) == 3


async def test_empty_goal_explains_usage():
    assert "/task" in await start_task(InMemoryTaskStore(), "  ")


async def test_progress_updates_are_capped_and_only_for_tasks():
    sent = []

    async def send(task_id, text):
        sent.append((task_id, text))

    progress = make_progress_tool(send)
    config = {"configurable": {"thread_id": "task:7"}}
    for i in range(MAX_PROGRESS_UPDATES + 2):
        await progress.ainvoke({"message": f"step {i}"}, config)
    assert len(sent) == MAX_PROGRESS_UPDATES and sent[0] == (7, "step 0")
    chat = {"configurable": {"thread_id": "telegram:42"}}
    assert "only for background" in await progress.ainvoke({"message": "x"}, chat)
