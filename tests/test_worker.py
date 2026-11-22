import asyncio
from datetime import timedelta
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from fuse.actions import InMemoryActionStore
from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.tasks.queue import InMemoryTaskStore
from fuse.tasks.worker import Worker
from fuse.tools.basic import make_basic_tools
from fuse.tools.drafts import LocalDraftSink, make_draft_tools
from tests.conftest import ScriptedModel, tool_call


class FlakyModel(ScriptedModel):
    """A scripted model whose script may contain exceptions to raise."""

    script: list[Any]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.position < len(self.script) and isinstance(self.script[self.position], Exception):
            error = self.script[self.position]
            self.position += 1
            self.calls.append(list(messages))
            raise error
        return super()._generate(messages, stop, run_manager, **kwargs)


class Recorder:
    def __init__(self):
        self.events = []

    async def started(self, task):
        self.events.append(("started", task.id))

    async def needs_approval(self, task, approval):
        self.events.append(("approval", approval["tool"]))

    async def finished(self, task, result):
        self.events.append(("finished", result))

    async def failed(self, task, error):
        self.events.append(("failed", error))

    async def retrying(self, task, error, delay):
        self.events.append(("retrying", delay))

    async def cancelled(self, task):
        self.events.append(("cancelled", task.id))


@tool
async def slow_lookup() -> str:
    """Takes a long time."""
    await asyncio.sleep(30)
    return "done"


def setup(script, tools=None, max_steps=20, **worker_kwargs):
    model = FlakyModel(script=script)
    tools = tools if tools is not None else make_basic_tools("UTC")
    runner = AgentRunner(build_graph(model, tools, InMemorySaver()), max_steps=max_steps)
    tasks, recorder = InMemoryTaskStore(), Recorder()
    worker = Worker(tasks, runner, recorder, worker_id="w1", **worker_kwargs)
    return worker, tasks, recorder, model


async def test_nothing_to_do():
    worker, *_ = setup([])
    assert await worker.run_one() is False


async def test_a_task_runs_to_completion():
    worker, tasks, recorder, model = setup(
        [tool_call("get_current_time", {"timezone": "UTC"}), AIMessage("Here's your summary.")]
    )
    task = await tasks.create("summarise my week")
    assert await worker.run_one()
    done = await tasks.get(task.id)
    assert (done.status, done.result) == ("done", "Here's your summary.")
    assert recorder.events == [("started", task.id), ("finished", "Here's your summary.")]
    assert "summarise my week" in model.calls[0][-1].content


async def test_rate_limit_is_retried_later_and_continues_from_the_checkpoint():
    worker, tasks, recorder, model = setup(
        [
            tool_call("get_current_time", {"timezone": "UTC"}),
            httpx.ConnectError("provider unreachable"),
            AIMessage("Finished after the retry."),
        ]
    )
    task = await tasks.create("goal")
    await worker.run_one()
    assert (await tasks.get(task.id)).status == "queued"
    kind, delay = recorder.events[-1]
    assert kind == "retrying" and timedelta(seconds=30) <= delay <= timedelta(seconds=36)
    assert await worker.run_one() is False  # waits for the retry delay

    tasks.now = lambda: _later(minutes=1)
    await worker.run_one()
    assert (await tasks.get(task.id)).result == "Finished after the retry."
    state = await worker.runner._graph.aget_state({"configurable": {"thread_id": task.thread_id}})
    humans = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert len(humans) == 1  # continued, not started over
    tools_run = [m for m in state.values["messages"] if m.type == "tool"]
    assert len(tools_run) == 1  # the finished step wasn't repeated


async def test_retries_run_out():
    worker, tasks, recorder, _ = setup([httpx.ConnectError("down")] * 3)
    task = await tasks.create("goal", max_attempts=1)
    await worker.run_one()
    assert (await tasks.get(task.id)).status == "failed"
    assert "gave up" in recorder.events[-1][1]


async def test_unexpected_error_fails_the_task_with_a_readable_reason():
    worker, tasks, recorder, _ = setup([ValueError("bad")])
    task = await tasks.create("goal")
    await worker.run_one()
    assert (await tasks.get(task.id)).status == "failed"
    assert recorder.events[-1] == ("failed", "Something went wrong: ValueError.")


async def test_task_pauses_for_approval_and_resumes_after_your_decision():
    actions, sink = InMemoryActionStore(), LocalDraftSink()
    draft = {"to": "pat@acme.com", "subject": "Re: Friday", "body": "Monday works."}
    worker, tasks, recorder, _ = setup(
        [tool_call("draft_email_reply", draft), AIMessage("Drafted the reply to Pat.")],
        tools=make_draft_tools(actions, sink),
    )
    task = await tasks.create("reply to Pat")
    await worker.run_one()
    waiting = await tasks.get(task.id)
    assert waiting.status == "waiting_approval"
    assert recorder.events[-1] == ("approval", "draft_email_reply")
    assert sink.drafts == []

    await tasks.submit_decision(task.id, waiting.pending["action_key"], {"approved": True})
    await worker.run_one()
    assert (await tasks.get(task.id)).result == "Drafted the reply to Pat."
    assert len(sink.drafts) == 1


async def test_cancel_stops_a_running_task():
    worker, tasks, recorder, _ = setup(
        [tool_call("slow_lookup", {})], tools=[slow_lookup], lease=timedelta(seconds=0.3)
    )
    task = await tasks.create("slow goal")

    async def cancel_soon():
        await asyncio.sleep(0.1)
        await tasks.request_cancel(task.id)

    await asyncio.gather(worker.run_one(), cancel_soon())
    assert (await tasks.get(task.id)).status == "cancelled"
    assert recorder.events[-1] == ("cancelled", task.id)


async def test_a_task_that_runs_too_long_is_stopped():
    worker, tasks, recorder, _ = setup(
        [tool_call("slow_lookup", {})], tools=[slow_lookup], timeout=timedelta(seconds=0.2)
    )
    task = await tasks.create("slow goal")
    await worker.run_one()
    assert (await tasks.get(task.id)).status == "failed"
    assert "longer than" in recorder.events[-1][1]


async def test_step_budget():
    loop = [tool_call("get_current_time", {"timezone": "UTC"}, f"c{i}") for i in range(10)]
    worker, tasks, recorder, _ = setup(loop, max_steps=4)
    task = await tasks.create("goal")
    await worker.run_one()
    assert (await tasks.get(task.id)).status == "failed"
    assert "budget" in recorder.events[-1][1]


def _later(**delta):
    from datetime import UTC, datetime

    return datetime.now(UTC) + timedelta(**delta)


async def test_task_approval_buttons_resume_the_task_end_to_end():
    """Worker pauses -> buttons on Telegram -> your tap (handled by the bot) -> worker resumes."""
    from fuse.proactive.notify import RecordingNotifier
    from fuse.tasks.report import TelegramTaskReporter, decide_task_approval

    actions, sink, notifier = InMemoryActionStore(), LocalDraftSink(), RecordingNotifier()
    draft = {"to": "pat@acme.com", "subject": "Re: Friday", "body": "Monday works."}
    model = FlakyModel(script=[tool_call("draft_email_reply", draft), AIMessage("Done.")])
    runner = AgentRunner(build_graph(model, make_draft_tools(actions, sink), InMemorySaver()))
    tasks = InMemoryTaskStore()
    worker = Worker(tasks, runner, TelegramTaskReporter(notifier), worker_id="w1")
    task = await tasks.create("reply to Pat")

    await worker.run_one()
    text, buttons = notifier.sent[-1]
    assert "To: pat@acme.com" in text
    assert [b.label for b in buttons] == ["Approve", "Reject"]

    assert await decide_task_approval(tasks, "tk:approve:999:abc") == (
        "This is no longer waiting for approval."
    )
    reply = await decide_task_approval(tasks, buttons[0].data)
    assert reply == f"Approved. Task #{task.id} continues."
    assert "no longer waiting" in await decide_task_approval(tasks, buttons[1].data)  # once

    await worker.run_one()
    assert len(sink.drafts) == 1
    assert notifier.texts[-1].endswith("Done.")


async def test_rejecting_a_task_action_lets_the_task_finish_without_it():
    from fuse.proactive.notify import RecordingNotifier
    from fuse.tasks.report import TelegramTaskReporter, decide_task_approval

    actions, sink, notifier = InMemoryActionStore(), LocalDraftSink(), RecordingNotifier()
    draft = {"to": "pat@acme.com", "subject": "Hi", "body": "x"}
    model = FlakyModel(script=[tool_call("draft_email_reply", draft), AIMessage("Skipped it.")])
    runner = AgentRunner(build_graph(model, make_draft_tools(actions, sink), InMemorySaver()))
    tasks = InMemoryTaskStore()
    worker = Worker(tasks, runner, TelegramTaskReporter(notifier), worker_id="w1")
    task = await tasks.create("reply")
    await worker.run_one()
    await decide_task_approval(tasks, notifier.sent[-1][1][1].data)
    await worker.run_one()
    assert sink.drafts == [] and (await tasks.get(task.id)).status == "done"


def _usage(tokens: int) -> dict:
    return {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens}


async def test_token_budget_stops_an_expensive_task():
    calls = [
        AIMessage(
            "",
            tool_calls=[{"name": "get_current_time", "args": {"timezone": "UTC"}, "id": f"c{i}"}],
            usage_metadata=_usage(4000),
        )
        for i in range(5)
    ]
    worker, tasks, recorder, model = setup(calls, max_tokens=10_000)
    task = await tasks.create("expensive goal")
    await worker.run_one()
    assert (await tasks.get(task.id)).status == "failed"
    assert "12,000 tokens" in recorder.events[-1][1]
    assert model.position == 3  # stopped at the call that went over


async def test_a_task_within_budget_finishes():
    worker, tasks, _, _ = setup([AIMessage("ok", usage_metadata=_usage(500))], max_tokens=10_000)
    task = await tasks.create("cheap goal")
    await worker.run_one()
    assert (await tasks.get(task.id)).status == "done"
