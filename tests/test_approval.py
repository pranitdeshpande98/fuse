"""The approval gate end to end, with a scripted model and in-memory storage."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.actions import InMemoryActionStore
from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.tools.basic import make_basic_tools
from fuse.tools.drafts import DraftError, LocalDraftSink, make_draft_tools
from tests.conftest import ScriptedModel, tool_call

DRAFT = {"to": "pat@example.com", "subject": "Re: Proposal", "body": "Friday works."}


def make_runner(script: list[AIMessage]):
    store, sink = InMemoryActionStore(), LocalDraftSink()
    tools = [*make_basic_tools("UTC"), *make_draft_tools(store, sink)]
    model = ScriptedModel(script=script)
    runner = AgentRunner(build_graph(model, tools, InMemorySaver()))
    return runner, store, sink, model


async def test_write_tool_waits_for_approval_and_then_runs_once():
    runner, store, sink, _ = make_runner(
        [tool_call("draft_email_reply", DRAFT), AIMessage(content="Draft saved.")]
    )

    reply = await runner.send("t1", "Reply to Pat that Friday works")
    assert reply.kind == "approval"
    assert "pat@example.com" in reply.approval["preview"]
    assert sink.drafts == []  # nothing happened before approval
    key = reply.approval["action_key"]
    assert (await store.get(key)).status == "pending"

    reply = await runner.resume("t1", approved=True)
    assert reply.text == "Draft saved."
    assert len(sink.drafts) == 1
    assert (await store.get(key)).status == "executed"


async def test_rejecting_does_nothing_and_tells_the_model():
    runner, store, sink, model = make_runner(
        [tool_call("draft_email_reply", DRAFT), AIMessage(content="OK, I won't.")]
    )
    key = (await runner.send("t1", "Reply to Pat")).approval["action_key"]

    reply = await runner.resume("t1", approved=False)
    assert reply.text == "OK, I won't."
    assert sink.drafts == []
    assert (await store.get(key)).status == "rejected"
    assert "rejected" in model.calls[-1][-1].content  # the model was told


async def test_new_messages_wait_until_you_decide():
    runner, _, _, _ = make_runner([tool_call("draft_email_reply", DRAFT)])
    await runner.send("t1", "Reply to Pat")
    reply = await runner.send("t1", "Also, what time is it?")
    assert reply.kind == "approval"
    assert reply.text.startswith("Please approve or reject")


async def test_resume_with_nothing_pending():
    runner, _, _, _ = make_runner([])
    assert (await runner.resume("t1", approved=True)).text == "Nothing is waiting for approval."


async def test_read_tools_never_ask_for_approval():
    runner, _, _, _ = make_runner(
        [tool_call("get_current_time", {"timezone": "UTC"}), AIMessage(content="Noon.")]
    )
    reply = await runner.send("t1", "What time is it?")
    assert reply.kind == "text"


class FailingSink:
    async def save_draft(self, to, subject, body):
        raise DraftError("Google is not connected. Run: uv run python -m fuse.google.auth")


async def test_a_failed_draft_is_reported_and_recorded():
    store = InMemoryActionStore()
    model = ScriptedModel(
        script=[tool_call("draft_email_reply", DRAFT), AIMessage(content="Sorry, that failed.")]
    )
    runner = AgentRunner(
        build_graph(model, make_draft_tools(store, FailingSink()), InMemorySaver())
    )
    key = (await runner.send("t1", "Reply to Pat")).approval["action_key"]

    reply = await runner.resume("t1", approved=True)
    assert reply.text == "Sorry, that failed."
    assert "could not be saved" in model.calls[-1][-1].content  # the model was told why
    assert (await store.get(key)).status == "failed"
