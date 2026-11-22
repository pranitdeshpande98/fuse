"""Only recent messages go to the model; the full history stays in the checkpoint."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.agent.graph import build_graph, recent_messages
from fuse.agent.runner import AgentRunner
from tests.conftest import ScriptedModel


def turn(i: int, with_tool: bool = False) -> list:
    messages = [HumanMessage(f"question {i}")]
    if with_tool:
        messages += [
            AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": f"call{i}"}]),
            ToolMessage("result", tool_call_id=f"call{i}"),
        ]
    return [*messages, AIMessage(f"answer {i}")]


def test_short_history_is_untouched():
    messages = turn(0) + turn(1)
    assert recent_messages(messages, 30) == messages


def test_long_history_keeps_the_latest_and_starts_with_the_user():
    messages = [m for i in range(20) for m in turn(i)]
    kept = recent_messages(messages, 6)
    assert len(kept) <= 6
    assert isinstance(kept[0], HumanMessage)
    assert kept[-1].content == "answer 19"


def test_tool_calls_stay_with_their_results():
    messages = [m for i in range(10) for m in turn(i, with_tool=True)]
    kept = recent_messages(messages, 6)
    call_ids = {c["id"] for m in kept if isinstance(m, AIMessage) for c in m.tool_calls}
    result_ids = {m.tool_call_id for m in kept if isinstance(m, ToolMessage)}
    assert call_ids == result_ids


def test_a_current_turn_longer_than_the_limit_is_still_sent_whole():
    messages = turn(0) + turn(1, with_tool=True)  # current turn has 4 messages
    kept = recent_messages(messages, 2)
    assert kept[0].content == "question 1"
    assert len(kept) == 4


async def test_the_model_only_sees_recent_messages_but_nothing_is_lost():
    model = ScriptedModel(script=[AIMessage(content=f"answer {i}") for i in range(10)])
    runner = AgentRunner(build_graph(model, [], InMemorySaver(), max_history_messages=4))
    for i in range(10):
        await runner.send("t1", f"question {i}")

    last_call = model.calls[-1]
    assert isinstance(last_call[0], SystemMessage)
    assert len(last_call) - 1 <= 4  # minus the system prompt
    assert "question 0" not in [m.content for m in last_call]

    state = await runner._graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert len(state.values["messages"]) == 20  # full history kept


def test_a_message_fuse_sent_first_is_kept_with_your_reply():
    brief = AIMessage("Good morning. 1) Pat: Proposal 2) Sam: Budget")
    messages = [*turn(0), brief, HumanMessage("reply to the second one")]
    kept = recent_messages(messages, 30)
    assert kept[-2:] == [brief, messages[-1]]


def test_notes_never_push_the_window_past_the_limit():
    messages = [m for i in range(5) for m in turn(i)] + [AIMessage("alert")] * 5
    messages.append(HumanMessage("ok"))
    assert len(recent_messages(messages, 4)) <= 4
