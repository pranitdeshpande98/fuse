from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.tools.basic import make_basic_tools
from tests.conftest import ScriptedModel, tool_call


def make_runner(script: list[AIMessage], max_steps: int = 12) -> tuple[AgentRunner, ScriptedModel]:
    model = ScriptedModel(script=script)
    graph = build_graph(model, make_basic_tools("UTC"), InMemorySaver())
    return AgentRunner(graph, max_steps=max_steps), model


async def test_plain_answer():
    runner, _ = make_runner([AIMessage(content="Hello!")])
    assert (await runner.send("t1", "hi")).text == "Hello!"


async def test_tool_is_called_and_result_goes_back_to_the_model():
    runner, model = make_runner(
        [tool_call("get_current_time", {"timezone": "UTC"}), AIMessage(content="It's noon.")]
    )

    assert (await runner.send("t1", "what time is it?")).text == "It's noon."
    tool_result = model.calls[1][-1]  # the last message the model saw on its second call
    assert tool_result.type == "tool"
    assert "UTC" in tool_result.content


async def test_conversation_is_remembered_within_a_thread():
    runner, model = make_runner(
        [AIMessage(content="Nice to meet you."), AIMessage(content="Pranit")]
    )
    await runner.send("t1", "My name is Pranit")
    await runner.send("t1", "What's my name?")
    seen = [m.content for m in model.calls[1]]
    assert "My name is Pranit" in seen


async def test_threads_are_kept_separate():
    runner, model = make_runner([AIMessage(content="ok"), AIMessage(content="ok")])
    await runner.send("t1", "secret for thread one")
    await runner.send("t2", "hello")
    seen = [m.content for m in model.calls[1]]
    assert "secret for thread one" not in seen


async def test_step_budget_stops_a_looping_model():
    looping = [tool_call("get_current_time", {}, call_id=f"c{i}") for i in range(50)]
    runner, _ = make_runner(looping, max_steps=4)

    assert "step limit" in (await runner.send("t1", "loop forever")).text
