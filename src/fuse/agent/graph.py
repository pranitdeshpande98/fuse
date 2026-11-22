"""The agent loop: model -> tools -> model -> ... until the model answers without a tool call."""

from collections.abc import Sequence

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from fuse.agent.prompts import SYSTEM_PROMPT
from fuse.memory.context import memory_context
from fuse.memory.store import MemoryStore


def latest_user_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else ""
    return ""


def build_graph(
    model: Runnable,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver,
    memory: MemoryStore | None = None,
) -> CompiledStateGraph:
    async def call_model(state: MessagesState) -> dict:
        system = SYSTEM_PROMPT
        if memory is not None:
            notes = await memory_context(memory, latest_user_text(state["messages"]))
            if notes:
                system = f"{SYSTEM_PROMPT}\n{notes}"
        response = await model.ainvoke([SystemMessage(system), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else end
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
