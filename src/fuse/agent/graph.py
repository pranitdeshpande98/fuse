"""The agent loop: model -> tools -> model -> ... until the model answers without a tool call."""

from collections.abc import Sequence

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from fuse.agent.prompts import SYSTEM_PROMPT


def build_graph(
    model: Runnable,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    async def call_model(state: MessagesState) -> dict:
        response = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else end
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
