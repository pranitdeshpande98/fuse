"""The agent loop: model -> tools -> model -> ... until the model answers without a tool call."""

from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
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


def recent_messages(messages: Sequence[AnyMessage], limit: int) -> list[AnyMessage]:
    """The last `limit` messages the model should see, starting at a user message.

    The full history stays in the checkpoint; only what's sent to the model is trimmed.
    trim_messages keeps tool calls together with their results (a model API rejects a tool
    result without its call). If even the current turn is longer than `limit`, it would
    return nothing, so in that case the whole current turn is kept.

    Messages Fuse sent on its own (a morning brief, an alert) sit just before your reply to
    them, so they are kept too when there is room: "draft a reply to the second one" needs
    the brief it refers to.
    """
    messages = list(messages)
    trimmed = trim_messages(
        messages, strategy="last", token_counter=len, max_tokens=limit, start_on="human"
    )
    if not trimmed:
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                return messages[i:]
        return messages
    start = len(messages) - len(trimmed)
    notes: list[AnyMessage] = []
    j = start - 1
    while j >= 0 and len(trimmed) + len(notes) < limit and _is_note(messages[j]):
        notes.insert(0, messages[j])
        j -= 1
    return [*notes, *trimmed]


def _is_note(message: AnyMessage) -> bool:
    """A plain assistant message: no tool calls, so it's safe to start the window with."""
    return isinstance(message, AIMessage) and not message.tool_calls


def build_graph(
    model: Runnable,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver,
    memory: MemoryStore | None = None,
    max_history_messages: int = 30,
    system_prompt: str = SYSTEM_PROMPT,
) -> CompiledStateGraph:
    async def call_model(state: MessagesState) -> dict:
        system = system_prompt
        if memory is not None:
            notes = await memory_context(memory, latest_user_text(state["messages"]))
            if notes:
                system = f"{system_prompt}\n{notes}"
        history = recent_messages(state["messages"], max_history_messages)
        response = await model.ainvoke([SystemMessage(system), *history])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else end
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
