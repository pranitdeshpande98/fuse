"""The agent loop: model -> tools -> model -> ... until the model answers without a tool call."""

import json
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


def approx_tokens(message: AnyMessage) -> int:
    """A cheap token estimate (about 4 characters per token), good enough for budgeting."""
    size = len(str(message.content))
    for call in getattr(message, "tool_calls", None) or []:
        size += len(json.dumps(call.get("args", {})))
    return size // 4 + 4


def recent_messages(
    messages: Sequence[AnyMessage], limit: int, max_tokens: int | None = None
) -> list[AnyMessage]:
    """The last `limit` messages the model should see, starting at a user message.

    The full history stays in the checkpoint; only what's sent to the model is trimmed.
    trim_messages keeps tool calls together with their results (a model API rejects a tool
    result without its call). If even the current turn is longer than `limit`, it would
    return nothing, so in that case the whole current turn is kept.

    Messages Fuse sent on its own (a morning brief, an alert) sit just before your reply to
    them, so they are kept too when there is room: "draft a reply to the second one" needs
    the brief it refers to.

    `max_tokens` also caps the size: older turns are dropped, whole turns at a time, until
    the history fits. A long old task result (a 20-row table) would otherwise ride along on
    every call, and Groq's free tier allows only 8,000 tokens a minute. The current turn is
    always kept whole.
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
    kept = [*notes, *trimmed]
    if max_tokens is None:
        return kept
    return _fit_tokens(kept, max_tokens)


def _fit_tokens(kept: list[AnyMessage], max_tokens: int) -> list[AnyMessage]:
    """Drop the oldest turns until the history fits in max_tokens (current turn stays)."""
    current = max((i for i, m in enumerate(kept) if isinstance(m, HumanMessage)), default=0)
    # A note Fuse sent just before your latest message belongs with it ("reply to that").
    while current > 0 and _is_note(kept[current - 1]):
        current -= 1
    while current > 0 and sum(approx_tokens(m) for m in kept) > max_tokens:
        # Drop from the front up to the next message a turn can start with.
        cut = next((i for i in range(1, current) if isinstance(kept[i], HumanMessage)), current)
        kept, current = kept[cut:], current - cut
    return kept


def _is_note(message: AnyMessage) -> bool:
    """A plain assistant message: no tool calls, so it's safe to start the window with."""
    return isinstance(message, AIMessage) and not message.tool_calls


def build_graph(
    model: Runnable,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver,
    memory: MemoryStore | None = None,
    max_history_messages: int = 30,
    max_history_tokens: int | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> CompiledStateGraph:
    async def call_model(state: MessagesState) -> dict:
        system = system_prompt
        if memory is not None:
            notes = await memory_context(memory, latest_user_text(state["messages"]))
            if notes:
                system = f"{system_prompt}\n{notes}"
        history = recent_messages(state["messages"], max_history_messages, max_history_tokens)
        response = await model.ainvoke([SystemMessage(system), *history])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else end
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
