"""The one entry point channels use to talk to the agent.

Channels call send() for a new message and resume() for an approval decision, and render
the AgentReply they get back. Telegram today; Slack and WhatsApp later.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

log = logging.getLogger("fuse.agent")


@dataclass
class AgentReply:
    kind: Literal["text", "approval"]
    text: str
    approval: dict[str, Any] = field(default_factory=dict)  # set when kind == "approval"
    hit_limit: bool = False  # the step budget ran out before the agent finished


class AgentRunner:
    def __init__(self, graph: CompiledStateGraph, max_steps: int = 12) -> None:
        self._graph = graph
        self._max_steps = max_steps

    def _config(self, thread_id: str, callbacks: list | None = None) -> dict[str, Any]:
        # thread_id keeps conversations apart; recursion_limit is the per-turn step budget.
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._max_steps,
            "run_name": "fuse-turn",  # how each turn is named in traces
            "metadata": {"thread_id": thread_id},  # lets LangSmith group turns by conversation
        }
        if callbacks:
            config["callbacks"] = callbacks  # e.g. a background task's token budget
        return config

    async def pending_approval(self, thread_id: str) -> dict[str, Any] | None:
        """The approval this thread is paused on, if any. Read from the checkpoint."""
        snapshot = await self._graph.aget_state(self._config(thread_id))
        return dict(snapshot.interrupts[0].value) if snapshot.interrupts else None

    async def send(self, thread_id: str, text: str, callbacks: list | None = None) -> AgentReply:
        pending = await self.pending_approval(thread_id)
        if pending:
            return AgentReply("approval", "Please approve or reject this first:", pending)
        return await self._run(thread_id, {"messages": [HumanMessage(text)]}, callbacks)

    async def resume(self, thread_id: str, approved: bool, mode: str | None = None) -> AgentReply:
        """Continue after your decision. `mode` picks between options the tool offered,
        e.g. "send" or "draft" for an email."""
        decision: dict[str, Any] = {"approved": approved}
        if mode:
            decision["mode"] = mode
        return await self.resume_with(thread_id, decision)

    async def resume_with(
        self, thread_id: str, decision: dict[str, Any], callbacks: list | None = None
    ) -> AgentReply:
        if not await self.pending_approval(thread_id):
            return AgentReply("text", "Nothing is waiting for approval.")
        return await self._run(thread_id, Command(resume=decision), callbacks)

    async def has_history(self, thread_id: str) -> bool:
        snapshot = await self._graph.aget_state(self._config(thread_id))
        return bool(snapshot.values.get("messages"))

    async def continue_run(self, thread_id: str, callbacks: list | None = None) -> AgentReply:
        """Pick up where a crashed run stopped, from its last checkpoint. LangGraph saves
        state after every step, so only the unfinished step runs again."""
        pending = await self.pending_approval(thread_id)
        if pending:
            return AgentReply("approval", "I'd like to do this:", pending)
        return await self._run(thread_id, None, callbacks)

    async def add_note(self, thread_id: str, text: str) -> bool:
        """Put a message Fuse sent on its own (a brief, an alert) into the conversation, so
        you can reply to it naturally: "draft a reply to the second one".

        Skipped while an approval is waiting, so a paused action is never disturbed.
        """
        if await self.pending_approval(thread_id):
            return False
        await self._graph.aupdate_state(
            self._config(thread_id), {"messages": [AIMessage(text)]}, as_node="agent"
        )
        return True

    async def _run(
        self, thread_id: str, graph_input: Any, callbacks: list | None = None
    ) -> AgentReply:
        try:
            result = await self._graph.ainvoke(graph_input, self._config(thread_id, callbacks))
        except GraphRecursionError:
            log.warning("step budget exceeded in %s", thread_id)
            return AgentReply(
                "text", "I hit my step limit on that one. Could you narrow it down?", hit_limit=True
            )

        if result.get("__interrupt__"):
            payload = dict(result["__interrupt__"][0].value)
            log.info("waiting for approval of %s in %s", payload.get("tool"), thread_id)
            return AgentReply("approval", "I'd like to do this:", payload)

        content = result["messages"][-1].content
        if isinstance(content, list):  # some providers return a list of content blocks
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        return AgentReply("text", content or "(no reply)")
