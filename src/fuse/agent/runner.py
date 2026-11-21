"""The one entry point channels use to talk to the agent.

Telegram calls `send()` today; Slack and WhatsApp will call the same method later.
"""

import logging

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

log = logging.getLogger("fuse.agent")


class AgentRunner:
    def __init__(self, graph: CompiledStateGraph, max_steps: int = 12) -> None:
        self._graph = graph
        self._max_steps = max_steps

    async def send(self, thread_id: str, text: str) -> str:
        # thread_id keeps each conversation separate; recursion_limit is the per-turn step budget.
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self._max_steps}
        try:
            result = await self._graph.ainvoke({"messages": [HumanMessage(text)]}, config)
        except GraphRecursionError:
            log.warning("step budget exceeded in %s", thread_id)
            return "I hit my step limit on that one. Could you narrow it down?"
        content = result["messages"][-1].content
        if isinstance(content, list):  # some providers return a list of content blocks
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        return content or "(no reply)"
