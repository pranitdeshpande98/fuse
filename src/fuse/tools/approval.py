"""The approval gate: every tool that changes something goes through here.

How one approved action works:
1. Record the proposed action as "pending" (safe to repeat).
2. Pause the whole graph with interrupt() and show the user a preview.
3. When the user decides, LangGraph re-runs this tool from the top. Step 1 changes nothing
   the second time, and interrupt() now returns the user's decision instead of pausing.
4. Approved: run the action exactly once. Rejected: do nothing and tell the model.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.types import interrupt

from fuse.actions import ActionStore, execute_once, idempotency_key

REJECTED = "The user rejected this action. Do not retry it unless they ask."


async def require_approval(
    *,
    store: ActionStore,
    thread_id: str,
    tool_call_id: str,
    tool_name: str,
    args: dict[str, Any],
    preview: str,
    run: Callable[[], Awaitable[str]],
) -> str:
    key = idempotency_key(thread_id, tool_call_id)

    existing = await store.get(key)
    if existing and existing.status == "executed":
        return existing.result or "Already done."
    if existing and existing.status == "rejected":
        return REJECTED

    await store.record(key, tool_name, args)

    decision = interrupt(
        {"type": "approval", "action_key": key, "tool": tool_name, "preview": preview}
    )

    if not (isinstance(decision, dict) and decision.get("approved") is True):
        await store.set_status(key, "rejected")
        return REJECTED

    await store.set_status(key, "approved")
    return await execute_once(store, key, run)
