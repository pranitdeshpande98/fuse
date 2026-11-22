"""The approval gate: every tool that changes something goes through here.

How one approved action works:
1. Record the proposed action as "pending" (safe to repeat).
2. Pause the whole graph with interrupt() and show the user a preview.
3. When the user decides, LangGraph re-runs this tool from the top. Step 1 changes nothing
   the second time, and interrupt() now returns the user's decision instead of pausing.
4. Approved: run the action exactly once. Rejected: do nothing and tell the model.

Some actions offer a choice. An email can be saved as a draft (`run`) or sent (`run_send`):
the decision says {"approved": true, "mode": "send"} for the second. Sending goes through
execute_at_most_once, because an email can't be unsent.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.types import interrupt

from fuse.actions import (
    ALREADY_SENDING,
    ActionStore,
    execute_at_most_once,
    execute_once,
    idempotency_key,
)

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
    run_send: Callable[[], Awaitable[str]] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    key = idempotency_key(thread_id, tool_call_id)

    existing = await store.get(key)
    if existing and existing.status == "executed":
        return existing.result or "Already done."
    if existing and existing.status == "rejected":
        return REJECTED
    if existing and existing.status == "sending":
        return ALREADY_SENDING

    await store.record(key, tool_name, args)

    decision = interrupt(
        {
            "type": "approval",
            "action_key": key,
            "tool": tool_name,
            "preview": preview,
            **(extra or {}),
        }
    )

    if not (isinstance(decision, dict) and decision.get("approved") is True):
        await store.set_status(key, "rejected")
        return REJECTED

    await store.set_status(key, "approved")
    if decision.get("mode") == "send" and run_send is not None:
        return await execute_at_most_once(store, key, run_send)
    return await execute_once(store, key, run)
