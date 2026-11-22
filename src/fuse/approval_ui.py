"""What an approval looks like on Telegram: the preview text and its buttons.

Shared by your chat and background tasks; only the button data differs (`data_for`).
"""

from collections.abc import Callable
from typing import Any

from fuse.proactive.notify import Button


def approval_text(payload: dict[str, Any]) -> str:
    text = payload.get("preview", "")
    blocked = payload.get("blocked") or []
    if payload.get("email") and blocked:
        who = ", ".join(blocked)
        text += (
            f"\n\nSend isn't available: {who} "
            f"{'is' if len(blocked) == 1 else 'are'} not in your contacts or in this thread. "
            "Tap Always allow to add them, or save a draft and send it yourself."
        )
    return text


def approval_buttons(payload: dict[str, Any], data_for: Callable[[str], str]) -> list[list[Button]]:
    if not payload.get("email"):
        return [[Button("Approve", data_for("approve")), Button("Reject", data_for("reject"))]]
    if payload.get("send_allowed"):
        return [
            [
                Button("Send", data_for("send")),
                Button("Save as draft", data_for("approve")),
                Button("Reject", data_for("reject")),
            ]
        ]
    blocked = payload.get("blocked") or []
    allow = "Always allow " + (blocked[0] if len(blocked) == 1 else f"these {len(blocked)}")
    return [
        [Button("Save as draft", data_for("approve")), Button("Reject", data_for("reject"))],
        [Button(allow[:60], data_for("allow"))],
    ]


def after_allowing(payload: dict[str, Any]) -> dict[str, Any]:
    """The same approval once its blocked recipients are allowed: Send is now offered.
    (Only the buttons change; the send itself re-checks the allowlist in code.)"""
    return {**payload, "blocked": [], "send_allowed": True}
