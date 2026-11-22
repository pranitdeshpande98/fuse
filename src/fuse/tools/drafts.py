"""Write tool: an email, which you either save as a draft or send. Always needs approval.

The preview shows every recipient (To and CC; there is no BCC, so nobody can be added out of
sight). Under it you get *Send*, *Save as draft* and *Reject*, but *Send* only if every
recipient is on your allowlist (contacts/approved addresses, or people already in the
thread). Otherwise you get *Always allow* to add them, or you save a draft and send it
yourself from Gmail.
"""

from typing import Annotated, Any, Protocol

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from fuse.actions import ActionStore
from fuse.contacts import (
    ContactStore,
    InvalidRecipient,
    blocked_recipients,
    parse_recipients,
    thread_participants,
)
from fuse.google.reader import GoogleReader
from fuse.tools.approval import require_approval


class DraftError(Exception):
    """A draft couldn't be saved or an email couldn't be sent. Safe to show the user."""


class DraftSink(Protocol):
    async def save_draft(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        thread_id: str = "",
        in_reply_to: str = "",
    ) -> str: ...

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        thread_id: str = "",
        in_reply_to: str = "",
    ) -> str: ...


class LocalDraftSink:
    """Placeholder until Gmail is connected. Says honestly where the draft went."""

    def __init__(self) -> None:
        self.drafts: list[dict[str, str]] = []
        self.sent: list[dict[str, str]] = []

    async def save_draft(self, to, subject, body, cc="", thread_id="", in_reply_to="") -> str:
        self.drafts.append({"to": to, "cc": cc, "subject": subject, "body": body})
        return f"Draft to {to} saved locally (#{len(self.drafts)}). It is not in Gmail yet."

    async def send(self, to, subject, body, cc="", thread_id="", in_reply_to="") -> str:
        self.sent.append({"to": to, "cc": cc, "subject": subject, "body": body})
        return f"Email to {to} recorded locally (Gmail isn't connected, so nothing was sent)."


def preview_text(to: str, cc: str, subject: str, body: str) -> str:
    lines = [f"To: {to}"]
    if cc:
        lines.append(f"CC: {cc}")
    lines += [f"Subject: {subject}", "", body]
    return "\n".join(lines)


def make_draft_tools(
    store: ActionStore,
    sink: DraftSink,
    contacts: ContactStore | None = None,
    reader: GoogleReader | None = None,
) -> list[BaseTool]:
    async def thread_info(thread_id: str) -> tuple[set[str], str]:
        """Everyone in the thread, and the last message's ID (for a proper reply)."""
        if not thread_id or reader is None:
            return set(), ""
        try:
            messages = await reader.read_thread(thread_id)
        except Exception:
            return set(), ""
        last_id = messages[-1].get("message_id", "") if messages else ""
        return thread_participants(messages), last_id

    async def check(recipients: list[str], thread_id: str) -> list[str]:
        allowed_contacts = await contacts.all() if contacts is not None else set()
        participants, _ = await thread_info(thread_id)
        return blocked_recipients(recipients, allowed_contacts, participants)

    @tool
    async def draft_email_reply(
        to: str,
        subject: str,
        body: str,
        config: RunnableConfig,
        tool_call_id: Annotated[str, InjectedToolCallId],
        cc: str = "",
        reply_to_thread_id: str = "",
    ) -> str:
        """Write an email for the user. They see it first and choose to send it, save it as
        a draft, or reject it.

        Only address people the user named or who are already in the email thread. When
        replying to an email, pass its thread ID as reply_to_thread_id: the reply stays in
        the thread, and its participants count as allowed recipients.
        """
        try:
            to_list, cc_list = parse_recipients(to), parse_recipients(cc)
        except InvalidRecipient as exc:
            return f"Can't write that email: {exc}."
        if not to_list:
            return "Can't write that email: it has no recipient."
        # From here on only the checked, normalised addresses are used: what you see in
        # the preview and what the allowlist checked is exactly what Gmail gets.
        to, cc = ", ".join(to_list), ", ".join(a for a in cc_list if a not in to_list)
        recipients = parse_recipients(to, cc)

        thread_id = str(config["configurable"]["thread_id"])
        blocked = await check(recipients, reply_to_thread_id)
        _, in_reply_to = await thread_info(reply_to_thread_id)
        extra: dict[str, Any] = {
            "email": True,
            "recipients": recipients,
            "blocked": blocked,
            "send_allowed": not blocked,
        }

        async def save() -> str:
            return await sink.save_draft(to, subject, body, cc, reply_to_thread_id, in_reply_to)

        async def send() -> str:
            # Checked again here, at the last moment, in code: a Send decision for a
            # recipient who isn't allowed (an old button, a forged tap) sends nothing.
            still_blocked = await check(recipients, reply_to_thread_id)
            if still_blocked:
                return "Not sent: these recipients aren't on the allowlist: " + ", ".join(
                    still_blocked
                )
            return await sink.send(to, subject, body, cc, reply_to_thread_id, in_reply_to)

        try:
            return await require_approval(
                store=store,
                thread_id=thread_id,
                tool_call_id=tool_call_id,
                tool_name="draft_email_reply",
                args={"to": to, "cc": cc, "subject": subject, "body": body},
                preview=preview_text(to, cc, subject, body),
                run=save,
                run_send=send,
                extra=extra,
            )
        except DraftError as exc:
            # Recorded as "failed"; asking again will ask for approval again.
            return f"That didn't work: {exc}"

    return [draft_email_reply]
