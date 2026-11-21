"""Write tool: draft an email reply. Always goes through the approval gate.

For now drafts are kept locally. Step 10 (Google) replaces LocalDraftSink with one that
saves to your real Gmail drafts.
"""

from typing import Annotated, Protocol

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from fuse.actions import ActionStore
from fuse.tools.approval import require_approval


class DraftSink(Protocol):
    async def save_draft(self, to: str, subject: str, body: str) -> str: ...


class LocalDraftSink:
    """Placeholder until Gmail is connected. Says honestly where the draft went."""

    def __init__(self) -> None:
        self.drafts: list[dict[str, str]] = []

    async def save_draft(self, to: str, subject: str, body: str) -> str:
        self.drafts.append({"to": to, "subject": subject, "body": body})
        return f"Draft to {to} saved locally (#{len(self.drafts)}). It is not in Gmail yet."


def make_draft_tools(store: ActionStore, sink: DraftSink) -> list[BaseTool]:
    @tool
    async def draft_email_reply(
        to: str,
        subject: str,
        body: str,
        config: RunnableConfig,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """Draft an email for the user. The user must approve it before it is saved.

        Only address people the user named or who are already in the email thread.
        """
        thread_id = str(config["configurable"]["thread_id"])
        return await require_approval(
            store=store,
            thread_id=thread_id,
            tool_call_id=tool_call_id,
            tool_name="draft_email_reply",
            args={"to": to, "subject": subject, "body": body},
            preview=f"To: {to}\nSubject: {subject}\n\n{body}",
            run=lambda: sink.save_draft(to, subject, body),
        )

    return [draft_email_reply]
