"""Gmail: save approved drafts, or send approved emails (only after you tap Send)."""

import asyncio
import base64
from email.message import EmailMessage
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from fuse.google.auth import GoogleNotConnected, TokenStore, get_credentials
from fuse.tools.drafts import DraftError


def build_raw_message(to: str, subject: str, body: str, cc: str = "", in_reply_to: str = "") -> str:
    """An email in the format Gmail's API expects: MIME, then base64url-encoded."""
    message = EmailMessage()
    message["To"] = to
    if cc:
        message["Cc"] = cc
    message["Subject"] = subject
    if in_reply_to:  # makes it a real reply that mail apps show in the same conversation
        message["In-Reply-To"] = in_reply_to
        message["References"] = in_reply_to
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode()


class GmailDraftSink:
    def __init__(self, tokens: TokenStore) -> None:
        self._tokens = tokens

    async def _gmail(self) -> Any:
        creds = await get_credentials(self._tokens)
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    @staticmethod
    def _message(to, subject, body, cc, thread_id, in_reply_to) -> dict[str, str]:
        message = {"raw": build_raw_message(to, subject, body, cc, in_reply_to)}
        if thread_id:
            message["threadId"] = thread_id
        return message

    async def save_draft(self, to, subject, body, cc="", thread_id="", in_reply_to="") -> str:
        message = self._message(to, subject, body, cc, thread_id, in_reply_to)
        try:
            gmail = await self._gmail()
            request = gmail.users().drafts().create(userId="me", body={"message": message})
            draft = await asyncio.to_thread(request.execute)
        except GoogleNotConnected as exc:
            raise DraftError(str(exc)) from exc
        except HttpError as exc:
            raise DraftError(f"Gmail returned an error ({exc.status_code})") from exc
        return f"Draft to {to} saved in your Gmail drafts (id {draft['id']}). It was not sent."

    async def send(self, to, subject, body, cc="", thread_id="", in_reply_to="") -> str:
        message = self._message(to, subject, body, cc, thread_id, in_reply_to)
        try:
            gmail = await self._gmail()
            request = gmail.users().messages().send(userId="me", body=message)
            sent = await asyncio.to_thread(request.execute)
        except GoogleNotConnected as exc:
            raise DraftError(str(exc)) from exc
        except HttpError as exc:
            raise DraftError(f"Gmail refused to send it ({exc.status_code})") from exc
        recipients = to + (f", cc {cc}" if cc else "")
        return f"Sent to {recipients} (Gmail id {sent['id']})."
