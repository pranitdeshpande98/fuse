"""Save approved drafts to the user's real Gmail drafts folder. Never sends anything."""

import asyncio
import base64
from email.message import EmailMessage
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from fuse.google.auth import GoogleNotConnected, TokenStore, get_credentials
from fuse.tools.drafts import DraftError


def build_raw_message(to: str, subject: str, body: str) -> str:
    """An email in the format Gmail's API expects: MIME, then base64url-encoded."""
    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode()


class GmailDraftSink:
    def __init__(self, tokens: TokenStore) -> None:
        self._tokens = tokens

    async def _gmail(self) -> Any:
        creds = await get_credentials(self._tokens)
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    async def save_draft(self, to: str, subject: str, body: str) -> str:
        try:
            gmail = await self._gmail()
            request = (
                gmail.users()
                .drafts()
                .create(
                    userId="me", body={"message": {"raw": build_raw_message(to, subject, body)}}
                )
            )
            draft = await asyncio.to_thread(request.execute)
        except GoogleNotConnected as exc:
            raise DraftError(str(exc)) from exc
        except HttpError as exc:
            raise DraftError(f"Gmail returned an error ({exc.status_code})") from exc
        return f"Draft to {to} saved in your Gmail drafts (id {draft['id']}). It was not sent."
