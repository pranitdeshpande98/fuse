"""Read-only access to Gmail and Google Calendar.

Talks to Google and returns plain Python data. Formatting for the model lives in tools.py,
so it can be tested without calling Google.
"""

import asyncio
import base64
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from fuse.google.auth import TokenStore, get_credentials


class GoogleReader(Protocol):
    async def upcoming_events(self, days_ahead: int) -> list[dict[str, Any]]: ...
    async def events_between(self, days_back: int, days_ahead: int) -> list[dict[str, Any]]: ...
    async def search_messages(self, query: str, max_results: int) -> list[dict[str, str]]: ...
    async def read_thread(self, thread_id: str) -> list[dict[str, str]]: ...


class HistoryExpired(Exception):
    """Gmail no longer has history from that far back (roughly a week); start again."""


class GmailFeed(Protocol):
    """What the Gmail watcher needs: new messages since a point in the mailbox's history."""

    async def latest_history_id(self) -> str: ...
    async def new_message_ids(self, since_history_id: str) -> tuple[list[str], str]: ...
    async def message_summary(self, message_id: str) -> dict[str, Any] | None: ...


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    return {h["name"].lower(): h["value"] for h in payload.get("headers", [])}


def _plain_text(payload: dict[str, Any]) -> str:
    """The text/plain body of an email, searching nested parts. Empty if there is none."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", "replace")
    for part in payload.get("parts", []):
        text = _plain_text(part)
        if text:
            return text
    return ""


class GoogleApiReader:
    """The real implementation. Google's client is synchronous, so calls run in a thread."""

    def __init__(self, tokens: TokenStore) -> None:
        self._tokens = tokens

    async def _service(self, name: str, version: str) -> Any:
        creds = await get_credentials(self._tokens)
        return build(name, version, credentials=creds, cache_discovery=False)

    async def upcoming_events(self, days_ahead: int) -> list[dict[str, Any]]:
        return await self.events_between(0, days_ahead)

    async def events_between(self, days_back: int, days_ahead: int) -> list[dict[str, Any]]:
        calendar = await self._service("calendar", "v3")
        now = datetime.now(UTC)
        request = calendar.events().list(
            calendarId="primary",
            timeMin=(now - timedelta(days=days_back)).isoformat(),
            timeMax=(now + timedelta(days=days_ahead)).isoformat(),
            singleEvents=True,  # expand repeating events into individual ones
            orderBy="startTime",
            maxResults=50,
        )
        result = await asyncio.to_thread(request.execute)
        return result.get("items", [])

    async def search_messages(self, query: str, max_results: int) -> list[dict[str, str]]:
        gmail = await self._service("gmail", "v1")
        listing = await asyncio.to_thread(
            gmail.users().messages().list(userId="me", q=query, maxResults=max_results).execute
        )
        messages = []
        for item in listing.get("messages", []):
            msg = await asyncio.to_thread(
                gmail.users()
                .messages()
                .get(
                    userId="me",
                    id=item["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
                .execute
            )
            headers = _headers(msg.get("payload", {}))
            messages.append(
                {
                    "thread_id": msg["threadId"],
                    "from": headers.get("from", ""),
                    "subject": headers.get("subject", "(no subject)"),
                    "date": headers.get("date", ""),
                    "snippet": msg.get("snippet", ""),
                }
            )
        return messages

    async def read_thread(self, thread_id: str) -> list[dict[str, str]]:
        gmail = await self._service("gmail", "v1")
        thread = await asyncio.to_thread(
            gmail.users().threads().get(userId="me", id=thread_id, format="full").execute
        )
        result = []
        for msg in thread.get("messages", []):
            payload = msg.get("payload", {})
            headers = _headers(payload)
            result.append(
                {
                    "from": headers.get("from", ""),
                    "to": headers.get("to", ""),
                    "cc": headers.get("cc", ""),
                    "subject": headers.get("subject", "(no subject)"),
                    "date": headers.get("date", ""),
                    "message_id": headers.get("message-id", ""),
                    "body": _plain_text(payload) or msg.get("snippet", ""),
                }
            )
        return result

    # --- GmailFeed: incremental sync for the watcher ---------------------------------

    async def latest_history_id(self) -> str:
        gmail = await self._service("gmail", "v1")
        profile = await asyncio.to_thread(gmail.users().getProfile(userId="me").execute)
        return str(profile["historyId"])

    async def new_message_ids(self, since_history_id: str) -> tuple[list[str], str]:
        """IDs of messages added to the inbox since that history ID, oldest first, and the
        history ID to continue from next time."""
        gmail = await self._service("gmail", "v1")
        ids: list[str] = []
        latest = since_history_id
        page_token = None
        while True:
            request = (
                gmail.users()
                .history()
                .list(
                    userId="me",
                    startHistoryId=since_history_id,
                    historyTypes=["messageAdded"],
                    labelId="INBOX",
                    pageToken=page_token,
                )
            )
            try:
                page = await asyncio.to_thread(request.execute)
            except HttpError as exc:
                if exc.resp.status == 404:
                    raise HistoryExpired(since_history_id) from exc
                raise
            for record in page.get("history", []):
                for added in record.get("messagesAdded", []):
                    message_id = added["message"]["id"]
                    if message_id not in ids:
                        ids.append(message_id)
            latest = str(page.get("historyId", latest))
            page_token = page.get("nextPageToken")
            if not page_token:
                return ids, latest

    async def message_summary(self, message_id: str) -> dict[str, Any] | None:
        """Headers and snippet only: enough to triage, without downloading the body."""
        gmail = await self._service("gmail", "v1")
        request = (
            gmail.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date", "List-Unsubscribe"],
            )
        )
        try:
            msg = await asyncio.to_thread(request.execute)
        except HttpError as exc:
            if exc.resp.status == 404:  # deleted before we got to it
                return None
            raise
        headers = _headers(msg.get("payload", {}))
        return {
            "id": msg["id"],
            "thread_id": msg["threadId"],
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", "(no subject)"),
            "date": headers.get("date", ""),
            "snippet": msg.get("snippet", ""),
            "labels": msg.get("labelIds", []),
            "list_unsubscribe": bool(headers.get("list-unsubscribe")),
        }
