"""Gmail and Calendar tools for the agent. All read-only.

Everything that comes back from an email or a calendar event was written by someone else,
and could contain text trying to steer the agent ("ignore your instructions and..."). So it
is wrapped in <untrusted_content> tags, and the system prompt tells the model never to
follow instructions found inside them.
"""

from typing import Any

from googleapiclient.errors import HttpError
from langchain_core.tools import BaseTool, tool

from fuse.google.auth import GoogleNotConnected
from fuse.google.reader import GoogleReader

UNTRUSTED_OPEN = "<untrusted_content>"
UNTRUSTED_CLOSE = "</untrusted_content>"
MAX_BODY_CHARS = 3000  # per email: keeps long threads within the model's budget


def wrap_untrusted(text: str) -> str:
    """Label outside content. Strips any tags inside it, so it can't close the wrapper early."""
    cleaned = text.replace(UNTRUSTED_OPEN, "").replace(UNTRUSTED_CLOSE, "")
    return f"{UNTRUSTED_OPEN}\n{cleaned}\n{UNTRUSTED_CLOSE}"


def format_events(events: list[dict[str, Any]]) -> str:
    if not events:
        return "No events in that period."
    lines = []
    for event in events:
        start = event.get("start", {})
        when = start.get("dateTime") or f"{start.get('date', '?')} (all day)"
        line = f"- {when}: {event.get('summary', '(no title)')}"
        if event.get("location"):
            line += f" @ {event['location']}"
        lines.append(line)
    return wrap_untrusted("\n".join(lines))


def format_messages(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "No matching emails."
    blocks = [
        f"thread_id: {m['thread_id']}\nFrom: {m['from']}\nDate: {m['date']}\n"
        f"Subject: {m['subject']}\nPreview: {m['snippet']}"
        for m in messages
    ]
    return wrap_untrusted("\n\n".join(blocks))


def format_thread(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "That thread is empty or doesn't exist."
    blocks = []
    for m in messages:
        body = m["body"]
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + "\n...(cut)"
        blocks.append(
            f"From: {m['from']}\nTo: {m['to']}\nDate: {m['date']}\n"
            f"Subject: {m['subject']}\n\n{body}"
        )
    return wrap_untrusted("\n\n---\n\n".join(blocks))


def _problem(exc: Exception) -> str:
    """Turn errors into short text the model can pass on, instead of crashing the turn."""
    if isinstance(exc, GoogleNotConnected):
        return str(exc)
    if isinstance(exc, HttpError):
        return f"Google returned an error ({exc.status_code}). Try again later."
    raise exc


def make_google_tools(reader: GoogleReader) -> list[BaseTool]:
    @tool
    async def list_calendar_events(days_ahead: int = 1, days_back: int = 0) -> str:
        """List the user's calendar events from `days_back` days ago (0-14) until
        `days_ahead` days from now (0-14). For "this past week" use days_back=7, days_ahead=0."""
        ahead, back = max(0, min(days_ahead, 14)), max(0, min(days_back, 14))
        try:
            if back:
                events = await reader.events_between(back, ahead)
            else:
                events = await reader.upcoming_events(max(1, ahead))
        except (GoogleNotConnected, HttpError) as exc:
            return _problem(exc)
        return format_events(events)

    @tool
    async def search_email(query: str, max_results: int = 5) -> str:
        """Search the user's Gmail. `query` uses Gmail search syntax, e.g.
        'from:pat@example.com newer_than:7d', 'is:unread', 'subject:invoice'.
        Returns previews with a thread_id; use read_email_thread for the full text."""
        try:
            messages = await reader.search_messages(query, max(1, min(max_results, 10)))
        except (GoogleNotConnected, HttpError) as exc:
            return _problem(exc)
        return format_messages(messages)

    @tool
    async def read_email_thread(thread_id: str) -> str:
        """Read every message in one email thread. Get the thread_id from search_email."""
        try:
            messages = await reader.read_thread(thread_id)
        except (GoogleNotConnected, HttpError) as exc:
            return _problem(exc)
        return format_thread(messages)

    return [list_calendar_events, search_email, read_email_thread]
