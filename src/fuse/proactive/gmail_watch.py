"""Notices new email by polling Gmail's History API.

Gmail gives every change in a mailbox an increasing history ID. We store the last one we
processed (the cursor) and each poll asks "what was added since?". That is one cheap call
when nothing has changed, and nothing is ever fetched twice.

Delivery is at-least-once: the cursor only moves after every new message has been handed
on, so a crash halfway through means a few messages are seen again on the next poll. The
events table drops those duplicates by message ID, so overall each email is handled once.

Why polling, not push: Gmail push needs Google Pub/Sub and a public HTTPS endpoint, and a
laptop has no public address. When Fuse moves to the cloud (Phase 7), push can replace
this class; everything after it stays the same.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fuse.google.auth import GoogleNotConnected
from fuse.google.reader import GmailFeed, HistoryExpired
from fuse.proactive.jobs import JobLog
from fuse.proactive.notify import Notifier
from fuse.proactive.schedule import Clock
from fuse.proactive.store import CursorStore

log = logging.getLogger("fuse.gmail_watch")

CURSOR = "gmail"
SKIP_LABELS = {"SENT", "DRAFT", "SPAM", "TRASH"}
AUTH_EXPIRED = (
    "My Google sign-in has expired, so I can't watch your email or calendar.\n"
    "To fix it, run: uv run python -m fuse.google.auth"
)

Sink = Callable[[dict[str, Any]], Awaitable[None]]


class GmailWatcher:
    def __init__(
        self,
        feed: GmailFeed,
        cursors: CursorStore,
        sink: Sink,
        notifier: Notifier,
        clock: Clock,
        job_log: JobLog,
    ) -> None:
        self._feed, self._cursors, self._sink = feed, cursors, sink
        self._notifier, self._clock, self._job_log = notifier, clock, job_log

    async def poll(self) -> int:
        """Hand every new inbox message to the sink. Returns how many."""
        try:
            return await self._poll()
        except GoogleNotConnected:
            await self._warn_auth_expired()
            return 0

    async def _poll(self) -> int:
        cursor = await self._cursors.get(CURSOR)
        if cursor is None:
            # First run: start from now. Alerting on your whole existing inbox would be noise.
            await self._cursors.set(CURSOR, await self._feed.latest_history_id())
            log.info("Gmail watcher started; only email arriving from now on is watched")
            return 0
        try:
            ids, latest = await self._feed.new_message_ids(cursor)
        except HistoryExpired:
            log.warning("Gmail history too old (bot was off for days); starting from now")
            await self._cursors.set(CURSOR, await self._feed.latest_history_id())
            return 0

        handled = 0
        for message_id in ids:
            summary = await self._feed.message_summary(message_id)
            if summary is None or SKIP_LABELS & set(summary.get("labels", [])):
                continue
            await self._sink(summary)
            handled += 1
        await self._cursors.set(CURSOR, latest)  # only after everything was handed on
        if handled:
            log.info("%d new email(s)", handled)
        return handled

    async def _warn_auth_expired(self) -> None:
        """Tell you once a day, not every two minutes."""

        async def send() -> None:
            await self._notifier.send(AUTH_EXPIRED)

        today = self._clock.now().date().isoformat()
        if await self._job_log.run_once("google_auth_expired_alert", today, send):
            log.warning("Google sign-in expired; told the user")
