"""Nightly memory consolidation ("dreaming"), on a schedule instead of by hand.

It runs at 3 am, inside quiet hours, so it doesn't message you. If it learned something,
it leaves a held note that the morning brief shows: "Learned overnight: ...", plus any
memory it thinks changed but didn't save, for you to confirm.
"""

import logging
from collections.abc import Awaitable, Callable

from fuse.memory.consolidate import ConsolidationReport
from fuse.proactive.events import EventStore
from fuse.proactive.schedule import Clock

log = logging.getLogger("fuse.nightly")


def morning_note(report: ConsolidationReport) -> str | None:
    if not report.saved and not report.needs_review:
        return None
    parts = []
    if report.saved:
        parts.append("Learned overnight: " + "; ".join(report.saved))
    if report.needs_review:
        changed = "; ".join(f"{new} (was: {old})" for new, old in report.needs_review)
        parts.append(f"Possibly changed, not saved (tell me if it's right): {changed}")
    return "\n".join(parts)


class NightlyConsolidation:
    def __init__(
        self,
        run_consolidation: Callable[[], Awaitable[ConsolidationReport]],
        events: EventStore,
        clock: Clock,
    ) -> None:
        self._consolidate, self._events, self._clock = run_consolidation, events, clock

    async def run(self) -> None:
        report = await self._consolidate()
        log.info("nightly consolidation: %s", report.summary().replace("\n", " | "))
        note = morning_note(report)
        if note is None:
            return
        today = self._clock.now().date().isoformat()
        event = await self._events.add("memory", f"consolidation:{today}", {"text": note})
        if event is not None:
            await self._events.decide(event.id, "held", "for the morning brief", self._clock.now())
