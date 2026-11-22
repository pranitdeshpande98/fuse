"""Agent tools for the Job Alerts dashboard."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from langchain_core.tools import BaseTool, tool

from fuse.google.tools import wrap_untrusted
from fuse.jobalerts.client import JobAlertsClient, JobAlertsError, job_line

MAX_LISTED = 15


def status_text(status, base_url: str) -> str:
    lines = [
        f"Job Alerts: {status.active} of {status.total} portals active, "
        f"{status.matches:,} matches recorded.",
        f"Daily scan: {status.schedule}"
        + (f" (next {status.next_scan})" if status.next_scan else ""),
        "Email alerts: " + ("on" if status.email_configured else "not configured"),
    ]
    if status.pending_alerts:
        lines.append(f"{status.pending_alerts} matches are waiting to be emailed.")
    if status.failing:
        lines.append(f"Failing portals ({len(status.failing)}):")
        lines += [f"- {name}: {error[:120]}" for name, error in status.failing[:8]]
    lines.append(f"Dashboard: {base_url}")
    return "\n".join(lines)


def make_job_alerts_tools(client: JobAlertsClient) -> list[BaseTool]:
    @tool
    async def job_alerts_status() -> str:
        """Status of the user's Job Alerts dashboard: portals active, matches, next daily
        scan, email alerts, and which careers portals are failing."""
        try:
            return status_text(await client.status(), client.base_url)
        except JobAlertsError as exc:
            return str(exc)

    @tool
    async def recent_job_matches(days: int = 7, keyword: str = "") -> str:
        """Job matches Job Alerts found in the last `days` days (1-60), newest first,
        optionally only titles or companies containing `keyword`. Includes links."""
        try:
            jobs = await client.matches()
        except JobAlertsError as exc:
            return str(exc)
        since = datetime.now(UTC) - timedelta(days=max(1, min(days, 60)))
        word = keyword.strip().lower()
        found = []
        for job in jobs:
            try:
                seen = datetime.fromisoformat(job.get("first_seen", ""))
            except ValueError:
                continue
            if seen.tzinfo is None:
                seen = seen.replace(tzinfo=UTC)
            text = f"{job.get('title', '')} {job.get('company', '')}".lower()
            if seen >= since and (not word or word in text):
                found.append(job)
        if not found:
            return f"No matches in the last {days} days" + (f" for {keyword!r}." if word else ".")
        lines = [
            f"- {job_line(j)}, found {j['first_seen'][:10]}: {j.get('url', '')}"
            for j in found[:MAX_LISTED]
        ]
        more = (
            f"\n(+{len(found) - MAX_LISTED} more on the dashboard)"
            if len(found) > MAX_LISTED
            else ""
        )
        # Titles and locations come from employers' websites: data, not instructions.
        return f"{len(found)} matches:\n" + wrap_untrusted("\n".join(lines)) + more

    return [job_alerts_status, recent_job_matches]


def make_scan_tool(start_scan: Callable[[], str]) -> BaseTool:
    """Only where a scan can actually be started and reported (the bot process)."""

    @tool
    def scan_job_portals() -> str:
        """Start a Job Alerts scan of every enabled careers portal now. It takes a few
        minutes and runs in the background; the user gets the new matches on Telegram and
        by email when it finishes."""
        return start_scan()

    return scan_job_portals
