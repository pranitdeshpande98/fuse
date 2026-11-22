"""Talks to your Job Alerts dashboard (the Flask app in your autojobalert folder) over its
own JSON API, the same way its web page does.

Fuse never opens Job Alerts' database or files. That keeps the two projects independent:
Job Alerts owns the jobs it finds (and keeps emailing you about them), Fuse owns your
applications, and either can change inside without breaking the other, as long as the API
stays the same.
"""

from dataclasses import dataclass, field
from typing import Any

import httpx

STATE_TIMEOUT = 15.0
SCAN_TIMEOUT = 30 * 60.0  # a full scan visits 100+ careers portals one by one


class JobAlertsError(Exception):
    """A readable reason the dashboard couldn't be reached or used."""


class JobAlertsDown(JobAlertsError):
    pass


class ScanBusy(JobAlertsError):
    pass


@dataclass
class Status:
    active: int
    total: int
    matches: int
    last_new: str | None
    next_scan: str | None
    schedule: str
    email_configured: bool
    pending_alerts: int
    failing: list[tuple[str, str]] = field(default_factory=list)  # (portal, error)


@dataclass
class ScanResult:
    scanned: int
    new: list[dict[str, Any]]
    errors: list[str]
    suppressed: int
    emailed: bool
    email_status: str


def job_line(job: dict[str, Any]) -> str:
    where = f" ({job['location']})" if job.get("location") else ""
    return f"{job.get('company', '?')}: {job.get('title', '?')}{where}"


class JobAlertsClient:
    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport

    async def _request(self, method: str, path: str, timeout: float, **kwargs: Any) -> Any:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout, transport=self._transport
            ) as client:
                response = await client.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise JobAlertsDown(
                f"Job Alerts isn't running at {self.base_url}. Start it with start.bat in "
                "your autojobalert folder."
            ) from exc
        except httpx.TimeoutException as exc:
            raise JobAlertsError("Job Alerts took too long to answer.") from exc
        if response.status_code == 409:
            raise ScanBusy("A scan is already running in Job Alerts. Try again when it's done.")
        if response.status_code >= 400:
            raise JobAlertsError(f"Job Alerts answered {response.status_code}.")
        try:
            return response.json()
        except ValueError as exc:
            raise JobAlertsError("Job Alerts sent something that isn't JSON.") from exc

    async def status(self) -> Status:
        data = await self._request("GET", "/api/state", STATE_TIMEOUT)
        stats = data.get("stats", {})
        failing = [
            (c.get("name", "?"), c.get("last_error") or "error")
            for group in (data.get("groups") or {}).values()
            for c in group
            if c.get("enabled") and c.get("status") == "error"
        ]
        return Status(
            active=int(stats.get("active", 0)),
            total=int(stats.get("total", 0)),
            matches=int(stats.get("matches", 0)),
            last_new=stats.get("last_new"),
            next_scan=data.get("next_scan"),
            schedule=data.get("schedule", ""),
            email_configured=bool(data.get("email_configured")),
            pending_alerts=int(data.get("pending_alerts", 0)),
            failing=failing,
        )

    async def matches(self) -> list[dict[str, Any]]:
        """Every match Job Alerts has recorded, newest first."""
        rows = await self._request("GET", "/api/matches", STATE_TIMEOUT)
        return rows if isinstance(rows, list) else []

    async def scan(self) -> ScanResult:
        """Scan every enabled portal now. Job Alerts emails the new matches itself."""
        data = await self._request("POST", "/api/scan", SCAN_TIMEOUT, json={})
        alerts = data.get("alerts") or {}
        return ScanResult(
            scanned=int(data.get("scanned", 0)),
            new=list(data.get("new") or []),
            errors=[str(e) for e in data.get("errors") or []],
            suppressed=int(data.get("suppressed", 0)),
            emailed=bool(alerts.get("emailed")),
            email_status=str(alerts.get("status", "")),
        )
