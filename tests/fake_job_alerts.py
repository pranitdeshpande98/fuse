"""A fake Job Alerts dashboard that answers like the real Flask app, for tests."""

import json

import httpx

JOBS = [
    {
        "job_key": "k3",
        "company": "Carnegie Mellon University",
        "title": "Software Engineer, Research Computing",
        "location": "Pittsburgh, PA",
        "url": "https://cmu.example/jobs/3",
        "first_seen": "2026-11-21T23:31:00+00:00",
        "category": "University",
    },
    {
        "job_key": "k2",
        "company": "RAND",
        "title": "SAP Analyst",
        "location": "Santa Monica, CA",
        "url": "https://rand.example/jobs/2",
        "first_seen": "2026-11-20T23:31:00+00:00",
        "category": "Nonprofit research",
    },
    {
        "job_key": "k1",
        "company": "HHMI",
        "title": "Data Engineer",
        "location": "",
        "url": "https://hhmi.example/jobs/1",
        "first_seen": "2026-11-18T23:31:00+00:00",
        "category": "Nonprofit research",
    },
]


class FakeJobAlerts:
    def __init__(self, jobs=None, down=False, busy=False):
        self.jobs = list(jobs if jobs is not None else JOBS)
        self.down, self.busy = down, busy
        self.scans = 0
        self.next_new: list[dict] = []

    def add_job(self, job: dict) -> None:
        self.jobs.insert(0, job)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/api/state":
            return httpx.Response(
                200,
                json={
                    "groups": {
                        "University": [
                            {"name": "CMU", "enabled": True, "status": "working"},
                            {
                                "name": "Duke",
                                "enabled": True,
                                "status": "error",
                                "last_error": "Workday 500",
                            },
                            {"name": "MIT", "enabled": False, "status": "error"},
                        ]
                    },
                    "stats": {
                        "active": 2,
                        "total": 3,
                        "matches": len(self.jobs),
                        "last_new": self.jobs[0]["first_seen"] if self.jobs else None,
                    },
                    "next_scan": "2026-11-22T17:30:00-06:00",
                    "schedule": "17:30 America/Chicago",
                    "email_configured": True,
                    "pending_alerts": 0,
                },
            )
        if request.url.path == "/api/matches":
            return httpx.Response(200, content=json.dumps(self.jobs))
        if request.url.path == "/api/scan" and request.method == "POST":
            if self.busy:
                return httpx.Response(409, json={"error": "A scan is already running"})
            self.scans += 1
            new, self.next_new = self.next_new, []
            for job in new:
                self.add_job(job)
            return httpx.Response(
                200,
                json={
                    "scanned": 2,
                    "new": new,
                    "new_count": len(new),
                    "errors": ["Duke: Workday 500"],
                    "suppressed": 0,
                    "alerts": {
                        "emailed": bool(new),
                        "sent": len(new),
                        "status": "sent" if new else "nothing pending",
                    },
                },
            )
        return httpx.Response(404)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
