"""The job application tracker: one row per application (company + role).

Writing here doesn't need your approval: it's Fuse's own notebook, nothing leaves your
machine, `/jobs` shows all of it, and every change keeps the previous stage in `history`.
This table is also what the job dashboard reads in Phase 6.
"""

from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol, get_args

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

Stage = Literal["interested", "applied", "screening", "interview", "offer", "rejected", "withdrawn"]
STAGES: tuple[str, ...] = get_args(Stage)
OPEN_STAGES = ("interested", "applied", "screening", "interview", "offer")

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS job_applications (
        id            BIGSERIAL PRIMARY KEY,
        company       TEXT        NOT NULL,
        role          TEXT        NOT NULL,
        stage         TEXT        NOT NULL,
        last_contact  DATE,
        next_step     TEXT        NOT NULL DEFAULT '',
        thread_id     TEXT        NOT NULL DEFAULT '',
        notes         TEXT        NOT NULL DEFAULT '',
        history       JSONB       NOT NULL DEFAULT '[]',
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS job_applications_company_role
        ON job_applications (lower(company), lower(role))
    """,
    # Phase 6: the posting's link (e.g. from Job Alerts' Track button)
    "ALTER TABLE job_applications ADD COLUMN IF NOT EXISTS url TEXT NOT NULL DEFAULT ''",
]


@dataclass
class Application:
    id: int
    company: str
    role: str
    stage: str
    last_contact: date | None = None
    next_step: str = ""
    thread_id: str = ""
    notes: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    url: str = ""


def parse_day(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        raise ValueError(f"dates must look like 2026-11-21, got {value!r}") from None


class JobStore(Protocol):
    async def save(
        self,
        company: str,
        role: str,
        stage: str,
        last_contact: date | None = None,
        next_step: str | None = None,
        thread_id: str | None = None,
        notes: str | None = None,
        url: str | None = None,
    ) -> tuple[Application, bool]:
        """Insert or update by company + role. Returns (application, created)."""
        ...

    async def list(self, stage: str | None = None) -> list[Application]: ...


def _check(company: str, role: str, stage: str) -> tuple[str, str, str]:
    company, role, stage = company.strip(), role.strip(), stage.strip().lower()
    if not company or not role:
        raise ValueError("company and role are both needed")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of: {', '.join(STAGES)}")
    return company, role, stage


def _app(row: dict[str, Any]) -> Application:
    return Application(
        id=row["id"],
        company=row["company"],
        role=row["role"],
        stage=row["stage"],
        last_contact=row["last_contact"],
        next_step=row["next_step"],
        thread_id=row["thread_id"],
        notes=row["notes"],
        history=row["history"],
        url=row.get("url", ""),
    )


COLUMNS = "id, company, role, stage, last_contact, next_step, thread_id, notes, history, url"


class PostgresJobStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def save(
        self,
        company,
        role,
        stage,
        last_contact=None,
        next_step=None,
        thread_id=None,
        notes=None,
        url=None,
    ) -> tuple[Application, bool]:
        company, role, stage = _check(company, role, stage)
        now = datetime.now(UTC).date().isoformat()
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                INSERT INTO job_applications
                    (company, role, stage, last_contact, next_step, thread_id, notes, history,
                     url)
                VALUES (%s, %s, %s, %s, COALESCE(%s, ''), COALESCE(%s, ''), COALESCE(%s, ''),
                        %s, COALESCE(%s, ''))
                ON CONFLICT (lower(company), lower(role)) DO UPDATE SET
                    history = CASE WHEN job_applications.stage <> EXCLUDED.stage
                        THEN job_applications.history || jsonb_build_object(
                            'stage', EXCLUDED.stage, 'on', %s::text)
                        ELSE job_applications.history END,
                    stage = EXCLUDED.stage,
                    last_contact = COALESCE(EXCLUDED.last_contact, job_applications.last_contact),
                    next_step = COALESCE(%s, job_applications.next_step),
                    thread_id = COALESCE(NULLIF(%s, ''), job_applications.thread_id),
                    notes = COALESCE(%s, job_applications.notes),
                    url = COALESCE(NULLIF(%s, ''), job_applications.url),
                    updated_at = now()
                RETURNING {COLUMNS}, (xmax = 0) AS created
                """,
                (
                    company,
                    role,
                    stage,
                    last_contact,
                    next_step,
                    thread_id,
                    notes,
                    Jsonb([{"stage": stage, "on": now}]),
                    url,
                    now,
                    next_step,
                    thread_id,
                    notes,
                    url,
                ),
            )
            row = await cur.fetchone()
        return _app(row), row["created"]

    async def list(self, stage: str | None = None) -> list[Application]:
        async with self._pool.connection() as conn:
            if stage:
                cur = await conn.execute(
                    f"SELECT {COLUMNS} FROM job_applications WHERE stage = %s "
                    "ORDER BY updated_at DESC",
                    (stage,),
                )
            else:
                cur = await conn.execute(
                    f"SELECT {COLUMNS} FROM job_applications ORDER BY updated_at DESC"
                )
            return [_app(r) for r in await cur.fetchall()]


class InMemoryJobStore:
    def __init__(self) -> None:
        self.apps: list[Application] = []

    async def save(
        self,
        company,
        role,
        stage,
        last_contact=None,
        next_step=None,
        thread_id=None,
        notes=None,
        url=None,
    ) -> tuple[Application, bool]:
        company, role, stage = _check(company, role, stage)
        today = datetime.now(UTC).date().isoformat()
        for app in self.apps:
            if (app.company.lower(), app.role.lower()) == (company.lower(), role.lower()):
                if app.stage != stage:
                    app.history.append({"stage": stage, "on": today})
                app.stage = stage
                app.last_contact = last_contact or app.last_contact
                app.next_step = app.next_step if next_step is None else next_step
                app.thread_id = thread_id or app.thread_id
                app.notes = app.notes if notes is None else notes
                app.url = url or app.url
                self.apps.remove(app)
                self.apps.insert(0, app)
                return replace(app), False
        app = Application(
            len(self.apps) + 1,
            company,
            role,
            stage,
            last_contact,
            next_step or "",
            thread_id or "",
            notes or "",
            [{"stage": stage, "on": today}],
            url or "",
        )
        self.apps.insert(0, app)
        return replace(app), True

    async def list(self, stage: str | None = None) -> list[Application]:
        return [replace(a) for a in self.apps if stage is None or a.stage == stage]


def format_applications(apps: list[Application]) -> str:
    if not apps:
        return "No job applications tracked yet."
    lines = []
    for a in apps:
        line = f"- {a.company}: {a.role} [{a.stage}]"
        if a.last_contact:
            line += f", last contact {a.last_contact.isoformat()}"
        if a.next_step:
            line += f". Next: {a.next_step}"
        lines.append(line)
    return "\n".join(lines)


async def jobs_text(jobs: JobStore) -> str:
    """/jobs: open applications first, closed ones counted."""
    apps = await jobs.list()
    open_apps = [a for a in apps if a.stage in OPEN_STAGES]
    closed = len(apps) - len(open_apps)
    text = (
        "Your applications:\n" + format_applications(open_apps)
        if open_apps
        else ('No open applications tracked yet. Try: "track my application to Acme for SWE II".')
    )
    if closed:
        text += f"\n\n(+{closed} closed: rejected or withdrawn)"
    return text
