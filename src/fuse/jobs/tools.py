"""Agent tools for the job tracker."""

from langchain_core.tools import BaseTool, tool

from fuse.jobs.store import STAGES, JobStore, format_applications, parse_day


def make_job_tools(jobs: JobStore) -> list[BaseTool]:
    @tool
    async def save_job_application(
        company: str,
        role: str,
        stage: str,
        last_contact: str = "",
        next_step: str | None = None,
        thread_id: str = "",
        notes: str | None = None,
    ) -> str:
        """Add or update one job application in the user's tracker (matched by company and
        role). stage is one of: interested, applied, screening, interview, offer, rejected,
        withdrawn. last_contact is a date like 2026-11-21. thread_id is the Gmail thread.
        Leave next_step or notes out to keep what's already saved."""
        try:
            app, created = await jobs.save(
                company, role, stage, parse_day(last_contact), next_step, thread_id, notes
            )
        except ValueError as exc:
            return f"Not saved: {exc}."
        verb = "Added" if created else "Updated"
        return f"{verb}: {app.company}, {app.role} [{app.stage}]"

    @tool
    async def list_job_applications(stage: str = "") -> str:
        """The user's tracked job applications, newest first. Optionally only one stage."""
        stage = stage.strip().lower()
        if stage and stage not in STAGES:
            return f"Unknown stage {stage!r}; use one of: {', '.join(STAGES)}."
        return format_applications(await jobs.list(stage or None))

    return [save_job_application, list_job_applications]
