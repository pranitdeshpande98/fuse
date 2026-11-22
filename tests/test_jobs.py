import os
from datetime import date

import pytest

from fuse.db import open_pool
from fuse.jobs.store import InMemoryJobStore, PostgresJobStore, jobs_text
from fuse.jobs.tools import make_job_tools

DB_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(params=["memory", "postgres"])
async def jobs(request):
    if request.param == "memory":
        yield InMemoryJobStore()
        return
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    async with open_pool(DB_URL) as pool:
        store = PostgresJobStore(pool)
        await store.ensure_schema()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM job_applications")
        yield store


async def test_save_then_update_by_company_and_role(jobs):
    app, created = await jobs.save("Acme", "SWE II", "applied", date(2026, 11, 10), "Wait")
    assert created and app.stage == "applied"
    app, created = await jobs.save("acme", "swe ii", "interview", next_step="Tue 2 pm")
    assert not created
    assert (app.stage, app.next_step, app.last_contact) == (
        "interview",
        "Tue 2 pm",
        date(2026, 11, 10),  # kept: not given this time
    )
    assert [h["stage"] for h in app.history] == ["applied", "interview"]
    assert len(await jobs.list()) == 1


async def test_bad_stage_and_missing_fields_are_refused(jobs):
    with pytest.raises(ValueError, match="stage must be one of"):
        await jobs.save("Acme", "SWE", "hired-maybe")
    with pytest.raises(ValueError, match="company and role"):
        await jobs.save("", "SWE", "applied")


async def test_jobs_command_shows_open_first(jobs):
    await jobs.save("Acme", "SWE II", "interview")
    await jobs.save("Globex", "Data Eng", "rejected")
    text = await jobs_text(jobs)
    assert "Acme: SWE II [interview]" in text
    assert "Globex" not in text and "+1 closed" in text


async def test_tools(jobs):
    tools = {t.name: t for t in make_job_tools(jobs)}
    out = await tools["save_job_application"].ainvoke(
        {"company": "Acme", "role": "SWE II", "stage": "applied", "last_contact": "2026-11-10"}
    )
    assert out == "Added: Acme, SWE II [applied]"
    bad = await tools["save_job_application"].ainvoke(
        {"company": "Acme", "role": "SWE II", "stage": "applied", "last_contact": "last week"}
    )
    assert bad.startswith("Not saved: dates must look like")
    listed = await tools["list_job_applications"].ainvoke({"stage": "applied"})
    assert "Acme: SWE II [applied], last contact 2026-11-10" in listed
    assert "Unknown stage" in await tools["list_job_applications"].ainvoke({"stage": "zzz"})


async def test_url_is_kept_and_not_wiped_by_a_later_update(jobs):
    await jobs.save("RAND", "SAP Analyst", "interested", url="https://rand.example/1")
    app, _ = await jobs.save("RAND", "SAP Analyst", "applied")
    assert app.url == "https://rand.example/1" and app.stage == "applied"
