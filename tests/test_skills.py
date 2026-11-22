from pathlib import Path

import pytest

from fuse.skills.loader import SkillError, load_skills, parse_skill, skills_prompt
from fuse.skills.tools import make_skill_tools

REPO_SKILLS = Path(__file__).parent.parent / "skills"
GOOD = """---
name: weekly-review
description: Friday review of the week.
schedule: fri 17:00
---
1. Do the thing.
"""


def test_the_four_starter_skills_load():
    skills = {s.name: s for s in load_skills(REPO_SKILLS)}
    assert set(skills) == {
        "job-application-tracker",
        "interview-prep",
        "follow-up-nudges",
        "weekly-review",
    }
    assert skills["weekly-review"].schedule == "fri 17:00"
    assert skills["follow-up-nudges"].schedule == "mon 09:00"
    assert all(len(s.description) < 200 for s in skills.values())  # one line in every prompt


def test_parse_skill():
    skill = parse_skill(GOOD, "weekly-review")
    assert (skill.name, skill.schedule) == ("weekly-review", "fri 17:00")
    assert skill.instructions == "1. Do the thing."


@pytest.mark.parametrize(
    "text,folder,error",
    [
        ("no header", "x", "must start"),
        ("---\nname: Bad Name\ndescription: d\n---\nbody", "", "lowercase"),
        ("---\nname: other\ndescription: d\n---\nbody", "mine", "match its folder"),
        ("---\nname: ok-name\n---\nbody", "", "description is required"),
        ("---\nname: ok-name\ndescription: d\n---\n  ", "", "no instructions"),
    ],
)
def test_bad_skills_are_rejected_with_a_reason(text, folder, error):
    with pytest.raises(SkillError, match=error):
        parse_skill(text, folder)


def test_a_broken_skill_is_skipped_not_fatal(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "SKILL.md").write_text(GOOD.replace("weekly-review", "good"))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "SKILL.md").write_text("oops")
    assert [s.name for s in load_skills(tmp_path)] == ["good"]
    assert load_skills(tmp_path / "missing") == []


def test_only_names_and_descriptions_go_in_the_prompt():
    skills = load_skills(REPO_SKILLS)
    prompt = skills_prompt(skills)
    assert "use_skill" in prompt and "- weekly-review: A Friday review" in prompt
    assert "list_job_applications" not in prompt  # instructions load on demand
    assert skills_prompt([]) == ""


def test_use_skill_reads_the_latest_file(tmp_path):
    (tmp_path / "demo").mkdir()
    path = tmp_path / "demo" / "SKILL.md"
    path.write_text(GOOD.replace("weekly-review", "demo"))
    (tool,) = make_skill_tools(load_skills(tmp_path))
    assert "Do the thing." in tool.invoke({"name": "demo"})
    path.write_text(GOOD.replace("weekly-review", "demo").replace("Do the thing", "Edited"))
    assert "Edited" in tool.invoke({"name": "demo"})  # no restart needed
    path.write_text("broken now")
    assert "Edited" not in tool.invoke({"name": "demo"})  # falls back to the startup version
    assert "No skill called" in tool.invoke({"name": "nope"})
    assert make_skill_tools([]) == []


def test_agent_prompt_lists_skills():
    from langgraph.checkpoint.memory import InMemorySaver

    from fuse.actions import InMemoryActionStore
    from fuse.agent.factory import build_runner
    from fuse.config import Settings

    settings = Settings(_env_file=None, llm_primary_api_key="k")
    runner = build_runner(
        settings, InMemorySaver(), InMemoryActionStore(), skills=load_skills(REPO_SKILLS)
    )
    tool_names = set(runner._graph.nodes["tools"].bound.tools_by_name)
    assert "use_skill" in tool_names


def test_schedules():
    from datetime import time

    from fuse.skills.loader import describe_schedule, parse_schedule

    assert parse_schedule("fri 17:00") == (frozenset({4}), time(17, 0))
    assert parse_schedule("mon,wed 08:30")[0] == frozenset({0, 2})
    assert parse_schedule("weekdays 09:00")[0] == frozenset(range(5))
    assert describe_schedule("fri 17:00") == "Fri at 17:00"
    assert describe_schedule("daily 07:00") == "every day at 07:00"
    for bad in ["friday at 5", "fri 25:00", "funday 10:00"]:
        with pytest.raises(SkillError):
            parse_schedule(bad)
    with pytest.raises(SkillError, match="schedule"):
        parse_skill(GOOD.replace("fri 17:00", "fridays"), "weekly-review")


async def test_a_scheduled_skill_runs_on_its_day_only_once():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from fuse.proactive.jobs import InMemoryJobLog
    from fuse.proactive.schedule import FixedClock, Scheduler
    from fuse.skills.loader import parse_schedule
    from fuse.skills.tools import start_skill
    from fuse.tasks.queue import InMemoryTaskStore

    tasks, skills = InMemoryTaskStore(), load_skills(REPO_SKILLS)
    clock = FixedClock(datetime(2026, 11, 26, 17, 5, tzinfo=ZoneInfo("America/Chicago")))  # Thu
    scheduler = Scheduler(clock, InMemoryJobLog())
    days, at = parse_schedule("fri 17:00")
    scheduler.add_daily(
        "skill:weekly-review",
        at,
        lambda: start_skill(tasks, skills, "weekly-review"),
        weekdays=days,
    )
    await scheduler.tick()
    assert await tasks.recent() == []  # Thursday: not its day
    clock.advance(days=1)  # Friday 17:05
    await scheduler.tick()
    await scheduler.tick()
    (task,) = await tasks.recent()
    assert "use_skill('weekly-review')" in task.goal


async def test_skill_commands():
    from fuse.skills.tools import skills_text, start_skill
    from fuse.tasks.queue import InMemoryTaskStore

    skills = load_skills(REPO_SKILLS)
    text = skills_text(skills)
    assert "weekly-review" in text and "[runs Fri at 17:00]" in text
    tasks = InMemoryTaskStore()
    assert "Started background task" in await start_skill(tasks, skills, "interview-prep")
    assert "No skill called" in await start_skill(tasks, skills, "nope")
    assert "Usage" in await start_skill(tasks, skills, " ")
