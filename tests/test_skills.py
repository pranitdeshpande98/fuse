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
