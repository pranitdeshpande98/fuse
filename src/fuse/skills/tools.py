import contextlib

from langchain_core.tools import BaseTool, tool

from fuse.skills.loader import Skill, SkillError, describe_schedule, parse_skill
from fuse.tasks.queue import TaskStore
from fuse.tasks.tools import start_task


def make_skill_tools(skills: list[Skill]) -> list[BaseTool]:
    by_name = {s.name: s for s in skills}

    @tool
    def use_skill(name: str) -> str:
        """Load the full instructions of one of your skills, by name. Follow them."""
        skill = by_name.get(name.strip().lower())
        if skill is None:
            return f"No skill called {name!r}. Available: {', '.join(by_name) or 'none'}."
        if skill.path is not None:
            # Read it again, so edits to SKILL.md apply without a restart. If the edited
            # file is broken, keep the version loaded at startup.
            with contextlib.suppress(OSError, SkillError):
                skill = parse_skill(skill.path.read_text(encoding="utf-8"), skill.name)
        return f"Skill: {skill.name}\n\n{skill.instructions}"

    return [use_skill] if skills else []


def skill_goal(name: str) -> str:
    return f"Use the {name} skill: call use_skill('{name}') and follow its steps."


async def start_skill(
    tasks: TaskStore, skills: list[Skill], name: str, max_attempts: int = 3
) -> str:
    """/skill <name>, and scheduled skills: run one as a background task."""
    name = name.strip().lower()
    if not name:
        return "Usage: /skill <name>. See them with /skills."
    if name not in {s.name for s in skills}:
        return f"No skill called {name!r}. See them with /skills."
    return await start_task(tasks, skill_goal(name), max_attempts)


def skills_text(skills: list[Skill]) -> str:
    """/skills"""
    if not skills:
        return "No skills found. Add one as skills/<name>/SKILL.md."
    lines = ["Skills (run one now with /skill <name>, or just ask):"]
    for s in skills:
        line = f"- {s.name}: {s.description}"
        if s.schedule:
            line += f" [runs {describe_schedule(s.schedule)}]"
        lines.append(line)
    return "\n".join(lines)
