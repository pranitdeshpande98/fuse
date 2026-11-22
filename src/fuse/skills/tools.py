import contextlib

from langchain_core.tools import BaseTool, tool

from fuse.skills.loader import Skill, SkillError, parse_skill


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
