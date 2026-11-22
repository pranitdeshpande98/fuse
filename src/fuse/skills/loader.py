"""Skills: reusable procedures written as markdown, one folder each.

    skills/
      weekly-review/
        SKILL.md

A SKILL.md starts with a small header, then the instructions:

    ---
    name: weekly-review
    description: One line saying when to use it (the model sees only this, at first).
    schedule: fri 17:00          (optional: run it as a background task on a schedule)
    ---
    Step-by-step instructions...

Only the names and descriptions go into every prompt; the full instructions load when the
agent calls `use_skill`. That keeps each turn small (the free tier allows 8,000 tokens a
minute) however many skills there are. This is the same idea as Claude's agent skills:
progressive disclosure.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("fuse.skills")

NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")


class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    schedule: str = ""
    path: Path | None = None


def parse_skill(text: str, folder_name: str = "") -> Skill:
    if not text.startswith("---"):
        raise SkillError("SKILL.md must start with a --- header")
    try:
        _, header, body = text.split("---", 2)
    except ValueError:
        raise SkillError("SKILL.md header must end with ---") from None
    fields: dict[str, str] = {}
    for line in header.strip().splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            raise SkillError(f"header line {line!r} should look like `key: value`")
        fields[key.strip().lower()] = value.strip()
    name = fields.get("name", "")
    if not NAME.match(name):
        raise SkillError(f"name {name!r} must be lowercase letters, digits and dashes")
    if folder_name and name != folder_name:
        raise SkillError(f"name {name!r} must match its folder {folder_name!r}")
    description = fields.get("description", "")
    if not description:
        raise SkillError("a description is required: it's how the agent knows when to use it")
    if not body.strip():
        raise SkillError("the skill has no instructions")
    return Skill(name, description, body.strip(), fields.get("schedule", ""))


def load_skills(directory: str | Path) -> list[Skill]:
    """Every valid skill in the folder. A broken one is logged and skipped, never fatal."""
    root = Path(directory)
    if not root.is_dir():
        return []
    skills = []
    for skill_file in sorted(root.glob("*/SKILL.md")):
        try:
            skill = parse_skill(skill_file.read_text(encoding="utf-8"), skill_file.parent.name)
        except (SkillError, OSError, UnicodeDecodeError) as exc:
            log.warning("skipping skill %s: %s", skill_file.parent.name, exc)
            continue
        skills.append(Skill(**{**skill.__dict__, "path": skill_file}))
    return skills


def skills_prompt(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = [
        "",
        "Skills (step-by-step procedures; call `use_skill` with the name to load one, then",
        "follow it. Use one whenever the request matches its description):",
    ]
    lines += [f"- {s.name}: {s.description}" for s in skills]
    return "\n".join(lines) + "\n"
