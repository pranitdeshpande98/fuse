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
from datetime import time
from pathlib import Path

log = logging.getLogger("fuse.skills")

NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")


class SkillError(ValueError):
    pass


DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def parse_schedule(text: str) -> tuple[frozenset[int], time]:
    """'fri 17:00' -> ({4}, 17:00). Days: daily, weekdays, weekends, or names like mon,wed."""
    try:
        days_part, clock = text.strip().lower().rsplit(" ", 1)
        hours, minutes = clock.split(":")
        at = time(int(hours), int(minutes))
    except ValueError:
        raise SkillError(f"schedule {text!r} should look like 'fri 17:00'") from None
    named = {"daily": range(7), "weekdays": range(5), "weekends": range(5, 7)}
    if days_part in named:
        return frozenset(named[days_part]), at
    days = set()
    for day in days_part.replace(" ", "").split(","):
        if day not in DAYS:
            raise SkillError(f"unknown day {day!r} in schedule; use mon, tue, ... or daily")
        days.add(DAYS.index(day))
    return frozenset(days), at


def describe_schedule(text: str) -> str:
    days, at = parse_schedule(text)
    if days == frozenset(range(7)):
        when = "every day"
    elif days == frozenset(range(5)):
        when = "weekdays"
    else:
        when = ", ".join(DAYS[d].capitalize() for d in sorted(days))
    return f"{when} at {at:%H:%M}"


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
    schedule = fields.get("schedule", "")
    if schedule:
        parse_schedule(schedule)  # a bad schedule is reported now, not silently never run
    return Skill(name, description, body.strip(), schedule)


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
