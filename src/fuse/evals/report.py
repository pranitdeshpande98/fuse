"""Saving eval results and comparing a run with the previous one."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("evals/results")


@dataclass
class Comparison:
    regressions: list[str] = field(default_factory=list)  # passed before, fails now
    fixed: list[str] = field(default_factory=list)  # failed before, passes now
    changes: list[str] = field(default_factory=list)  # metric moves, for the summary


def save(results: dict[str, Any], directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"{stamp}.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def previous(directory: Path = RESULTS_DIR, before: Path | None = None) -> dict[str, Any] | None:
    files = sorted(p for p in directory.glob("*.json") if p != before) if directory.is_dir() else []
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def compare(old: dict[str, Any] | None, new: dict[str, Any]) -> Comparison:
    result = Comparison()
    if not old:
        return result
    old_b = {s["name"]: s["passed"] for s in old.get("behaviour", {}).get("scenarios", [])}
    for s in new.get("behaviour", {}).get("scenarios", []):
        if s.get("skipped") or s["name"] not in old_b:
            continue
        if old_b[s["name"]] and not s["passed"]:
            result.regressions.append(f"behaviour: {s['name']}")
        elif not old_b[s["name"]] and s["passed"]:
            result.fixed.append(f"behaviour: {s['name']}")
    for section, key in [
        ("recall", "recall"),
        ("recall", "false_recall"),
        ("triage", "accuracy"),
        ("behaviour", "pass_rate"),
    ]:
        before = old.get(section, {}).get(key)
        after = new.get(section, {}).get(key)
        if isinstance(before, float) and isinstance(after, float) and abs(after - before) > 1e-9:
            result.changes.append(f"{section} {key}: {before:.0%} -> {after:.0%}")
    return result
