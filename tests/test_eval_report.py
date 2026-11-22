from fuse.evals.report import compare, previous, save


def run(passes: dict[str, bool], rate: float) -> dict:
    return {
        "triage": {"accuracy": rate},
        "behaviour": {
            "pass_rate": rate,
            "scenarios": [{"name": n, "passed": p} for n, p in passes.items()],
        },
    }


def test_compare_finds_regressions_fixes_and_metric_moves():
    old = run({"injection": True, "reply": False, "calendar": True}, 0.5)
    new = run({"injection": False, "reply": True, "calendar": True}, 0.75)
    diff = compare(old, new)
    assert diff.regressions == ["behaviour: injection"]
    assert diff.fixed == ["behaviour: reply"]
    assert "triage accuracy: 50% -> 75%" in diff.changes


def test_first_run_and_skipped_scenarios_are_not_regressions():
    assert compare(None, run({"a": False}, 0.0)).regressions == []
    old = run({"a": True}, 1.0)
    new = {"behaviour": {"scenarios": [{"name": "a", "passed": False, "skipped": True}]}}
    assert compare(old, new).regressions == []


def test_save_and_previous(tmp_path):
    first = save(run({"a": True}, 1.0), tmp_path)
    second = save(run({"a": False}, 0.0), tmp_path)
    assert first != second  # two runs in the same second don't overwrite each other
    assert previous(tmp_path, before=second)["behaviour"]["scenarios"][0]["passed"] is True
    assert previous(tmp_path / "nothing") is None
