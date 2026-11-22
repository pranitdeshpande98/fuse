from fuse.memory.embeddings import HashingEmbedder
from fuse.proactive.decide import TriageResult
from fuse.proactive.evals import CASES, run_triage_eval


class AlwaysNotify:
    async def ainvoke(self, messages):
        return TriageResult(action="notify", reason="always")


class AlwaysDigest:
    async def ainvoke(self, messages):
        return TriageResult(action="digest", reason="always")


async def test_rules_alone_get_their_cases_right():
    result = await run_triage_eval(None, HashingEmbedder())  # no model: rules + fallback
    assert result.total == len(CASES)
    assert result.by["rule"] == 3
    assert not any("(rule)" in line for line in result.missed + result.false_alarms + result.wrong)


async def test_eval_separates_false_alarms_from_misses():
    noisy = await run_triage_eval(AlwaysNotify(), HashingEmbedder())
    quiet = await run_triage_eval(AlwaysDigest(), HashingEmbedder())
    assert noisy.false_alarms and not noisy.missed
    assert quiet.missed and not quiet.false_alarms
    assert any("injection" in line for line in noisy.false_alarms)
