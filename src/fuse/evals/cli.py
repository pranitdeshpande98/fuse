"""One command for every eval, with a token cap, saved results and a comparison with the
previous run:

    uv run fuse-evals                     # recall, triage and behaviour
    uv run fuse-evals --only behaviour    # one part
    uv run fuse-evals --scenario injection --max-tokens 30000

Recall runs locally (free). Triage and behaviour call your real model, pinned to the
primary provider with no fallback, so you know which model you measured. Groq's free tier
allows about 8,000 tokens a minute, so a full run takes several minutes: the client waits
and retries when rate-limited. Exit code 1 means something that passed last time fails now.
"""

import argparse
import asyncio
import sys
from typing import Any

from fuse.evals.report import compare, previous, save


def _pinned_model(settings, tools=()):
    """The primary provider only, with patient retries: what you measure is what you run."""
    from langchain_openai import ChatOpenAI

    providers = settings.llm_providers()
    if not providers:
        raise SystemExit("No LLM provider configured: set LLM_PRIMARY_API_KEY in .env")
    provider = providers[0]
    model = ChatOpenAI(
        model=provider.model,
        base_url=provider.base_url,
        api_key=provider.api_key,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=6,  # the OpenAI client waits as the rate-limit headers ask
    )
    return (model.bind_tools(list(tools)) if tools else model), provider.model


async def _recall(settings) -> dict[str, Any]:
    from fuse.memory.embeddings import FastEmbedder
    from fuse.memory.evals import run_eval

    result = await run_eval(FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir))
    return {
        "recall": result.recall,
        "false_recall": result.false_recall_rate,
        "misses": result.misses,
    }


async def _triage(settings) -> dict[str, Any]:
    from fuse.memory.embeddings import FastEmbedder
    from fuse.proactive.decide import TriageResult
    from fuse.proactive.evals import run_triage_eval

    model, _ = _pinned_model(settings)
    triager = model.with_structured_output(TriageResult, method="function_calling")
    embedder = FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir)
    r = await run_triage_eval(triager, embedder)
    return {
        "accuracy": r.accuracy,
        "missed": r.missed,
        "false_alarms": r.false_alarms,
        "wrong": r.wrong,
    }


async def _behaviour(settings, name_filter: str, budget: int) -> dict[str, Any]:
    from fuse.evals.behaviour import SCENARIOS, run_scenario
    from fuse.evals.world import World
    from fuse.memory.embeddings import FastEmbedder
    from fuse.skills.loader import load_skills

    skills = load_skills(settings.skills_dir)
    embedder = FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir)
    chosen = [s for s in SCENARIOS if name_filter.lower() in s.name.lower()]
    rows, used = [], 0
    for scenario in chosen:
        if used >= budget:
            rows.append({"name": scenario.name, "passed": False, "skipped": True})
            print(f"  SKIP  {scenario.name} (token cap reached)", flush=True)
            continue
        result = await run_scenario(
            scenario,
            lambda tools, _s: _pinned_model(settings, tools)[0],
            skills,
            World(embedder=embedder),
            max_tokens=min(30_000, budget - used),
        )
        used += result.trajectory.tokens
        mark = "PASS" if result.passed else "FAIL"
        print(f"  {mark}  {scenario.name}  ({result.trajectory.tokens:,} tokens)", flush=True)
        for check in result.failed_checks:
            print(f"        x {check}", flush=True)
        if result.trajectory.error:
            print(f"        ! {result.trajectory.error}", flush=True)
        rows.append(
            {
                "name": scenario.name,
                "category": scenario.category,
                "passed": result.passed,
                "failed_checks": result.failed_checks,
                "tools": [name for name, _ in result.trajectory.tool_calls],
                "tokens": result.trajectory.tokens,
                "error": result.trajectory.error,
            }
        )
    ran = [r for r in rows if not r.get("skipped")]
    return {
        "pass_rate": (sum(r["passed"] for r in ran) / len(ran)) if ran else 0.0,
        "passed": sum(r["passed"] for r in ran),
        "ran": len(ran),
        "tokens": used,
        "scenarios": rows,
    }


async def run(parts: set[str], name_filter: str, max_tokens: int) -> int:
    from fuse.config import get_settings

    settings = get_settings()
    results: dict[str, Any] = {}
    if parts & {"triage", "behaviour"}:
        _, results["model"] = _pinned_model(settings)
        print(f"Model under test: {results['model']} (no fallback)\n", flush=True)

    if "recall" in parts:
        print("Memory recall (local, free)...", flush=True)
        results["recall"] = await _recall(settings)
        r = results["recall"]
        print(f"  recall {r['recall']:.0%}, false recall {r['false_recall']:.0%}\n", flush=True)
    if "triage" in parts:
        print("Email triage...", flush=True)
        results["triage"] = await _triage(settings)
        t = results["triage"]
        print(
            f"  accuracy {t['accuracy']:.0%}, missed {len(t['missed'])}, "
            f"false alarms {len(t['false_alarms'])}\n",
            flush=True,
        )
    if "behaviour" in parts:
        print(f"Agent behaviour (token cap {max_tokens:,})...", flush=True)
        results["behaviour"] = await _behaviour(settings, name_filter, max_tokens)
        b = results["behaviour"]
        print(f"  {b['passed']}/{b['ran']} passed, {b['tokens']:,} tokens\n", flush=True)

    path = save(results)
    diff = compare(previous(before=path), results)
    print(f"Saved {path}")
    for line in diff.changes:
        print(f"  {line}")
    for line in diff.fixed:
        print(f"  fixed:      {line}")
    for line in diff.regressions:
        print(f"  REGRESSION: {line}")
    return 1 if diff.regressions else 0


def main() -> None:
    from fuse.compat import use_selector_event_loop_on_windows

    parser = argparse.ArgumentParser(description="Run Fuse's evals.")
    parser.add_argument("--only", default="recall,triage,behaviour")
    parser.add_argument("--scenario", default="", help="only behaviour scenarios matching this")
    parser.add_argument("--max-tokens", type=int, default=100_000, help="cap for behaviour")
    args = parser.parse_args()
    use_selector_event_loop_on_windows()
    parts = {p.strip() for p in args.only.split(",") if p.strip()}
    sys.exit(asyncio.run(run(parts, args.scenario, args.max_tokens)))


if __name__ == "__main__":
    main()
