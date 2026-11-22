"""Memory recall eval: does the real embedding model bring back the right memories?

Unit tests check the code works. This checks the *quality* of recall with the real model:
for each question, the memories that should come back, and ones that must not.

Run it:
    uv run python -m fuse.memory.evals
"""

import asyncio
from dataclasses import dataclass

from fuse.memory.context import RECALL_LIMIT, RECALL_MIN_SIMILARITY
from fuse.memory.embeddings import Embedder
from fuse.memory.store import InMemoryMemoryStore

MEMORIES = [
    ("profile", "Name is Pranit"),
    ("profile", "Prefers meetings in the afternoon"),
    ("profile", "Is vegetarian"),
    ("fact", "Pat is the client contact for the Acme proposal"),
    ("fact", "The Acme proposal is due on Friday"),
    ("fact", "Sam is the user's manager"),
    ("fact", "Is building a personal AI agent called Fuse"),
    ("fact", "Has a dentist appointment every six months"),
    ("fact", "Favourite colour is green"),
    ("fact", "Plays badminton on Sunday mornings"),
    ("fact", "Is applying for software engineering jobs"),
    ("fact", "Uses Windows for development"),
]


@dataclass
class Case:
    question: str
    expected: list[str]  # should be recalled
    unrelated: list[str]  # should NOT be recalled


CASES = [
    Case(
        "Draft a reply to Pat about the proposal",
        ["Pat is the client contact"],
        ["badminton", "green"],
    ),
    Case("When is the Acme deadline?", ["Acme proposal is due"], ["dentist", "badminton"]),
    Case("Book a team lunch for Thursday", ["vegetarian"], ["Acme", "badminton"]),
    Case("What time should I schedule the call with Pat?", ["afternoon"], ["green", "dentist"]),
    Case("Who do I report to?", ["manager"], ["badminton", "green"]),
    Case("Any updates on my job search?", ["software engineering jobs"], ["dentist", "vegetarian"]),
    Case("Am I free this Sunday morning?", ["badminton"], ["Acme", "vegetarian"]),
    Case("Remind me when my next checkup is due", ["dentist"], ["Acme", "badminton"]),
    Case("How's my agent project going?", ["Fuse"], ["dentist", "vegetarian"]),
    Case("Which OS do I code on?", ["Windows"], ["vegetarian", "badminton"]),
]


@dataclass
class EvalResult:
    hits: int = 0
    expected_total: int = 0
    false_recalls: int = 0
    unrelated_total: int = 0
    misses: list[str] | None = None

    @property
    def recall(self) -> float:
        return self.hits / self.expected_total if self.expected_total else 1.0

    @property
    def false_recall_rate(self) -> float:
        return self.false_recalls / self.unrelated_total if self.unrelated_total else 0.0


async def run_eval(
    embedder: Embedder,
    min_similarity: float = RECALL_MIN_SIMILARITY,
    limit: int = RECALL_LIMIT,
) -> EvalResult:
    store = InMemoryMemoryStore(embedder)
    for kind, content in MEMORIES:
        await store.add(kind, content)
    result = EvalResult(misses=[])
    for case in CASES:
        found = [m.content for m in await store.search(case.question, limit, min_similarity)]
        for want in case.expected:
            result.expected_total += 1
            if any(want in f for f in found):
                result.hits += 1
            else:
                result.misses.append(f"MISSED  {case.question!r} -> {want!r}")
        for avoid in case.unrelated:
            result.unrelated_total += 1
            if any(avoid in f for f in found):
                result.false_recalls += 1
                result.misses.append(f"EXTRA   {case.question!r} -> {avoid!r}")
    return result


async def _main() -> None:
    from fuse.config import get_settings
    from fuse.memory.embeddings import FastEmbedder

    settings = get_settings()
    embedder = FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir)
    print(f"Model: {settings.embedding_model}  (recall threshold {RECALL_MIN_SIMILARITY})")
    for threshold in (0.5, 0.55, 0.6, 0.65, 0.7):
        r = await run_eval(embedder, min_similarity=threshold)
        marker = "  <- used" if threshold == RECALL_MIN_SIMILARITY else ""
        print(
            f"threshold {threshold:.2f}: recall {r.recall:.0%}, "
            f"false recall {r.false_recall_rate:.0%}{marker}"
        )
    result = await run_eval(embedder)
    for line in result.misses or []:
        print(line)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
