"""Consolidation ("dreaming"): review recent conversations and save what's worth keeping.

During a chat the agent saves things it notices, but it misses some. This job reads the
conversations since its last run, asks the LLM to pull out lasting facts about the user,
and saves the ones that are new.

Safety: it reads ONLY the user's own messages. Tool results (emails, calendar events) are
written by other people and could try to plant false "memories", so they are never used.
It also never overwrites existing memories: anything that looks like an update to one is
listed for you to review instead.

Run it:
    uv run python -m fuse.memory.consolidate            # save new memories
    uv run python -m fuse.memory.consolidate --dry-run  # only show what it would save
Phase 3 will run it automatically every night.
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from fuse.memory.store import MemoryStore
from fuse.memory.tools import RELATED_SIMILARITY

log = logging.getLogger("fuse.consolidate")

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS consolidation_progress (
        thread_id      TEXT PRIMARY KEY,
        messages_seen  INTEGER     NOT NULL,
        updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
]

EXTRACTION_PROMPT = """Below are messages a user sent to their personal assistant.
List the facts about the user that are worth remembering for weeks or months:
preferences, people they work with and their roles, ongoing projects, decisions, routines.

Rules:
- Only facts the user stated about themselves or their world. No guesses.
- Skip one-off requests ("what time is it", "draft a reply") and anything temporary.
- Each fact is one short sentence in the third person, e.g. "Prefers meetings in the afternoon".
- kind="profile" for stable facts that matter in most conversations (name, role, timezone,
  standing preferences); kind="fact" for everything else.
- If nothing is worth remembering, return an empty list.

User messages:
{messages}
"""


class ExtractedFact(BaseModel):
    content: str = Field(description="One short sentence about the user")
    kind: Literal["profile", "fact"]


class ExtractedFacts(BaseModel):
    facts: list[ExtractedFact]


class Extractor(Protocol):
    async def ainvoke(self, prompt: str) -> ExtractedFacts: ...


@dataclass
class ConsolidationReport:
    threads: int = 0
    new_messages: int = 0
    saved: list[str] = field(default_factory=list)
    already_known: list[str] = field(default_factory=list)
    needs_review: list[tuple[str, str]] = field(default_factory=list)  # (new, existing)

    def summary(self) -> str:
        def plural(n: int, one: str, many: str) -> str:
            return f"{n} {one if n == 1 else many}"

        lines = [
            f"Read {plural(self.new_messages, 'new message', 'new messages')} from "
            f"{plural(self.threads, 'conversation', 'conversations')}.",
            f"Saved {plural(len(self.saved), 'new memory', 'new memories')}"
            + (":" if self.saved else "."),
            *[f"  + {s}" for s in self.saved],
        ]
        if self.already_known:
            lines.append(f"Already known: {len(self.already_known)}")
        if self.needs_review:
            lines.append("Looks like an update to an existing memory (not saved; review it):")
            lines += [f"  ? new: {new}\n    existing: {old}" for new, old in self.needs_review]
        return "\n".join(lines)


def user_texts(messages: list[AnyMessage]) -> list[str]:
    """Only what the user typed: never tool results, never the assistant's words."""
    return [
        m.content.strip()
        for m in messages
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.strip()
    ]


async def extract_and_save(
    texts: list[str],
    extractor: Extractor,
    store: MemoryStore,
    report: ConsolidationReport,
    dry_run: bool = False,
) -> None:
    if not texts:
        return
    prompt = EXTRACTION_PROMPT.format(messages="\n".join(f"- {t}" for t in texts))
    result = await extractor.ainvoke(prompt)
    for fact in result.facts:
        similar = await store.search(fact.content, limit=1, min_similarity=RELATED_SIMILARITY)
        if similar and (similar[0].similarity or 0) >= 0.98:
            report.already_known.append(fact.content)
        elif similar:
            report.needs_review.append((fact.content, f"[{similar[0].id}] {similar[0].content}"))
        elif dry_run:
            report.saved.append(f"{fact.content} ({fact.kind}) [dry run: not saved]")
        else:
            saved = await store.add(fact.kind, fact.content, source="consolidation")
            report.saved.append(f"[{saved.id}] {saved.content} ({saved.kind})")


class ConsolidationProgress:
    """Remembers how many messages of each conversation were already reviewed."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def seen(self, thread_id: str) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT messages_seen FROM consolidation_progress WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()
        return row["messages_seen"] if row else 0

    async def mark(self, thread_id: str, count: int) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO consolidation_progress (thread_id, messages_seen) VALUES (%s, %s)
                ON CONFLICT (thread_id)
                DO UPDATE SET messages_seen = EXCLUDED.messages_seen, updated_at = now()
                """,
                (thread_id, count),
            )


async def consolidate(
    pool: AsyncConnectionPool,
    checkpointer: AsyncPostgresSaver,
    extractor: Extractor,
    store: MemoryStore,
    dry_run: bool = False,
    thread_prefix: str = "telegram:",
) -> ConsolidationReport:
    progress = ConsolidationProgress(pool)
    await progress.ensure_schema()
    report = ConsolidationReport()

    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE %s",
            (thread_prefix + "%",),
        )
        thread_ids = [r["thread_id"] for r in await cur.fetchall()]

    for thread_id in thread_ids:
        saved_state = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        if saved_state is None:
            continue
        messages = saved_state.checkpoint["channel_values"].get("messages", [])
        already = await progress.seen(thread_id)
        new = messages[already:]
        if not new:
            continue
        report.threads += 1
        texts = user_texts(new)
        report.new_messages += len(texts)
        await extract_and_save(texts, extractor, store, report, dry_run)
        if not dry_run:
            await progress.mark(thread_id, len(messages))
    return report


async def _main(dry_run: bool) -> None:
    from fuse.config import get_settings
    from fuse.db import open_pool
    from fuse.llm import build_structured_llm
    from fuse.memory.embeddings import FastEmbedder
    from fuse.memory.store import PostgresMemoryStore

    settings = get_settings()
    extractor = build_structured_llm(settings, ExtractedFacts, tier="light")
    async with open_pool(settings.database_url.get_secret_value()) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        store = PostgresMemoryStore(
            pool, FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir)
        )
        await store.ensure_schema()
        report = await consolidate(pool, checkpointer, extractor, store, dry_run=dry_run)
    print(report.summary())


def main() -> None:
    from fuse.compat import use_selector_event_loop_on_windows
    from fuse.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Save lasting facts from recent conversations")
    parser.add_argument("--dry-run", action="store_true", help="show, but don't save")
    args = parser.parse_args()
    use_selector_event_loop_on_windows()
    configure_logging("WARNING")
    asyncio.run(_main(args.dry_run))


if __name__ == "__main__":
    main()
