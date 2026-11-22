"""Token usage per model per day, so you can see how much of the free quota is left.

Every chat model Fuse builds carries one callback that adds each reply's token count to
`llm_usage` (both the bot and the worker write to it, so the total is the real one). Groq's
free tier allows about 200,000 tokens a day *per model*; `/status` shows where you are, and
a Telegram note arrives once a day when a model passes 80%.

Days are counted in UTC. Groq's own counter is close to this but not identical, so treat the
numbers as a good estimate, not the provider's bill.
"""

import logging
from datetime import UTC, date, datetime
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from psycopg_pool import AsyncConnectionPool

from fuse.proactive.jobs import JobLog
from fuse.proactive.notify import Notifier
from fuse.tasks.budget import tokens_used

log = logging.getLogger("fuse.usage")

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS llm_usage (
        day       DATE    NOT NULL,
        model     TEXT    NOT NULL,
        tokens    BIGINT  NOT NULL DEFAULT 0,
        requests  INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (day, model)
    )
    """,
]
WARN_AT = 0.8


def today() -> date:
    return datetime.now(UTC).date()


class PostgresUsageStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def record(self, day: date, model: str, tokens: int) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO llm_usage (day, model, tokens, requests) VALUES (%s, %s, %s, 1)
                ON CONFLICT (day, model) DO UPDATE SET
                    tokens = llm_usage.tokens + EXCLUDED.tokens,
                    requests = llm_usage.requests + 1
                """,
                (day, model, tokens),
            )

    async def on(self, day: date) -> dict[str, tuple[int, int]]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT model, tokens, requests FROM llm_usage WHERE day = %s ORDER BY model",
                (day,),
            )
            return {r["model"]: (r["tokens"], r["requests"]) for r in await cur.fetchall()}


class InMemoryUsageStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[date, str], list[int]] = {}

    async def record(self, day: date, model: str, tokens: int) -> None:
        row = self.rows.setdefault((day, model), [0, 0])
        row[0] += tokens
        row[1] += 1

    async def on(self, day: date) -> dict[str, tuple[int, int]]:
        return {m: (t, r) for (d, m), (t, r) in sorted(self.rows.items()) if d == day}


_store: Any = None  # set once per process at startup


def set_usage_store(store: Any) -> None:
    global _store
    _store = store


def model_name(response: LLMResult) -> str:
    output = response.llm_output or {}
    if output.get("model_name"):
        return str(output["model_name"])
    for generations in response.generations:
        for generation in generations:
            meta = getattr(getattr(generation, "message", None), "response_metadata", {}) or {}
            if meta.get("model_name"):
                return str(meta["model_name"])
    return "unknown"


class UsageCallback(AsyncCallbackHandler):
    """Attached to every chat model. Never breaks a reply: a failed write is only logged."""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if _store is None:
            return
        tokens = tokens_used(response)
        if not tokens:
            return
        try:
            await _store.record(today(), model_name(response), tokens)
        except Exception:
            log.warning("couldn't record token usage", exc_info=True)


USAGE_CALLBACK = UsageCallback()


def usage_lines(usage: dict[str, tuple[int, int]], quota: int) -> list[str]:
    if not usage:
        return ["No model calls yet today."]
    lines = []
    for model, (tokens, requests) in usage.items():
        share = f" ({tokens / quota:.0%} of {quota:,})" if quota else ""
        lines.append(f"- {model}: {tokens:,} tokens in {requests} calls{share}")
    return lines


class QuotaWatcher:
    """Once a day per model, a Telegram note when usage passes 80% of the quota."""

    def __init__(self, store: Any, notifier: Notifier, job_log: JobLog, quota: int) -> None:
        self._store, self._notifier, self._job_log, self._quota = store, notifier, job_log, quota

    async def check(self) -> list[str]:
        if not self._quota:
            return []
        day = today()
        warned = []
        for model, (tokens, _) in (await self._store.on(day)).items():
            if tokens < self._quota * WARN_AT:
                continue

            async def send(model: str = model, tokens: int = tokens) -> None:
                await self._notifier.send(
                    f"Heads up: {model} has used {tokens:,} tokens today, "
                    f"{tokens / self._quota:.0%} of its ~{self._quota:,} free daily quota. "
                    "When it runs out I switch to the fallback model, and big background "
                    "tasks may fail until the quota resets."
                )

            if await self._job_log.run_once(f"quota_warning:{model}", day.isoformat(), send):
                warned.append(model)
        return warned
