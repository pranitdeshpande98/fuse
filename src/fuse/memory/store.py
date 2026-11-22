"""Long-term memory: facts about the user, stored with embeddings in Postgres (pgvector).

Two kinds of memory:
- "profile": short, stable facts always shown to the agent (name, timezone, preferences).
- "fact": everything else worth remembering, found by meaning when relevant.

Memories are never edited in place. Updating one inserts a new row and marks the old one
as superseded, so there is always a history of what Fuse believed and when it changed.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from psycopg_pool import AsyncConnectionPool

from fuse.memory.embeddings import EMBEDDING_DIM, Embedder, cosine_similarity

MemoryKind = Literal["profile", "fact"]

# Saving something almost identical to an existing memory returns the existing one instead.
# Kept high on purpose: with bge-small, "meetings in the afternoon" vs "...in the morning"
# already scores ~0.94, and those are different (contradicting) facts, not duplicates.
DUPLICATE_SIMILARITY = 0.98

SCHEMA = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    f"""
    CREATE TABLE IF NOT EXISTS memories (
        id            BIGSERIAL PRIMARY KEY,
        kind          TEXT        NOT NULL CHECK (kind IN ('profile', 'fact')),
        content       TEXT        NOT NULL,
        embedding     vector({EMBEDDING_DIM}) NOT NULL,
        source        TEXT        NOT NULL DEFAULT 'chat',
        superseded_by BIGINT      REFERENCES memories (id) ON DELETE SET NULL,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS memories_active_idx ON memories (kind) WHERE superseded_by IS NULL",
]


@dataclass
class Memory:
    id: int
    kind: MemoryKind
    content: str
    created_at: datetime | None = None
    similarity: float | None = None  # set on search results


class MemoryStore(Protocol):
    async def add(self, kind: MemoryKind, content: str, source: str = "chat") -> Memory: ...
    async def search(
        self, query: str, limit: int = 5, min_similarity: float = 0.0
    ) -> list[Memory]: ...
    async def list(self, kind: MemoryKind | None = None) -> list[Memory]: ...
    async def get(self, memory_id: int) -> Memory | None: ...
    async def supersede(self, memory_id: int, new_content: str, source: str = "chat") -> Memory: ...
    async def delete(self, memory_id: int) -> bool: ...


def _to_vector_literal(vector: list[float]) -> str:
    """pgvector accepts vectors as text like '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


class PostgresMemoryStore:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder) -> None:
        self._pool = pool
        self._embedder = embedder

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def _embed(self, text: str) -> str:
        (vector,) = await self._embedder.embed([text])
        return _to_vector_literal(vector)

    async def _nearest(self, vector: str, kind: MemoryKind | None, limit: int) -> list[Memory]:
        kind_filter = "AND kind = %s" if kind else ""
        params: list = [vector]
        if kind:
            params.append(kind)
        params += [vector, limit]
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT id, kind, content, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE superseded_by IS NULL {kind_filter}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            )
            rows = await cur.fetchall()
        return [
            Memory(r["id"], r["kind"], r["content"], r["created_at"], r["similarity"]) for r in rows
        ]

    async def add(self, kind: MemoryKind, content: str, source: str = "chat") -> Memory:
        content = content.strip()
        vector = await self._embed(content)
        nearest = await self._nearest(vector, kind, limit=1)
        if nearest and (nearest[0].similarity or 0) >= DUPLICATE_SIMILARITY:
            return nearest[0]  # already known: don't store a duplicate
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO memories (kind, content, embedding, source)
                VALUES (%s, %s, %s::vector, %s)
                RETURNING id, kind, content, created_at
                """,
                (kind, content, vector, source),
            )
            r = await cur.fetchone()
        return Memory(r["id"], r["kind"], r["content"], r["created_at"])

    async def search(self, query: str, limit: int = 5, min_similarity: float = 0.0) -> list[Memory]:
        found = await self._nearest(await self._embed(query), None, limit)
        return [m for m in found if (m.similarity or 0) >= min_similarity]

    async def list(self, kind: MemoryKind | None = None) -> list[Memory]:
        kind_filter = "AND kind = %s" if kind else ""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                SELECT id, kind, content, created_at FROM memories
                WHERE superseded_by IS NULL {kind_filter}
                ORDER BY kind DESC, created_at
                """,
                [kind] if kind else [],
            )
            rows = await cur.fetchall()
        return [Memory(r["id"], r["kind"], r["content"], r["created_at"]) for r in rows]

    async def get(self, memory_id: int) -> Memory | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, kind, content, created_at FROM memories "
                "WHERE id = %s AND superseded_by IS NULL",
                (memory_id,),
            )
            r = await cur.fetchone()
        return Memory(r["id"], r["kind"], r["content"], r["created_at"]) if r else None

    async def supersede(self, memory_id: int, new_content: str, source: str = "chat") -> Memory:
        old = await self.get(memory_id)
        if old is None:
            raise KeyError(f"No active memory with id {memory_id}")
        vector = await self._embed(new_content.strip())
        async with self._pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                """
                INSERT INTO memories (kind, content, embedding, source)
                VALUES (%s, %s, %s::vector, %s)
                RETURNING id, kind, content, created_at
                """,
                (old.kind, new_content.strip(), vector, source),
            )
            r = await cur.fetchone()
            await conn.execute(
                "UPDATE memories SET superseded_by = %s WHERE id = %s", (r["id"], memory_id)
            )
        return Memory(r["id"], r["kind"], r["content"], r["created_at"])

    async def delete(self, memory_id: int) -> bool:
        """Forget for real: the user asked, so the row and its history are removed."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                WITH RECURSIVE chain AS (
                    SELECT id FROM memories WHERE id = %s
                    UNION ALL
                    SELECT m.id FROM memories m JOIN chain c ON m.superseded_by = c.id
                )
                DELETE FROM memories WHERE id IN (SELECT id FROM chain)
                """,
                (memory_id,),
            )
            return cur.rowcount > 0


class InMemoryMemoryStore:
    """Same behaviour as PostgresMemoryStore, kept in a list. For tests."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._rows: list[dict] = []
        self._next_id = 1

    def _active(self, kind: MemoryKind | None = None) -> list[dict]:
        return [
            r
            for r in self._rows
            if r["superseded_by"] is None and (kind is None or r["kind"] == kind)
        ]

    async def _ranked(self, query: str, kind: MemoryKind | None) -> list[Memory]:
        (q,) = await self._embedder.embed([query])
        scored = [
            Memory(r["id"], r["kind"], r["content"], None, cosine_similarity(q, r["embedding"]))
            for r in self._active(kind)
        ]
        return sorted(scored, key=lambda m: m.similarity or 0, reverse=True)

    async def _insert(self, kind: MemoryKind, content: str) -> Memory:
        (vector,) = await self._embedder.embed([content])
        row = {
            "id": self._next_id,
            "kind": kind,
            "content": content,
            "embedding": vector,
            "superseded_by": None,
        }
        self._rows.append(row)
        self._next_id += 1
        return Memory(row["id"], kind, content)

    async def add(self, kind: MemoryKind, content: str, source: str = "chat") -> Memory:
        content = content.strip()
        nearest = await self._ranked(content, kind)
        if nearest and (nearest[0].similarity or 0) >= DUPLICATE_SIMILARITY:
            return nearest[0]
        return await self._insert(kind, content)

    async def search(self, query: str, limit: int = 5, min_similarity: float = 0.0) -> list[Memory]:
        ranked = await self._ranked(query, None)
        return [m for m in ranked if (m.similarity or 0) >= min_similarity][:limit]

    async def list(self, kind: MemoryKind | None = None) -> list[Memory]:
        rows = sorted(self._active(kind), key=lambda r: (r["kind"] != "profile", r["id"]))
        return [Memory(r["id"], r["kind"], r["content"]) for r in rows]

    async def get(self, memory_id: int) -> Memory | None:
        for r in self._active():
            if r["id"] == memory_id:
                return Memory(r["id"], r["kind"], r["content"])
        return None

    async def supersede(self, memory_id: int, new_content: str, source: str = "chat") -> Memory:
        old = await self.get(memory_id)
        if old is None:
            raise KeyError(f"No active memory with id {memory_id}")
        new = await self._insert(old.kind, new_content.strip())
        for r in self._rows:
            if r["id"] == memory_id:
                r["superseded_by"] = new.id
        return new

    async def delete(self, memory_id: int) -> bool:
        doomed = {memory_id}
        changed = True
        while changed:  # also remove older versions that this one replaced
            changed = False
            for r in self._rows:
                if r["superseded_by"] in doomed and r["id"] not in doomed:
                    doomed.add(r["id"])
                    changed = True
        before = len(self._rows)
        self._rows = [r for r in self._rows if r["id"] not in doomed]
        return len(self._rows) < before
