"""Embeddings: turn text into a list of numbers so similar meanings end up close together.

"I prefer afternoon meetings" and "schedule calls after lunch" share almost no words, but
their embeddings are close, which is what lets Fuse recall memories by meaning.
"""

import asyncio
import hashlib
import math
import re
from typing import Protocol

# bge-small-en-v1.5 produces 384 numbers per text. The memories table is created with this
# size, so changing the model to one with a different size needs a new table.
EMBEDDING_DIM = 384


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """A small embedding model that runs locally on the CPU. Nothing is sent anywhere.

    The model (about 130 MB) is downloaded once on first use and cached in `cache_dir`.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None  # loaded on first use, so startup stays fast

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily: it's a heavy import

            self._model = TextEmbedding(self._model_name, cache_dir=self._cache_dir)
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._load().embed(texts)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Embedding is CPU work; a thread keeps the bot responsive meanwhile.
        return await asyncio.to_thread(self._embed_sync, texts)


class HashingEmbedder:
    """A tiny stand-in for tests: no download, deterministic, and texts that share words
    get similar vectors. Not for real use: it matches words, not meaning."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
