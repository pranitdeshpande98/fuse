"""Embeddings. The real-model test downloads ~130 MB once, so it is opt-in:
set FUSE_TEST_EMBEDDINGS=1 to run it."""

import os

import pytest

from fuse.memory.embeddings import EMBEDDING_DIM, FastEmbedder, HashingEmbedder, cosine_similarity


async def test_hashing_embedder_is_deterministic_and_sized():
    embedder = HashingEmbedder()
    a, b = await embedder.embed(["afternoon meetings", "afternoon meetings"])
    assert a == b
    assert len(a) == EMBEDDING_DIM


async def test_hashing_embedder_shared_words_are_closer():
    embedder = HashingEmbedder()
    base, near, far = await embedder.embed(
        ["client meeting on Friday", "Friday meeting with the client", "green colour"]
    )
    assert cosine_similarity(base, near) > cosine_similarity(base, far)


@pytest.mark.skipif(not os.environ.get("FUSE_TEST_EMBEDDINGS"), reason="downloads a model")
async def test_real_model_matches_by_meaning_not_words():
    embedder = FastEmbedder("BAAI/bge-small-en-v1.5", cache_dir=".cache/fastembed")
    base, near, far = await embedder.embed(
        [
            "I prefer meetings in the afternoon",
            "Please schedule my calls after lunch",  # no words in common, same meaning
            "My favourite colour is green",
        ]
    )
    assert len(base) == EMBEDDING_DIM
    assert cosine_similarity(base, near) > cosine_similarity(base, far)
