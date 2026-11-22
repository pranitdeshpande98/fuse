import os

import pytest

from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.evals import CASES, run_eval


async def test_eval_runs_and_counts_every_case():
    result = await run_eval(HashingEmbedder())
    assert result.expected_total == sum(len(c.expected) for c in CASES)
    assert result.unrelated_total == sum(len(c.unrelated) for c in CASES)


@pytest.mark.skipif(
    os.getenv("FUSE_TEST_EMBEDDINGS") != "1",
    reason="downloads the real model; set FUSE_TEST_EMBEDDINGS=1",
)
async def test_real_model_recall_quality():
    from fuse.config import get_settings
    from fuse.memory.embeddings import FastEmbedder

    settings = get_settings()
    result = await run_eval(FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir))
    # Measured with bge-small at 0.6: recall 80%, false recall 5%.
    assert result.recall >= 0.7, result.misses
    assert result.false_recall_rate <= 0.1, result.misses
