"""Conversations survive a restart when checkpoints are stored in Postgres."""

import os
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.db import open_pool
from tests.conftest import ScriptedModel

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set"),
]


async def test_conversation_survives_a_restart():
    thread = f"test:{uuid.uuid4().hex}"  # a fresh thread, so repeated runs don't interfere

    async with open_pool(DB_URL) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        first = ScriptedModel(script=[AIMessage(content="Nice to meet you.")])
        await AgentRunner(build_graph(first, [], checkpointer)).send(thread, "My name is Pranit")

    # "Restart": a brand-new pool, checkpointer, model and graph. Only the database remains.
    async with open_pool(DB_URL) as pool:
        second = ScriptedModel(script=[AIMessage(content="Pranit")])
        runner = AgentRunner(build_graph(second, [], AsyncPostgresSaver(pool)))
        assert await runner.send(thread, "What's my name?") == "Pranit"

    seen = [m.content for m in second.calls[0]]
    assert "My name is Pranit" in seen
