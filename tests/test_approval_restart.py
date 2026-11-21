"""An approval survives a restart: pause, rebuild everything, approve, and it runs once."""

import os
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from fuse.actions import PostgresActionStore
from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.db import open_pool
from fuse.tools.drafts import LocalDraftSink, make_draft_tools
from tests.conftest import ScriptedModel, tool_call

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set"),
]
DRAFT = {"to": "pat@example.com", "subject": "Re: Proposal", "body": "Friday works."}


async def test_approval_survives_a_restart():
    thread = f"test:{uuid.uuid4().hex}"

    async with open_pool(DB_URL) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        store = PostgresActionStore(pool)
        await store.ensure_schema()
        sink_before = LocalDraftSink()
        model = ScriptedModel(script=[tool_call("draft_email_reply", DRAFT)])
        graph = build_graph(model, make_draft_tools(store, sink_before), checkpointer)
        reply = await AgentRunner(graph).send(thread, "Reply to Pat")
        assert reply.kind == "approval"
        key = reply.approval["action_key"]

    # "Restart": new pool, model, tools and graph. Only Postgres carries over.
    async with open_pool(DB_URL) as pool:
        store = PostgresActionStore(pool)
        sink_after = LocalDraftSink()
        model = ScriptedModel(script=[AIMessage(content="Draft saved.")])
        graph = build_graph(model, make_draft_tools(store, sink_after), AsyncPostgresSaver(pool))
        runner = AgentRunner(graph)

        assert (await runner.pending_approval(thread))["action_key"] == key
        reply = await runner.resume(thread, approved=True)
        assert reply.text == "Draft saved."
        assert (await store.get(key)).status == "executed"

    assert sink_before.drafts == []
    assert len(sink_after.drafts) == 1
