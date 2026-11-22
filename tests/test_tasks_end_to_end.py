"""The whole background-task path against real Postgres: chat starts a task, a worker runs
it and pauses for approval, the bot records your tap, the worker is restarted, and a new
worker finishes the task. Nothing is lost and the action runs once."""

import os

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from fuse.actions import PostgresActionStore
from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.contacts import PostgresContactStore
from fuse.db import open_pool
from fuse.proactive.notify import RecordingNotifier
from fuse.tasks.queue import PostgresTaskStore
from fuse.tasks.report import TelegramTaskReporter, allow_task_recipients, decide_task_approval
from fuse.tasks.tools import make_task_tools
from fuse.tasks.worker import Worker
from fuse.tools.drafts import LocalDraftSink, make_draft_tools
from tests.conftest import ScriptedModel, tool_call

DB_URL = os.environ.get("TEST_DATABASE_URL")
EMAIL = {"to": "dana@recruit.io", "subject": "Availability", "body": "Tuesday works."}


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_task_survives_a_restart_while_waiting_for_send():
    async with open_pool(DB_URL) as pool:
        async with pool.connection() as conn:
            for table in ("tasks", "contacts"):
                await conn.execute(f"DROP TABLE IF EXISTS {table}")

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        actions, tasks = PostgresActionStore(pool), PostgresTaskStore(pool)
        contacts = PostgresContactStore(pool)
        for store in (actions, tasks, contacts):
            await store.ensure_schema()
        async with pool.connection() as conn:
            # Action keys come from thread + tool call IDs, which repeat between test runs.
            await conn.execute("DELETE FROM actions WHERE tool = 'draft_email_reply'")
        sink, notifier = LocalDraftSink(), RecordingNotifier()

        # 1. In chat, the agent hands the job to a background task.
        chat_model = ScriptedModel(
            script=[
                tool_call("start_background_task", {"goal": "Reply to Dana with my availability"}),
                AIMessage("Started it."),
            ]
        )
        chat = AgentRunner(build_graph(chat_model, make_task_tools(tasks), checkpointer))
        await checkpointer.adelete_thread("telegram:e2e")
        await chat.send("telegram:e2e", "reply to the recruiter")
        (task,) = await tasks.recent(1)
        # The tasks table was recreated, so IDs start at 1 again; clear old test threads.
        await checkpointer.adelete_thread(task.thread_id)

        def worker(name, script):
            model = ScriptedModel(script=script)
            tools = make_draft_tools(actions, sink, contacts)
            runner = AgentRunner(build_graph(model, tools, checkpointer))
            return Worker(tasks, runner, TelegramTaskReporter(notifier), worker_id=name)

        # 2. A worker runs it until it wants to send an email.
        first = worker("w1", [tool_call("draft_email_reply", EMAIL)])
        await first.run_one()
        text, buttons = notifier.sent[-1]
        assert "dana@recruit.io is not in your contacts" in text
        allow = next(b for b in buttons if b.label.startswith("Always allow"))

        # 3. You tap "Always allow", then "Send" (the bot does this; no worker involved).
        _, new_buttons = await allow_task_recipients(tasks, contacts, allow.data)
        send = next(b for row in new_buttons for b in row if b.label == "Send")
        assert "Sending" in await decide_task_approval(tasks, send.data)

        # 4. The first worker is gone (restart). A new one, with a fresh model, finishes.
        second = worker("w2", [AIMessage("Sent Dana your availability.")])
        await second.run_one()
        done = await tasks.get(task.id)
        assert (done.status, done.result) == ("done", "Sent Dana your availability.")
        assert len(sink.sent) == 1 and sink.drafts == []
        assert notifier.texts[-1].endswith("Sent Dana your availability.")
