import os
import uuid

import pytest

from fuse.actions import PostgresActionStore, execute_once
from fuse.db import open_pool

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set"),
]


async def test_actions_are_stored_in_postgres_and_run_once():
    key = uuid.uuid4().hex
    async with open_pool(DB_URL) as pool:
        store = PostgresActionStore(pool)
        await store.ensure_schema()
        await store.ensure_schema()  # running it twice must be harmless

        row = await store.record(key, "send_email", {"to": "a@example.com"})
        assert row.status == "pending"
        assert row.args == {"to": "a@example.com"}

        runs = []

        async def send() -> str:
            runs.append(1)
            return "sent"

        await execute_once(store, key, send)

    # A fresh pool, like after a restart: the action is still recorded as done.
    async with open_pool(DB_URL) as pool:
        store = PostgresActionStore(pool)
        assert await execute_once(store, key, send) == "sent"
        row = await store.get(key)
    assert row.status == "executed"
    assert row.result == "sent"
    assert runs == [1]
