import pytest

from fuse.actions import InMemoryActionStore, execute_once, idempotency_key


def test_idempotency_key_is_stable_and_unique():
    assert idempotency_key("t1", "call_1") == idempotency_key("t1", "call_1")
    assert idempotency_key("t1", "call_1") != idempotency_key("t1", "call_2")
    assert idempotency_key("t1", "call_1") != idempotency_key("t2", "call_1")


async def test_recording_the_same_key_twice_keeps_the_first():
    store = InMemoryActionStore()
    await store.record("k", "send_email", {"to": "a@example.com"})
    await store.record("k", "send_email", {"to": "changed@example.com"})
    assert (await store.get("k")).args == {"to": "a@example.com"}


async def test_an_action_never_runs_twice():
    store = InMemoryActionStore()
    await store.record("k", "send_email", {})
    sent = []

    async def send() -> str:
        sent.append("email")
        return "sent"

    assert await execute_once(store, "k", send) == "sent"
    assert await execute_once(store, "k", send) == "sent"  # e.g. a retry after a crash
    assert sent == ["email"]


async def test_a_failure_is_recorded_and_can_be_retried():
    store = InMemoryActionStore()
    await store.record("k", "send_email", {})

    async def broken() -> str:
        raise ConnectionError("mail server down")

    with pytest.raises(ConnectionError):
        await execute_once(store, "k", broken)
    assert (await store.get("k")).status == "failed"

    async def works() -> str:
        return "sent"

    assert await execute_once(store, "k", works) == "sent"
    assert (await store.get("k")).status == "executed"
