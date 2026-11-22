"""The Send button: only you can send, only to allowed recipients, and never twice."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.actions import ALREADY_SENDING, InMemoryActionStore, execute_at_most_once
from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.approval_ui import after_allowing, approval_buttons, approval_text
from fuse.contacts import (
    InMemoryContactStore,
    InvalidRecipient,
    contacts_text,
    parse_recipients,
    thread_participants,
)
from fuse.tools.drafts import LocalDraftSink, make_draft_tools
from tests.conftest import ScriptedModel, tool_call

THREAD = [
    {
        "from": "Pat Lee <pat@acme.com>",
        "to": "me@example.com",
        "cc": "sam@corp.com",
        "message_id": "<m1@acme.com>",
    },
]


class Reader:
    def __init__(self, thread=THREAD):
        self.thread = thread

    async def read_thread(self, thread_id):
        return self.thread if thread_id == "t-acme" else []

    async def upcoming_events(self, days_ahead):
        return []

    async def search_messages(self, query, max_results):
        return []


def make(args, contacts=None, reader=None):
    actions, sink = InMemoryActionStore(), LocalDraftSink()
    contacts = contacts if contacts is not None else InMemoryContactStore()
    model = ScriptedModel(script=[tool_call("draft_email_reply", args), AIMessage("Done.")])
    tools = make_draft_tools(actions, sink, contacts, reader or Reader())
    runner = AgentRunner(build_graph(model, tools, InMemorySaver()))
    return runner, sink, contacts, actions


REPLY = {
    "to": "pat@acme.com",
    "subject": "Re: Friday",
    "body": "Monday works.",
    "reply_to_thread_id": "t-acme",
}


def test_recipients_are_parsed_strictly():
    assert parse_recipients("Pat <Pat@Acme.com>, sam@corp.com", "pat@acme.com") == [
        "pat@acme.com",
        "sam@corp.com",
    ]
    for bad in ["not an address", "a@b", "x@@y.com"]:
        try:
            parse_recipients(bad)
        except InvalidRecipient:
            continue
        raise AssertionError(f"{bad!r} was accepted")


def test_thread_participants_include_from_to_and_cc():
    assert thread_participants(THREAD) == {"pat@acme.com", "me@example.com", "sam@corp.com"}


async def test_replying_in_a_thread_offers_send_and_send_sends():
    runner, sink, _, _ = make(REPLY)
    reply = await runner.send("chat", "reply to Pat that Monday works")
    payload = reply.approval
    assert payload["send_allowed"] and payload["blocked"] == []
    labels = [b.label for row in approval_buttons(payload, str) for b in row]
    assert labels == ["Send", "Save as draft", "Reject"]

    await runner.resume("chat", approved=True, mode="send")
    assert len(sink.sent) == 1 and sink.drafts == []


async def test_save_as_draft_never_sends():
    runner, sink, _, _ = make(REPLY)
    await runner.send("chat", "reply to Pat")
    await runner.resume("chat", approved=True)  # "Save as draft"
    assert len(sink.drafts) == 1 and sink.sent == []


async def test_a_new_recipient_blocks_send_until_you_allow_them():
    args = {"to": "dana@recruit.io", "subject": "Hello", "body": "Hi Dana"}
    runner, sink, contacts, _ = make(args)
    payload = (await runner.send("chat", "email Dana")).approval
    assert not payload["send_allowed"] and payload["blocked"] == ["dana@recruit.io"]
    labels = [b.label for row in approval_buttons(payload, str) for b in row]
    assert "Send" not in labels and "Always allow dana@recruit.io" in labels
    assert "Send isn't available: dana@recruit.io is not in your contacts" in approval_text(payload)

    await contacts.add("dana@recruit.io")  # what "Always allow" does
    labels = [b.label for row in approval_buttons(after_allowing(payload), str) for b in row]
    assert labels[0] == "Send"
    await runner.resume("chat", approved=True, mode="send")
    assert len(sink.sent) == 1


async def test_send_is_refused_in_code_even_if_the_button_is_forged():
    """The allowlist is checked again at the moment of sending, not just in the buttons."""
    args = {
        "to": "attacker@evil.com",
        "cc": "pat@acme.com",
        "subject": "x",
        "body": "y",
        "reply_to_thread_id": "t-acme",
    }
    runner, sink, _, _ = make(args)
    payload = (await runner.send("chat", "...")).approval
    assert payload["blocked"] == ["attacker@evil.com"]
    await runner.resume("chat", approved=True, mode="send")  # a forged "Send" decision
    assert sink.sent == []
    state = await runner._graph.aget_state({"configurable": {"thread_id": "chat"}})
    tool_result = [m for m in state.values["messages"] if m.type == "tool"][-1].content
    assert "Not sent" in tool_result and "attacker@evil.com" in tool_result


async def test_cc_is_shown_and_there_is_no_bcc():
    args = {**REPLY, "cc": "sam@corp.com"}
    runner, *_ = make(args)
    payload = (await runner.send("chat", "...")).approval
    assert "CC: sam@corp.com" in payload["preview"]
    assert "bcc" not in make_draft_tools(InMemoryActionStore(), LocalDraftSink())[0].args


async def test_send_happens_at_most_once_even_after_a_crash_mid_send():
    store = InMemoryActionStore()
    await store.record("k", "draft_email_reply", {})
    sent = []

    async def send():
        sent.append(1)
        return "Sent."

    await store.set_status("k", "sending")  # Fuse died after starting to send
    assert await execute_at_most_once(store, "k", send) == ALREADY_SENDING
    assert sent == []  # didn't risk a second copy

    await store.record("k2", "draft_email_reply", {})
    assert await execute_at_most_once(store, "k2", send) == "Sent."
    assert await execute_at_most_once(store, "k2", send) == "Sent."
    assert sent == [1]


async def test_contacts_command():
    contacts = InMemoryContactStore()
    assert "No approved contacts" in await contacts_text(contacts, "")
    assert "Added dana@recruit.io" in await contacts_text(contacts, "add Dana@Recruit.io")
    assert "dana@recruit.io" in await contacts_text(contacts, "")
    assert "doesn't look like" in await contacts_text(contacts, "add nonsense")
    assert "Removed" in await contacts_text(contacts, "remove dana@recruit.io")
    assert "Usage" in await contacts_text(contacts, "delete everything")


async def test_only_the_checked_normalised_addresses_reach_gmail():
    args = {**REPLY, "to": "Pat Lee <PAT@acme.com>"}
    runner, sink, _, _ = make(args)
    await runner.send("chat", "...")
    await runner.resume("chat", approved=True, mode="send")
    assert sink.sent[0]["to"] == "pat@acme.com"


async def test_a_malformed_recipient_list_is_refused_before_any_preview():
    runner, sink, _, actions = make({**REPLY, "to": "pat@acme.com, junk@@"})
    reply = await runner.send("chat", "...")
    assert reply.kind == "text"  # no approval was even offered
    assert sink.sent == [] and sink.drafts == []


async def test_postgres_contacts_and_the_sending_status():
    import os

    import pytest

    from fuse.actions import PostgresActionStore
    from fuse.contacts import PostgresContactStore
    from fuse.db import open_pool

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    async with open_pool(url) as pool:
        contacts = PostgresContactStore(pool)
        await contacts.ensure_schema()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM contacts")
        await contacts.add("Dana@Recruit.io")
        await contacts.add("dana@recruit.io")
        assert await contacts.all() == {"dana@recruit.io"}

        actions = PostgresActionStore(pool)
        await actions.ensure_schema()
        await actions.ensure_schema()  # the constraint migration is safe to repeat
        await actions.record("send-test", "draft_email_reply", {})
        await actions.set_status("send-test", "sending")
        assert (await actions.get("send-test")).status == "sending"
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM actions WHERE idempotency_key = 'send-test'")
