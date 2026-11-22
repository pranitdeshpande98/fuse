from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.google.auth import GoogleNotConnected
from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.store import InMemoryMemoryStore
from fuse.proactive.brief import MorningBrief, format_inputs, gather, plain_brief
from fuse.proactive.notify import RecordingNotifier
from fuse.proactive.schedule import FixedClock
from fuse.tools.basic import make_basic_tools
from tests.conftest import ScriptedModel
from tests.test_google_tools import FakeReader

CHICAGO = ZoneInfo("America/Chicago")
MONDAY_7_30 = datetime(2026, 11, 23, 7, 30, tzinfo=CHICAGO)

EVENTS = [
    {"summary": "Standup", "start": {"dateTime": "2026-11-23T09:00:00-06:00"}},
    # 23:30 UTC is 17:30 in Chicago: still today
    {"summary": "Call with Pat", "start": {"dateTime": "2026-11-23T23:30:00Z"}},
    {"summary": "Tomorrow's review", "start": {"dateTime": "2026-11-24T08:00:00-06:00"}},
    {"summary": "Thanksgiving prep", "start": {"date": "2026-11-23"}},
]
EMAILS = [{"from": "Pat <pat@acme.com>", "subject": "Proposal", "snippet": "Can we move Friday?"}]


class Held:
    def __init__(self, items):
        self.items = items

    async def take_held(self):
        items, self.items = self.items, []
        return items


async def test_gather_keeps_only_today_in_the_users_timezone():
    inputs = await gather(FixedClock(MONDAY_7_30), FakeReader(events=EVENTS, messages=EMAILS))
    titles = [e["summary"] for e in inputs.events]
    assert titles == ["Standup", "Call with Pat", "Thanksgiving prep"]
    assert inputs.emails == EMAILS


async def test_brief_input_wraps_calendar_and_email_as_untrusted():
    inputs = await gather(FixedClock(MONDAY_7_30), FakeReader(events=EVENTS, messages=EMAILS))
    text = format_inputs(inputs, CHICAGO)
    assert "17:30 Call with Pat" in text
    assert text.count("<untrusted_content>") == 2
    assert text.index("<untrusted_content>") < text.index("Pat <pat@acme.com>")


async def test_brief_is_written_by_the_llm_and_includes_your_profile():
    memory = InMemoryMemoryStore(HashingEmbedder())
    await memory.add("profile", "Prefers afternoon meetings")
    llm = ScriptedModel(script=[AIMessage("Good morning! Standup at 9.")])
    notifier = RecordingNotifier()
    brief = MorningBrief(FixedClock(MONDAY_7_30), notifier, llm, FakeReader(events=EVENTS), memory)
    await brief.run()
    assert notifier.texts == ["Good morning! Standup at 9."]
    prompt = llm.calls[0][-1].content
    assert "Prefers afternoon meetings" in prompt


async def test_brief_falls_back_to_a_plain_list_when_the_llm_fails():
    class BrokenLLM:
        async def ainvoke(self, _):
            raise RuntimeError("rate limited")

    notifier = RecordingNotifier()
    reader = FakeReader(events=EVENTS, messages=EMAILS)
    await MorningBrief(FixedClock(MONDAY_7_30), notifier, BrokenLLM(), reader).run()
    (text,) = notifier.texts
    assert "09:00 Standup" in text and "Pat <pat@acme.com>: Proposal" in text


async def test_brief_still_arrives_when_google_sign_in_expired():
    notifier = RecordingNotifier()
    reader = FakeReader(error=GoogleNotConnected("expired"))
    await MorningBrief(FixedClock(MONDAY_7_30), notifier, None, reader).run()
    (text,) = notifier.texts
    assert "No meetings" in text
    assert "uv run python -m fuse.google.auth" in text


async def test_held_items_are_included_once():
    held = Held(["Pat: Proposal (arrived 23:10)"])
    first = await gather(FixedClock(MONDAY_7_30), FakeReader(), held=held)
    second = await gather(FixedClock(MONDAY_7_30), FakeReader(), held=held)
    assert "Pat: Proposal" in plain_brief(first, CHICAGO)
    assert second.held == []


async def test_brief_goes_into_the_conversation_so_you_can_reply_to_it():
    model = ScriptedModel(script=[AIMessage("Drafting a reply to Pat.")])
    runner = AgentRunner(build_graph(model, make_basic_tools("UTC"), InMemorySaver()))
    notifier = RecordingNotifier()
    brief = MorningBrief(
        FixedClock(MONDAY_7_30),
        notifier,
        None,
        FakeReader(messages=EMAILS),
        on_sent=lambda text: runner.add_note("telegram:42", text),
    )
    await brief.run()
    await runner.send("telegram:42", "reply to the first email")
    seen = [m.content for m in model.calls[0]]
    assert any("Pat <pat@acme.com>: Proposal" in str(c) for c in seen)
    assert seen[-1] == "reply to the first email"
