"""The channel-neutral core: what every channel (Telegram now, WhatsApp later) relies on."""

from datetime import timedelta

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.actions import InMemoryActionStore
from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.channels.core import COMMAND_NAMES, ChatCore
from fuse.contacts import InMemoryContactStore
from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.store import InMemoryMemoryStore
from fuse.proactive.alerts import AlertActions, alert_buttons
from fuse.proactive.events import InMemoryEventStore
from fuse.proactive.store import InMemoryMuteStore
from fuse.tasks.queue import InMemoryTaskStore
from fuse.tools.drafts import LocalDraftSink, make_draft_tools
from tests.conftest import ScriptedModel, tool_call

THREAD = "chat:1"
EMAIL = {"to": "dana@recruit.io", "subject": "Hi", "body": "Hello"}


def make_core(script, **stores):
    actions, sink = InMemoryActionStore(), LocalDraftSink()
    contacts = stores.pop("contacts", InMemoryContactStore())
    model = ScriptedModel(script=script)
    runner = AgentRunner(
        build_graph(model, make_draft_tools(actions, sink, contacts), InMemorySaver())
    )
    core = ChatCore(runner, contacts=contacts, **stores)
    return core, sink, contacts


def test_command_names_match_what_the_core_handles():
    core, *_ = make_core([])
    assert set(core.commands) == set(COMMAND_NAMES)


async def test_plain_reply_and_long_replies_are_split():
    core, *_ = make_core([AIMessage("x\n" * 3000)])
    out = await core.message(THREAD, "hi")
    assert len(out) == 2 and not out[0].buttons


async def test_email_to_a_new_person_then_always_allow_then_send():
    core, sink, contacts = make_core([tool_call("draft_email_reply", EMAIL), AIMessage("Sent.")])
    (preview,) = await core.message(THREAD, "email Dana")
    labels = [b.label for row in preview.buttons for b in row]
    assert labels == ["Save as draft", "Reject", "Always allow dana@recruit.io"]
    allow = preview.buttons[1][0].data

    result = await core.button(THREAD, allow)
    assert result.toast == "Added to your contacts"
    assert [b.label for b in result.buttons[0]] == ["Send", "Save as draft", "Reject"]
    assert await contacts.all() == {"dana@recruit.io"}

    send = result.buttons[0][0].data
    result = await core.button(THREAD, send)
    assert result.toast == "Sending" and result.buttons is None  # buttons removed
    assert len(sink.sent) == 1
    assert result.messages[-1].text == "Sent."

    stale = await core.button(THREAD, send)  # the same button again does nothing
    assert stale.toast == "This is no longer waiting for approval." and len(sink.sent) == 1


async def test_alert_and_task_buttons_are_routed():
    events, mutes = InMemoryEventStore(), InMemoryMuteStore()
    event = await events.add("email", "gmail:1", {"from": "Pat <pat@acme.com>", "thread_id": "t"})
    core, *_ = make_core([], mutes=mutes)
    core.alert_actions = AlertActions(events, mutes, core.runner)
    mute = next(b for b in alert_buttons(event) if b.label == "Mute sender")
    result = await core.button(THREAD, mute.data)
    assert "Muted pat@acme.com" in result.messages[0].text

    tasks = InMemoryTaskStore()
    core.tasks = tasks
    task = await tasks.create("goal")
    await tasks.claim("w", timedelta(minutes=2))
    await tasks.wait_for_approval(task.id, "w", {"action_key": "abcdef1234567890xyz"})
    result = await core.button(THREAD, f"tk:approve:{task.id}:abcdef1234567890")
    assert result.toast == f"Approved. Task #{task.id} continues."


async def test_commands():
    memory = InMemoryMemoryStore(HashingEmbedder())
    await memory.add("profile", "Prefers mornings")
    briefs = []

    async def brief():
        briefs.append(1)

    core, *_ = make_core([], memory=memory, tasks=InMemoryTaskStore(), brief=brief)
    assert "Prefers mornings" in (await core.command(THREAD, "memories"))[0].text
    assert "Started background task #1" in (await core.command(THREAD, "task", "list jobs"))[0].text
    assert "#1 [queued]" in (await core.command(THREAD, "tasks"))[0].text
    assert await core.command(THREAD, "brief") == [] and briefs == [1]
    assert "don't know" in (await core.command(THREAD, "nope"))[0].text
    assert "aren't set up" in (await core.command(THREAD, "unmute"))[0].text


async def test_agent_errors_become_a_friendly_message():
    core, *_ = make_core([])  # the scripted model has nothing to say: it raises
    (out,) = await core.message(THREAD, "hi")
    assert out.text.startswith("Something went wrong")


async def test_scan_and_dashboard_commands():
    from fuse.jobalerts.client import JobAlertsClient
    from fuse.jobalerts.service import ScanRunner
    from fuse.proactive.notify import RecordingNotifier
    from tests.fake_job_alerts import FakeJobAlerts

    core, *_ = make_core([])
    assert "isn't set up" in (await core.command(THREAD, "scan"))[0].text
    fake = FakeJobAlerts()
    core.job_alerts = JobAlertsClient("http://127.0.0.1:5000", fake.transport())
    core.scan_runner = ScanRunner(core.job_alerts, RecordingNotifier())
    assert "Scanning" in (await core.command(THREAD, "scan"))[0].text
    await core.scan_runner.wait()
    assert fake.scans == 1
    dashboard = (await core.command(THREAD, "dashboard"))[0].text
    assert "2 of 3 portals active" in dashboard and "Duke" in dashboard
    fake.down = True
    assert "isn't running" in (await core.command(THREAD, "dashboard"))[0].text


async def test_track_button_adds_a_job_match_to_your_applications():
    from fuse.jobs.store import InMemoryJobStore

    events, jobs = InMemoryEventStore(), InMemoryJobStore()
    match = {
        "job_key": "k1",
        "company": "RAND",
        "title": "SAP Analyst",
        "location": "Santa Monica, CA",
        "url": "https://rand.example/1",
    }
    event = await events.add("job", "job:k1", match)
    core, *_ = make_core([], events=events, jobs=jobs)

    result = await core.button(THREAD, f"jt:{event.id}")
    assert result.toast == "Tracked" and result.only_this_button
    assert "Added to your applications: RAND, SAP Analyst [interested]" in result.messages[0].text
    (app,) = await jobs.list()
    assert app.url == "https://rand.example/1" and "Santa Monica" in app.notes

    again = await core.button(THREAD, f"jt:{event.id}")
    assert "Already tracked" in again.messages[0].text and len(await jobs.list()) == 1
    assert "can't find" in (await core.button(THREAD, "jt:999")).toast
    email = await events.add("email", "gmail:1", {"from": "x"})
    assert "can't find" in (await core.button(THREAD, f"jt:{email.id}")).toast


def test_without_button_keeps_the_other_track_buttons():
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    from fuse.channels.telegram_bot import without_button

    markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton(f"Track {i}", callback_data=f"jt:{i}") for i in (1, 2, 3)]]
    )
    left = without_button(markup, "jt:2")
    assert [b.callback_data for b in left.inline_keyboard[0]] == ["jt:1", "jt:3"]
    single = InlineKeyboardMarkup([[InlineKeyboardButton("Track 1", callback_data="jt:1")]])
    assert without_button(single, "jt:1") is None


async def test_reminder_command_and_buttons_are_routed():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from fuse.proactive.schedule import FixedClock
    from fuse.reminders.store import InMemoryReminderStore

    clock = FixedClock(datetime(2026, 11, 22, 10, 0, tzinfo=ZoneInfo("America/Chicago")))
    store = InMemoryReminderStore()
    r = await store.add("Call Pat", clock.now())
    core, *_ = make_core([], reminders=store, clock=clock)
    assert "Call Pat" in (await core.command(THREAD, "reminders"))[0].text
    assert (await core.button(THREAD, f"rm:done:{r.id}")).toast == "Done."


async def test_applied_button_marks_applied_and_sets_a_follow_up_reminder():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from fuse.jobs.store import InMemoryJobStore
    from fuse.proactive.schedule import FixedClock
    from fuse.reminders.store import InMemoryReminderStore

    clock = FixedClock(datetime(2026, 11, 22, 10, 0, tzinfo=ZoneInfo("America/Chicago")))
    events, jobs, reminders = InMemoryEventStore(), InMemoryJobStore(), InMemoryReminderStore()
    match = {
        "job_key": "k1",
        "company": "RAND",
        "title": "SAP Analyst",
        "url": "https://rand.example/1",
    }
    event = await events.add("job", "job:k1", match)
    core, *_ = make_core([], events=events, jobs=jobs, reminders=reminders, clock=clock)

    result = await core.button(THREAD, f"ja:{event.id}")
    assert result.only_this_row
    assert "Marked applied: RAND, SAP Analyst" in result.messages[0].text
    assert "Wed 02 Dec at 9:00" in result.messages[0].text
    (app,) = await jobs.list()
    assert (app.stage, str(app.last_contact)) == ("applied", "2026-11-22")
    (reminder,) = await reminders.upcoming()
    assert "Follow up on your RAND application" in reminder.text
    assert reminder.due_at == datetime(2026, 12, 2, 9, 0, tzinfo=ZoneInfo("America/Chicago"))

    # Track afterwards must not move it back to "interested"
    tracked = await core.button(THREAD, f"jt:{event.id}")
    assert "Already tracked: RAND, SAP Analyst [applied]" in tracked.messages[0].text
    assert (await jobs.list())[0].stage == "applied"


def test_applied_removes_that_jobs_whole_row():
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    from fuse.channels.telegram_bot import without_button

    markup = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(f"{v} {i}", callback_data=f"{c}:{i}")
                for v, c in (("Track", "jt"), ("Applied", "ja"))
            ]
            for i in (1, 2)
        ]
    )
    left = without_button(markup, "ja:1", whole_row=True)
    assert [[b.callback_data for b in row] for row in left.inline_keyboard] == [["jt:2", "ja:2"]]
