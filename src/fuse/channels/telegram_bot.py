"""Telegram channel: passes your messages to the agent and sends back its reply."""

import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from fuse.agent.factory import build_runner
from fuse.agent.runner import AgentReply, AgentRunner
from fuse.approval_ui import after_allowing, approval_buttons, approval_text
from fuse.compat import use_selector_event_loop_on_windows
from fuse.config import Settings, get_settings
from fuse.contacts import ContactStore, contacts_text
from fuse.google.reader import GoogleApiReader
from fuse.llm import build_llm, build_structured_llm
from fuse.logging_setup import configure_logging
from fuse.memory.commands import forget_text, list_memories_text
from fuse.memory.consolidate import ExtractedFacts, consolidate
from fuse.memory.store import MemoryStore
from fuse.proactive.alerts import AlertActions, AlertSender, EventHeldItems
from fuse.proactive.brief import MorningBrief
from fuse.proactive.decide import DecisionEngine, TriageResult
from fuse.proactive.events import PostgresEventStore
from fuse.proactive.gmail_watch import GmailWatcher
from fuse.proactive.jobs import PostgresJobLog
from fuse.proactive.meetings import MeetingReminders
from fuse.proactive.nightly import NightlyConsolidation
from fuse.proactive.notify import TelegramNotifier, split_message, telegram_markup
from fuse.proactive.schedule import Scheduler, SystemClock, parse_hhmm
from fuse.proactive.store import MuteStore, PostgresCursorStore, PostgresMuteStore
from fuse.services import open_services
from fuse.tasks.queue import TaskStore
from fuse.tasks.report import (
    allow_task_recipients,
    cancel_text,
    decide_task_approval,
    tasks_text,
)
from fuse.tasks.tools import make_task_tools, start_task
from fuse.tracing import enable_tracing

log = logging.getLogger("fuse.telegram")

TICK_SECONDS = 30  # how often the scheduler checks for due proactive jobs

TELEGRAM_LIMIT = 4000  # Telegram's hard cap is 4096 characters per message


def is_allowed(chat_id: int | None, allowed_chat_id: int) -> bool:
    """Only the configured chat gets through; an unset (0) config lets nobody in."""
    return allowed_chat_id != 0 and chat_id == allowed_chat_id


def thread_id_for(chat_id: int) -> str:
    return f"telegram:{chat_id}"


async def send_reply(update: Update, reply: AgentReply) -> None:
    """Plain text, or a preview with its approval buttons (Send / Save as draft / Reject for
    an email, Approve / Reject for anything else)."""
    chat = update.effective_chat
    assert chat is not None
    if reply.kind == "approval":
        payload = reply.approval
        key = payload["action_key"]
        buttons = approval_buttons(payload, lambda verb: f"{verb}:{key}")
        text = f"{reply.text}\n\n{approval_text(payload)}"
        await chat.send_message(text[:TELEGRAM_LIMIT], reply_markup=telegram_markup(buttons))
    else:
        for part in split_message(reply.text):
            await chat.send_message(part)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None or update.effective_chat is None:
        return
    runner: AgentRunner = context.application.bot_data["runner"]
    log.info("message received (%d chars)", len(update.message.text))
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await runner.send(thread_id_for(update.effective_chat.id), update.message.text)
    except Exception:
        log.exception("agent failed")
        reply = AgentReply("text", "Something went wrong on my side. Check the terminal logs.")
    await send_reply(update, reply)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Called when you tap Send, Save as draft / Approve, Reject or Always allow."""
    query = update.callback_query
    if query is None or update.effective_chat is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    if not is_allowed(update.effective_chat.id, settings.telegram_allowed_chat_id):
        await query.answer()  # buttons pressed by anyone else are ignored
        return

    verb, _, key = (query.data or "").partition(":")
    thread_id = thread_id_for(update.effective_chat.id)
    runner: AgentRunner = context.application.bot_data["runner"]

    # Only act if this exact action is still the one waiting (old buttons do nothing).
    pending = await runner.pending_approval(thread_id)
    if not pending or pending.get("action_key") != key:
        await query.answer("This is no longer waiting for approval.")
        await query.edit_message_reply_markup(reply_markup=None)
        return

    if verb == "allow":
        contacts: ContactStore = context.application.bot_data["contacts"]
        for address in pending.get("blocked") or []:
            await contacts.add(address, source="always allow")
        await query.answer("Added to your contacts")
        buttons = approval_buttons(after_allowing(pending), lambda v: f"{v}:{key}")
        await query.edit_message_reply_markup(reply_markup=telegram_markup(buttons))
        return

    approved = verb in ("approve", "send")
    labels = {"send": "Sending", "approve": "Approved", "reject": "Rejected"}
    await query.answer(labels.get(verb, "Done"))
    await query.edit_message_reply_markup(reply_markup=None)  # buttons work only once
    log.info("action %s in %s", verb, thread_id)
    try:
        reply = await runner.resume(
            thread_id, approved=approved, mode="send" if verb == "send" else None
        )
    except Exception:
        log.exception("agent failed")
        reply = AgentReply("text", "Something went wrong on my side. Check the terminal logs.")
    await send_reply(update, reply)


async def on_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs every TICK_SECONDS: lets the scheduler start whatever proactive job is due."""
    scheduler: Scheduler = context.application.bot_data["scheduler"]
    await scheduler.tick()


async def on_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/brief: today's brief right now, without waiting for the morning."""
    brief: MorningBrief = context.application.bot_data["brief"]
    await update.effective_chat.send_action(ChatAction.TYPING)  # type: ignore[union-attr]
    await brief.run()


async def on_alert_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Draft reply / Dismiss / Mute sender, tapped on a proactive alert."""
    query = update.callback_query
    if query is None or update.effective_chat is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    if not is_allowed(update.effective_chat.id, settings.telegram_allowed_chat_id):
        await query.answer()
        return
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)  # each alert is handled once
    actions: AlertActions = context.application.bot_data["alert_actions"]
    try:
        reply = await actions.handle(query.data or "", thread_id_for(update.effective_chat.id))
    except Exception:
        log.exception("alert button failed")
        reply = AgentReply("text", "Something went wrong on my side. Check the terminal logs.")
    await send_reply(update, reply)


async def on_task_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Buttons on something a background task wants to do. The bot only records your
    decision in Postgres; the worker, a separate process, resumes the task."""
    query = update.callback_query
    if query is None or update.effective_chat is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    if not is_allowed(update.effective_chat.id, settings.telegram_allowed_chat_id):
        await query.answer()
        return
    tasks = context.application.bot_data["tasks"]
    if (query.data or "").startswith("tk:allow:"):
        text, buttons = await allow_task_recipients(
            tasks, context.application.bot_data["contacts"], query.data or ""
        )
        await query.answer(text[:200])
        await query.edit_message_reply_markup(reply_markup=telegram_markup(buttons))
        return
    text = await decide_task_approval(tasks, query.data or "")
    await query.answer(text[:200])
    await query.edit_message_reply_markup(reply_markup=None)
    await update.effective_chat.send_message(text)


async def on_unmute(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/unmute <address>: get alerts from a sender again."""
    mutes: MuteStore = context.application.bot_data["mutes"]
    address = " ".join(context.args or []).strip()
    if not address:
        muted = sorted(await mutes.all())
        text = "Muted: " + ", ".join(muted) if muted else "You haven't muted anyone."
        text += "\nUsage: /unmute name@example.com" if muted else ""
    elif await mutes.unmute(address):
        text = f"Unmuted {address}."
    else:
        text = f"{address} wasn't muted."
    await update.effective_chat.send_message(text)  # type: ignore[union-attr]


async def on_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/task <goal>: start a background task directly."""
    tasks: TaskStore = context.application.bot_data["tasks"]
    settings: Settings = context.application.bot_data["settings"]
    text = await start_task(tasks, " ".join(context.args or []), settings.task_max_attempts)
    await update.effective_chat.send_message(text)  # type: ignore[union-attr]


async def on_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/tasks: recent background tasks and where each one is."""
    text = await tasks_text(context.application.bot_data["tasks"])
    await update.effective_chat.send_message(text)  # type: ignore[union-attr]


async def on_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/cancel <id>: stop a background task."""
    text = await cancel_text(context.application.bot_data["tasks"], " ".join(context.args or []))
    await update.effective_chat.send_message(text)  # type: ignore[union-attr]


async def on_contacts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/contacts [add|remove <address>]: who Fuse may send email to."""
    contacts: ContactStore = context.application.bot_data["contacts"]
    text = await contacts_text(contacts, " ".join(context.args or []))
    await update.effective_chat.send_message(text)  # type: ignore[union-attr]


async def on_memories(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/memories: show everything Fuse remembers about you."""
    memory: MemoryStore = context.application.bot_data["memory"]
    text = await list_memories_text(memory)
    await update.effective_chat.send_message(text[:TELEGRAM_LIMIT])  # type: ignore[union-attr]


async def on_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/forget <id>: delete one memory."""
    memory: MemoryStore = context.application.bot_data["memory"]
    text = await forget_text(memory, " ".join(context.args or []))
    await update.effective_chat.send_message(text)  # type: ignore[union-attr]


async def setup_proactive(
    app: Application,
    settings: Settings,
    pool: Any,
    google: GoogleApiReader | None,
    memory: MemoryStore,
    checkpointer: AsyncPostgresSaver,
) -> None:
    """Everything Fuse does on its own: the brief, the Gmail watcher, triage and alerts.

    All of it runs from one scheduler tick, so there is a single place to turn it off
    (PROACTIVE_ENABLED=false) and a single loop to reason about.
    """
    runner: AgentRunner = app.bot_data["runner"]
    chat_id = settings.telegram_allowed_chat_id
    clock = SystemClock(settings.user_timezone)
    notifier = TelegramNotifier(app.bot, chat_id)

    async def add_to_conversation(text: str) -> None:
        await runner.add_note(thread_id_for(chat_id), text)

    job_log = PostgresJobLog(pool)
    events = PostgresEventStore(pool)
    mutes = PostgresMuteStore(pool)
    cursors = PostgresCursorStore(pool)
    for store in (job_log, events, mutes, cursors):
        await store.ensure_schema()

    brief = MorningBrief(
        clock,
        notifier,
        build_llm(settings, tier="light"),
        google,
        memory,
        held=EventHeldItems(events),
        on_sent=add_to_conversation,
    )
    scheduler = Scheduler(clock, job_log)
    scheduler.add_daily("morning_brief", parse_hhmm(settings.brief_time), brief.run)

    extractor = build_structured_llm(settings, ExtractedFacts, tier="light")
    nightly = NightlyConsolidation(
        lambda: consolidate(pool, checkpointer, extractor, memory), events, clock
    )
    scheduler.add_daily(
        "memory_consolidation", parse_hhmm(settings.consolidation_time), nightly.run
    )

    if google is not None:

        async def on_new_email(summary: dict[str, Any]) -> None:
            await events.add("email", f"gmail:{summary['id']}", summary)

        watcher = GmailWatcher(google, cursors, on_new_email, notifier, clock, job_log)
        engine = DecisionEngine(
            events,
            mutes,
            build_structured_llm(settings, TriageResult, tier="light"),
            clock,
            AlertSender(notifier, on_sent=add_to_conversation),
            quiet_start=parse_hhmm(settings.quiet_hours_start),
            quiet_end=parse_hhmm(settings.quiet_hours_end),
            max_alerts_per_hour=settings.max_alerts_per_hour,
            memory=memory,
        )
        scheduler.add_interval(
            "gmail_watch", timedelta(seconds=settings.gmail_poll_seconds), watcher.poll
        )
        scheduler.add_interval("decide", timedelta(seconds=TICK_SECONDS), engine.process)

        reminders = MeetingReminders(
            google,
            events,
            notifier,
            clock,
            settings.meeting_reminder_minutes,
            memory,
            on_sent=add_to_conversation,
        )
        # Checking at least once per window means no meeting can slip between two checks.
        window = timedelta(minutes=settings.meeting_reminder_minutes)
        scheduler.add_interval(
            "meeting_reminders", min(timedelta(minutes=5), window), reminders.check
        )

    app.bot_data.update(
        brief=brief,
        scheduler=scheduler,
        mutes=mutes,
        alert_actions=AlertActions(events, mutes, runner),
    )
    if settings.proactive_enabled and app.job_queue is not None:
        app.job_queue.run_repeating(on_tick, interval=TICK_SECONDS, first=15, name="tick")
        log.info("Proactive jobs on (timezone %s)", settings.user_timezone)


def build_application(settings: Settings) -> Application:
    # Opened when the bot starts and closed when it stops: the database pool lives this long.
    resources = AsyncExitStack()

    async def on_startup(app: Application) -> None:
        services = await resources.enter_async_context(open_services(settings))
        pool, memory, google = services.pool, services.memory, services.google
        checkpointer = services.checkpointer
        app.bot_data["runner"] = build_runner(
            settings,
            checkpointer,
            services.actions,
            google,
            services.drafts,
            memory,
            extra_tools=make_task_tools(services.tasks, settings.task_max_attempts),
            contacts=services.contacts,
        )
        app.bot_data["contacts"] = services.contacts
        app.bot_data["tasks"] = services.tasks
        app.bot_data["settings"] = settings
        app.bot_data["memory"] = memory
        app.bot_data["services"] = services

        await setup_proactive(app, settings, pool, google, memory, checkpointer)
        log.info("Agent ready; conversations are stored in Postgres")

    async def on_shutdown(app: Application) -> None:
        await resources.aclose()

    app = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token.get_secret_value())
        .post_init(on_startup)
        .post_shutdown(on_shutdown)
        .build()
    )
    only_me = filters.Chat(chat_id=settings.telegram_allowed_chat_id)
    app.add_handler(MessageHandler(only_me & filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(CommandHandler("brief", on_brief, filters=only_me))
    app.add_handler(CommandHandler("memories", on_memories, filters=only_me))
    app.add_handler(CommandHandler("forget", on_forget, filters=only_me))
    app.add_handler(CommandHandler("task", on_task, filters=only_me))
    app.add_handler(CommandHandler("tasks", on_tasks, filters=only_me))
    app.add_handler(CommandHandler("cancel", on_cancel, filters=only_me))
    app.add_handler(CommandHandler("unmute", on_unmute, filters=only_me))
    app.add_handler(CommandHandler("contacts", on_contacts, filters=only_me))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^(approve|reject|send|allow):"))
    app.add_handler(CallbackQueryHandler(on_alert_button, pattern=r"^ev:"))
    app.add_handler(CallbackQueryHandler(on_task_button, pattern=r"^tk:"))
    return app


def main() -> None:
    use_selector_event_loop_on_windows()
    settings = get_settings()
    configure_logging(settings.log_level)
    enable_tracing(settings)
    if not settings.telegram_bot_token.get_secret_value():
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")
    if settings.telegram_allowed_chat_id == 0:
        raise SystemExit(
            "Set TELEGRAM_ALLOWED_CHAT_ID in .env (run: uv run python -m fuse.channels.whoami)"
        )
    log.info("Fuse bot starting; only chat %s is allowed", settings.telegram_allowed_chat_id)
    build_application(settings).run_polling()


if __name__ == "__main__":
    main()
