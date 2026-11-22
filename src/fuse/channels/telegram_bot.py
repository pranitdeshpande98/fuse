"""Telegram channel: passes your messages to the agent and sends back its reply."""

import contextlib
import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
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
from fuse.agent.runner import AgentRunner
from fuse.channels.core import COMMAND_NAMES, ChatCore, Outgoing
from fuse.compat import use_selector_event_loop_on_windows
from fuse.config import Settings, get_settings
from fuse.google.reader import GoogleApiReader
from fuse.jobalerts.service import MatchWatcher, ScanRunner
from fuse.jobalerts.tools import make_scan_tool
from fuse.llm import build_llm, build_structured_llm
from fuse.logging_setup import configure_logging
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
from fuse.proactive.notify import TelegramNotifier, telegram_markup
from fuse.proactive.schedule import Scheduler, SystemClock, parse_hhmm
from fuse.proactive.store import PostgresCursorStore, PostgresMuteStore
from fuse.services import open_services
from fuse.skills.loader import describe_schedule, parse_schedule
from fuse.skills.tools import start_skill
from fuse.tasks.tools import make_task_tools
from fuse.tracing import enable_tracing

log = logging.getLogger("fuse.telegram")

TICK_SECONDS = 30  # how often the scheduler checks for due proactive jobs

TELEGRAM_LIMIT = 4000  # Telegram's hard cap is 4096 characters per message


def is_allowed(chat_id: int | None, allowed_chat_id: int) -> bool:
    """Only the configured chat gets through; an unset (0) config lets nobody in."""
    return allowed_chat_id != 0 and chat_id == allowed_chat_id


def thread_id_for(chat_id: int) -> str:
    return f"telegram:{chat_id}"


async def deliver(update: Update, messages: list[Outgoing]) -> None:
    chat = update.effective_chat
    assert chat is not None
    for message in messages:
        await chat.send_message(
            message.text[:TELEGRAM_LIMIT], reply_markup=telegram_markup(message.buttons)
        )


def core_of(context: ContextTypes.DEFAULT_TYPE) -> ChatCore:
    return context.application.bot_data["core"]


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None or update.effective_chat is None:
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    thread_id = thread_id_for(update.effective_chat.id)
    await deliver(update, await core_of(context).message(thread_id, update.message.text))


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Every button: approvals, alert buttons, task approvals. The core decides what it means."""
    query = update.callback_query
    if query is None or update.effective_chat is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    if not is_allowed(update.effective_chat.id, settings.telegram_allowed_chat_id):
        await query.answer()  # buttons pressed by anyone else are ignored
        return
    thread_id = thread_id_for(update.effective_chat.id)
    result = await core_of(context).button(thread_id, query.data or "")
    await query.answer(result.toast[:200] or None)
    # Buttons work once: they're removed, or replaced (e.g. Send appears after Always allow).
    # BadRequest here means "message is not modified": the buttons were already gone.
    with contextlib.suppress(BadRequest):
        await query.edit_message_reply_markup(reply_markup=telegram_markup(result.buttons))
    await deliver(update, result.messages)


async def on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None or update.effective_chat is None:
        return
    name = update.message.text.split()[0].lstrip("/").split("@")[0].lower()
    thread_id = thread_id_for(update.effective_chat.id)
    await update.effective_chat.send_action(ChatAction.TYPING)
    args = " ".join(context.args or [])
    await deliver(update, await core_of(context).command(thread_id, name, args))


async def on_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs every TICK_SECONDS: lets the scheduler start whatever proactive job is due."""
    scheduler: Scheduler = context.application.bot_data["scheduler"]
    await scheduler.tick()


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

    services = app.bot_data.get("services")
    scan_runner: ScanRunner | None = app.bot_data.get("scan_runner")
    job_alerts = services.job_alerts if services else None
    if job_alerts is not None:
        watcher = MatchWatcher(
            job_alerts, cursors, events, notifier, clock, on_sent=add_to_conversation
        )
        scheduler.add_interval(
            "job_alerts_matches", timedelta(minutes=settings.job_alerts_poll_minutes), watcher.check
        )
        if scan_runner is not None:
            scan_runner.after = watcher.check  # a scan's new matches arrive right after it
    if services is not None:
        for skill in services.skills:
            if not skill.schedule:
                continue
            days, at = parse_schedule(skill.schedule)

            async def run_skill(name: str = skill.name) -> None:
                text = await start_skill(
                    services.tasks, services.skills, name, settings.task_max_attempts
                )
                log.info("scheduled skill %s: %s", name, text)

            scheduler.add_daily(f"skill:{skill.name}", at, run_skill, weekdays=days)
            log.info("Skill %s scheduled: %s", skill.name, describe_schedule(skill.schedule))
    app.bot_data.update(
        scheduler=scheduler,
        core=ChatCore(
            runner,
            memory=memory,
            tasks=services.tasks if services else None,
            contacts=services.contacts if services else None,
            mutes=mutes,
            alert_actions=AlertActions(events, mutes, runner),
            brief=brief.run,
            task_max_attempts=settings.task_max_attempts,
            skills=services.skills if services else None,
            jobs=services.jobs if services else None,
            job_alerts=job_alerts,
            scan_runner=scan_runner,
            events=events,
        ),
    )
    if settings.proactive_enabled and app.job_queue is not None:
        app.job_queue.run_repeating(on_tick, interval=TICK_SECONDS, first=15, name="tick")
        log.info("Proactive jobs on (timezone %s)", settings.user_timezone)


def build_application(settings: Settings) -> Application:
    # Opened when the bot starts and closed when it stops: the database pool lives this long.
    resources = AsyncExitStack()

    async def on_startup(app: Application) -> None:
        services = await resources.enter_async_context(open_services(settings))
        # Built before the agent, so the agent can get a "scan now" tool. The scan's
        # report goes to Telegram; its new matches follow (wired in setup_proactive).
        scan_runner = None
        extra_tools = [make_task_tools(services.tasks, settings.task_max_attempts)]
        if services.job_alerts is not None:
            notifier = TelegramNotifier(app.bot, settings.telegram_allowed_chat_id)
            scan_runner = ScanRunner(services.job_alerts, notifier)
            extra_tools.append([make_scan_tool(scan_runner.start)])
        app.bot_data["scan_runner"] = scan_runner
        pool, memory, google = services.pool, services.memory, services.google
        checkpointer = services.checkpointer
        app.bot_data["runner"] = build_runner(
            settings,
            checkpointer,
            services.actions,
            google,
            services.drafts,
            memory,
            extra_tools=[tool for group in extra_tools for tool in group],
            contacts=services.contacts,
            skills=services.skills,
            jobs=services.jobs,
            job_alerts=services.job_alerts,
        )
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
    app.add_handler(CommandHandler(list(COMMAND_NAMES), on_command, filters=only_me))
    app.add_handler(CallbackQueryHandler(on_button))
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
