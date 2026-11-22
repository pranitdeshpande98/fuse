"""Telegram channel: passes your messages to the agent and sends back its reply."""

import logging
from contextlib import AsyncExitStack

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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

from fuse.actions import ActionStore, PostgresActionStore
from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentReply, AgentRunner
from fuse.compat import use_selector_event_loop_on_windows
from fuse.config import Settings, get_settings
from fuse.crypto import SecretBox
from fuse.db import open_pool
from fuse.google.auth import TokenStore
from fuse.google.drafts import GmailDraftSink
from fuse.google.reader import GoogleApiReader, GoogleReader
from fuse.google.tools import make_google_tools
from fuse.llm import build_llm
from fuse.logging_setup import configure_logging
from fuse.memory.commands import forget_text, list_memories_text
from fuse.memory.embeddings import FastEmbedder
from fuse.memory.store import MemoryStore, PostgresMemoryStore
from fuse.memory.tools import make_memory_tools
from fuse.proactive.jobs import PostgresJobLog
from fuse.proactive.schedule import Scheduler, SystemClock
from fuse.tools.basic import make_basic_tools
from fuse.tools.drafts import DraftSink, LocalDraftSink, make_draft_tools
from fuse.tracing import enable_tracing

log = logging.getLogger("fuse.telegram")

TICK_SECONDS = 30  # how often the scheduler checks for due proactive jobs

TELEGRAM_LIMIT = 4000  # Telegram's hard cap is 4096 characters per message


def is_allowed(chat_id: int | None, allowed_chat_id: int) -> bool:
    """Only the configured chat gets through; an unset (0) config lets nobody in."""
    return allowed_chat_id != 0 and chat_id == allowed_chat_id


def build_runner(
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    actions: ActionStore,
    google: GoogleReader | None = None,
    drafts: DraftSink | None = None,
    memory: MemoryStore | None = None,
) -> AgentRunner:
    tools = [
        *make_basic_tools(settings.user_timezone),
        *(make_google_tools(google) if google else []),
        *make_draft_tools(actions, drafts or LocalDraftSink()),
        *(make_memory_tools(memory, actions) if memory else []),
    ]
    model = build_llm(settings, tools)
    graph = build_graph(
        model, tools, checkpointer, memory, max_history_messages=settings.agent_history_messages
    )
    return AgentRunner(graph, max_steps=settings.agent_max_steps)


def thread_id_for(chat_id: int) -> str:
    return f"telegram:{chat_id}"


async def send_reply(update: Update, reply: AgentReply) -> None:
    """Plain text, or a preview with Approve / Reject buttons."""
    chat = update.effective_chat
    assert chat is not None
    if reply.kind == "approval":
        key = reply.approval["action_key"]
        buttons = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data=f"approve:{key}"),
                    InlineKeyboardButton("Reject", callback_data=f"reject:{key}"),
                ]
            ]
        )
        text = f"{reply.text}\n\n{reply.approval['preview']}"
        await chat.send_message(text[:TELEGRAM_LIMIT], reply_markup=buttons)
    else:
        await chat.send_message(reply.text[:TELEGRAM_LIMIT])


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
    """Called when you tap Approve or Reject."""
    query = update.callback_query
    if query is None or update.effective_chat is None:
        return
    settings: Settings = context.application.bot_data["settings"]
    if not is_allowed(update.effective_chat.id, settings.telegram_allowed_chat_id):
        await query.answer()  # buttons pressed by anyone else are ignored
        return

    decision, _, key = (query.data or "").partition(":")
    thread_id = thread_id_for(update.effective_chat.id)
    runner: AgentRunner = context.application.bot_data["runner"]

    # Only act if this exact action is still the one waiting (old buttons do nothing).
    pending = await runner.pending_approval(thread_id)
    if not pending or pending.get("action_key") != key:
        await query.answer("This is no longer waiting for approval.")
        await query.edit_message_reply_markup(reply_markup=None)
        return

    approved = decision == "approve"
    await query.answer("Approved" if approved else "Rejected")
    await query.edit_message_reply_markup(reply_markup=None)  # buttons work only once
    log.info("action %s in %s", "approved" if approved else "rejected", thread_id)
    try:
        reply = await runner.resume(thread_id, approved=approved)
    except Exception:
        log.exception("agent failed")
        reply = AgentReply("text", "Something went wrong on my side. Check the terminal logs.")
    await send_reply(update, reply)


async def on_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs every TICK_SECONDS: lets the scheduler start whatever proactive job is due."""
    scheduler: Scheduler = context.application.bot_data["scheduler"]
    await scheduler.tick()


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


def build_application(settings: Settings) -> Application:
    # Opened when the bot starts and closed when it stops: the database pool lives this long.
    resources = AsyncExitStack()

    async def on_startup(app: Application) -> None:
        pool = await resources.enter_async_context(
            open_pool(settings.database_url.get_secret_value())
        )
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()  # creates LangGraph's tables on first run; no-op afterwards
        actions = PostgresActionStore(pool)
        await actions.ensure_schema()  # creates the actions table on first run

        google, drafts = None, None
        if settings.encryption_key.get_secret_value():
            tokens = TokenStore(pool, SecretBox(settings.encryption_key.get_secret_value()))
            await tokens.ensure_schema()
            google = GoogleApiReader(tokens)
            drafts = GmailDraftSink(tokens)
        else:
            log.warning("ENCRYPTION_KEY not set: Gmail and Calendar tools are disabled")

        log.info("Loading the embedding model (the first run downloads ~130 MB)")
        embedder = FastEmbedder(settings.embedding_model, settings.fastembed_cache_dir)
        await embedder.embed(["warm up"])  # load it now, not during your first message
        memory = PostgresMemoryStore(pool, embedder)
        await memory.ensure_schema()

        app.bot_data["runner"] = build_runner(
            settings, checkpointer, actions, google, drafts, memory
        )
        app.bot_data["settings"] = settings
        app.bot_data["memory"] = memory

        job_log = PostgresJobLog(pool)
        await job_log.ensure_schema()
        scheduler = Scheduler(SystemClock(settings.user_timezone), job_log)
        app.bot_data["scheduler"] = scheduler
        if settings.proactive_enabled and app.job_queue is not None:
            app.job_queue.run_repeating(on_tick, interval=TICK_SECONDS, first=15, name="tick")
            log.info("Proactive jobs on (timezone %s)", settings.user_timezone)
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
    app.add_handler(CommandHandler("memories", on_memories, filters=only_me))
    app.add_handler(CommandHandler("forget", on_forget, filters=only_me))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^(approve|reject):"))
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
