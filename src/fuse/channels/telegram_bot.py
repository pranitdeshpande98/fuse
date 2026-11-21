"""Telegram channel: passes your messages to the agent and sends back its reply."""

import logging
from contextlib import AsyncExitStack

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters

from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.compat import use_selector_event_loop_on_windows
from fuse.config import Settings, get_settings
from fuse.db import open_pool
from fuse.llm import build_llm
from fuse.logging_setup import configure_logging
from fuse.tools.basic import make_basic_tools

log = logging.getLogger("fuse.telegram")

TELEGRAM_LIMIT = 4000  # Telegram's hard cap is 4096 characters per message


def is_allowed(chat_id: int | None, allowed_chat_id: int) -> bool:
    """Only the configured chat gets through; an unset (0) config lets nobody in."""
    return allowed_chat_id != 0 and chat_id == allowed_chat_id


def build_runner(settings: Settings, checkpointer: BaseCheckpointSaver) -> AgentRunner:
    tools = make_basic_tools(settings.user_timezone)
    model = build_llm(settings, tools)
    graph = build_graph(model, tools, checkpointer)
    return AgentRunner(graph, max_steps=settings.agent_max_steps)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None or update.effective_chat is None:
        return
    runner: AgentRunner = context.application.bot_data["runner"]
    thread_id = f"telegram:{update.effective_chat.id}"
    log.info("message received (%d chars)", len(update.message.text))
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        reply = await runner.send(thread_id, update.message.text)
    except Exception:
        log.exception("agent failed")
        reply = "Something went wrong on my side. Check the terminal logs."
    await update.message.reply_text(reply[:TELEGRAM_LIMIT])


def build_application(settings: Settings) -> Application:
    # Opened when the bot starts and closed when it stops: the database pool lives this long.
    resources = AsyncExitStack()

    async def on_startup(app: Application) -> None:
        pool = await resources.enter_async_context(
            open_pool(settings.database_url.get_secret_value())
        )
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()  # creates LangGraph's tables on first run; no-op afterwards
        app.bot_data["runner"] = build_runner(settings, checkpointer)
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
    return app


def main() -> None:
    use_selector_event_loop_on_windows()
    settings = get_settings()
    configure_logging(settings.log_level)
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
