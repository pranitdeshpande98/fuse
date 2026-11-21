"""Telegram channel: for now it echoes messages, and only for the allowed chat."""

import logging

from telegram import Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters

from fuse.config import Settings, get_settings
from fuse.logging_setup import configure_logging

log = logging.getLogger("fuse.telegram")


def is_allowed(chat_id: int | None, allowed_chat_id: int) -> bool:
    """Only the configured chat gets through; an unset (0) config lets nobody in."""
    return allowed_chat_id != 0 and chat_id == allowed_chat_id


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None:
        return
    log.info("message received (%d chars)", len(update.message.text))
    await update.message.reply_text(f"You said: {update.message.text}")


def build_application(settings: Settings) -> Application:
    app = ApplicationBuilder().token(settings.telegram_bot_token.get_secret_value()).build()
    only_me = filters.Chat(chat_id=settings.telegram_allowed_chat_id)
    app.add_handler(MessageHandler(only_me & filters.TEXT & ~filters.COMMAND, echo))
    return app


def main() -> None:
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
