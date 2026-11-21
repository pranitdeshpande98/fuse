"""One-off helper: prints the chat ID of whoever messages your bot.

Run it, send your bot any message from your own Telegram account, copy the printed ID
into TELEGRAM_ALLOWED_CHAT_ID in .env, then stop it with Ctrl+C.
"""

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from fuse.config import get_settings
from fuse.logging_setup import configure_logging


async def print_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is None:
        return
    print(f"chat_id={chat.id}")
    await chat.send_message(f"Your chat ID is {chat.id}")


def main() -> None:
    configure_logging("WARNING")
    token = get_settings().telegram_bot_token.get_secret_value()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env first")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.ALL, print_chat_id))
    print("Waiting... send your bot a message from Telegram (Ctrl+C to stop)")
    app.run_polling()


if __name__ == "__main__":
    main()
