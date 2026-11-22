from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

from fuse.channels.telegram_bot import build_application, is_allowed
from fuse.config import Settings


def test_only_the_allowed_chat_gets_through():
    assert is_allowed(42, 42)
    assert not is_allowed(43, 42)
    assert not is_allowed(None, 42)


def test_unset_allowed_chat_lets_nobody_in():
    assert not is_allowed(0, 0)
    assert not is_allowed(42, 0)


def test_handlers_are_registered():
    settings = Settings(_env_file=None, telegram_bot_token="123:fake", telegram_allowed_chat_id=42)
    app = build_application(settings)  # builds without connecting to Telegram or Postgres
    handlers = [h for group in app.handlers.values() for h in group]
    messages = [h for h in handlers if isinstance(h, MessageHandler)]
    buttons = [h for h in handlers if isinstance(h, CallbackQueryHandler)]
    assert len(messages) == 1 and "42" in repr(messages[0].filters)  # locked to your chat
    assert len(buttons) == 3  # chat approvals, alert buttons, task approvals (all check the chat)
    commands = {c for h in handlers if isinstance(h, CommandHandler) for c in h.commands}
    assert commands == {
        "brief",
        "memories",
        "forget",
        "unmute",
        "task",
        "tasks",
        "cancel",
        "contacts",
    }
    assert all("42" in repr(h.filters) for h in handlers if isinstance(h, CommandHandler))


def test_job_queue_is_available():
    """The job-queue extra must be installed, or proactive jobs would silently never run."""
    settings = Settings(_env_file=None, telegram_bot_token="123:fake", telegram_allowed_chat_id=42)
    assert build_application(settings).job_queue is not None
