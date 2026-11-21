from telegram.ext import MessageHandler

from fuse.channels.telegram_bot import build_application, is_allowed
from fuse.config import Settings


def test_only_the_allowed_chat_gets_through():
    assert is_allowed(42, 42)
    assert not is_allowed(43, 42)
    assert not is_allowed(None, 42)


def test_unset_allowed_chat_lets_nobody_in():
    assert not is_allowed(0, 0)
    assert not is_allowed(42, 0)


def test_handler_is_locked_to_the_allowed_chat():
    settings = Settings(_env_file=None, telegram_bot_token="123:fake", telegram_allowed_chat_id=42)
    app = build_application(settings)
    handlers = [h for group in app.handlers.values() for h in group]
    assert len(handlers) == 1
    assert isinstance(handlers[0], MessageHandler)
    assert "42" in repr(handlers[0].filters)
