"""How proactive jobs reach you. Jobs talk to a Notifier, not to Telegram directly, so they
can be tested with a recording fake (and a WhatsApp notifier can be added in Phase 6)."""

from dataclasses import dataclass, field
from typing import Protocol

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

TELEGRAM_LIMIT = 4000


@dataclass(frozen=True)
class Button:
    label: str
    data: str  # what the button sends back when tapped (max 64 bytes on Telegram)


class Notifier(Protocol):
    async def send(self, text: str, buttons: list[Button] | None = None) -> None: ...


class TelegramNotifier:
    """Sends to your one allowed chat."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id

    async def send(self, text: str, buttons: list[Button] | None = None) -> None:
        markup = (
            InlineKeyboardMarkup(
                [[InlineKeyboardButton(b.label, callback_data=b.data) for b in buttons]]
            )
            if buttons
            else None
        )
        await self._bot.send_message(self._chat_id, text[:TELEGRAM_LIMIT], reply_markup=markup)


@dataclass
class RecordingNotifier:
    """For tests: keeps every message instead of sending it."""

    sent: list[tuple[str, list[Button]]] = field(default_factory=list)

    async def send(self, text: str, buttons: list[Button] | None = None) -> None:
        self.sent.append((text, list(buttons or [])))

    @property
    def texts(self) -> list[str]:
        return [text for text, _ in self.sent]
