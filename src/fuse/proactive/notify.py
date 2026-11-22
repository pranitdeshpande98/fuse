"""How proactive jobs reach you. Jobs talk to a Notifier, not to Telegram directly, so they
can be tested with a recording fake (and a WhatsApp notifier can be added in Phase 6)."""

from dataclasses import dataclass, field
from typing import Protocol

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

TELEGRAM_LIMIT = 4000


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split long text into Telegram-sized parts, at paragraph or line breaks if possible."""
    parts, rest = [], text.strip()
    while len(rest) > limit:
        cut = rest.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts or [""]


@dataclass(frozen=True)
class Button:
    label: str
    data: str  # what the button sends back when tapped (max 64 bytes on Telegram)


Buttons = list[Button] | list[list[Button]]  # one row, or several rows


def button_rows(buttons: Buttons | None) -> list[list[Button]]:
    if not buttons:
        return []
    if isinstance(buttons[0], Button):
        return [list(buttons)]  # type: ignore[arg-type]
    return [list(row) for row in buttons]  # type: ignore[union-attr]


def telegram_markup(buttons: Buttons | None) -> InlineKeyboardMarkup | None:
    rows = button_rows(buttons)
    if not rows:
        return None
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(b.label, callback_data=b.data) for b in row] for row in rows]
    )


class Notifier(Protocol):
    async def send(self, text: str, buttons: Buttons | None = None) -> None: ...


class TelegramNotifier:
    """Sends to your one allowed chat."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id

    async def send(self, text: str, buttons: Buttons | None = None) -> None:
        await self._bot.send_message(
            self._chat_id, text[:TELEGRAM_LIMIT], reply_markup=telegram_markup(buttons)
        )


@dataclass
class RecordingNotifier:
    """For tests: keeps every message instead of sending it."""

    sent: list[tuple[str, list[Button]]] = field(default_factory=list)

    async def send(self, text: str, buttons: Buttons | None = None) -> None:
        # Rows are flattened here: tests care which buttons there are, in order.
        self.sent.append((text, [b for row in button_rows(buttons) for b in row]))

    @property
    def texts(self) -> list[str]:
        return [text for text, _ in self.sent]
