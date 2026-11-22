"""How proactive jobs reach you. Jobs talk to a Notifier, not to Telegram directly, so they
can be tested with a recording fake (and a WhatsApp notifier can be added in Phase 6)."""

from dataclasses import dataclass, field
from typing import Any, Protocol

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
        await send_formatted(self._bot, self._chat_id, text, telegram_markup(buttons))


FORMATTED_PART = 3500  # leaves room for the HTML tags Telegram formatting adds


async def send_formatted(bot: Bot, chat_id: int, text: str, markup: Any = None) -> None:
    """Send Markdown-ish text with Telegram formatting; long text in parts (buttons on the
    last). If Telegram rejects the formatting, that part goes as plain text instead."""
    from telegram.error import BadRequest

    from fuse.channels.telegram_format import to_plain, to_telegram_html

    parts = split_message(text, FORMATTED_PART)
    for i, part in enumerate(parts):
        part_markup = markup if i == len(parts) - 1 else None
        try:
            await bot.send_message(
                chat_id, to_telegram_html(part), parse_mode="HTML", reply_markup=part_markup
            )
        except BadRequest as exc:
            if "parse" not in str(exc).lower() and "entit" not in str(exc).lower():
                raise
            await bot.send_message(
                chat_id, to_plain(part)[:TELEGRAM_LIMIT], reply_markup=part_markup
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
