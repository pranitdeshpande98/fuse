import pytest
from telegram.error import BadRequest

from fuse.channels.telegram_format import to_plain, to_telegram_html
from fuse.proactive.notify import send_formatted


def test_bold_italic_code_links_headings_and_bullets():
    text = (
        "## Plan\n"
        "Reminder set for **11:09 CST**: *Stretch*.\n"
        "- run `uv sync`\n"
        "- see [the docs](https://example.com/a?b=1&c=2)"
    )
    out = to_telegram_html(text)
    assert out.startswith("<b>Plan</b>")
    assert "<b>11:09 CST</b>" in out and "<i>Stretch</i>" in out
    assert "• run <code>uv sync</code>" in out
    assert '<a href="https://example.com/a?b=1&amp;c=2">the docs</a>' in out


def test_email_text_cant_break_the_message():
    out = to_telegram_html("From Pat <pat@acme.com>: R&D update for a_b_c")
    assert out == "From Pat &lt;pat@acme.com&gt;: R&amp;D update for a_b_c"


def test_tables_become_one_line_per_row():
    table = (
        "**Your emails**\n\n"
        "| # | From | Subject |\n|---|------|---------|\n"
        "| 1 | Zillow | **New listing** |\n| 2 | Sam | Q3 numbers |\n\nDone."
    )
    out = to_telegram_html(table)
    assert "|" not in out
    assert "<b># · From · Subject</b>" in out
    assert "• 1 · Zillow · <b>New listing</b>" in out
    assert "• 2 · Sam · Q3 numbers" in out and out.endswith("Done.")


def test_code_blocks_are_kept_as_is():
    out = to_telegram_html("Run:\n```bash\nuv run pytest -q **not bold**\n```")
    assert "<pre>uv run pytest -q **not bold**</pre>" in out


def test_plain_fallback_has_no_markdown_marks():
    assert to_plain("**Hi** `x`\n| a | b |\n|---|---|\n| 1 | 2 |") == "Hi x\na · b\n- 1 · 2"


class FakeBot:
    def __init__(self, reject_html=False):
        self.sent, self.reject_html = [], reject_html

    async def send_message(self, chat_id, text, parse_mode=None, reply_markup=None):
        if parse_mode == "HTML" and self.reject_html:
            raise BadRequest("Can't parse entities: unsupported start tag")
        self.sent.append((text, parse_mode, reply_markup))


async def test_send_formatted_uses_html_and_puts_buttons_on_the_last_part():
    bot = FakeBot()
    await send_formatted(bot, 42, ("**x** " * 400 + "\n\n") * 3, markup="BUTTONS")
    assert len(bot.sent) > 1
    assert all(mode == "HTML" for _, mode, _ in bot.sent)
    assert [m for _, _, m in bot.sent] == [None] * (len(bot.sent) - 1) + ["BUTTONS"]


async def test_rejected_formatting_falls_back_to_plain_text():
    bot = FakeBot(reject_html=True)
    await send_formatted(bot, 42, "Reminder set for **11:09**", markup="B")
    assert bot.sent == [("Reminder set for 11:09", None, "B")]


async def test_other_telegram_errors_are_not_swallowed():
    class Down(FakeBot):
        async def send_message(self, *a, **k):
            raise BadRequest("Chat not found")

    with pytest.raises(BadRequest):
        await send_formatted(Down(), 42, "hi")
