"""Turn the model's Markdown into Telegram's HTML formatting.

Models write Markdown (**bold**, tables, `code`), and Telegram shows it as raw asterisks and
pipes unless told how to read it. Telegram's HTML mode is the safest target: only <, > and &
need escaping, so email subjects and web text can't break the message.

Tables become one line per row (phones are too narrow for columns). If Telegram still
rejects a message, the sender falls back to plain text, so nothing is ever lost to
formatting.
"""

import html
import re

CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.S)
INLINE_CODE = re.compile(r"`([^`\n]+)`")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$", re.M)
BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__", re.S)
ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
BULLET = re.compile(r"^(\s*)[-*]\s+", re.M)
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
STRIP = re.compile(r"\*\*|__|`|^#{1,6}\s+", re.M)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _tables_to_lists(text: str) -> str:
    """| a | b | rows -> one line per row; the header row in bold."""
    out: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.lstrip().startswith("|")
            and i + 1 < len(lines)
            and TABLE_SEPARATOR.match(lines[i + 1].strip())
        ):
            header = _cells(line)
            out.append("**" + " · ".join(h for h in header if h) + "**")
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c for c in _cells(lines[i]) if c]
                if cells:
                    out.append("- " + " · ".join(cells))
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def to_telegram_html(text: str) -> str:
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    text = CODE_BLOCK.sub(
        lambda m: keep(f"<pre>{html.escape(m.group(1).strip(chr(10)))}</pre>"), text
    )
    text = INLINE_CODE.sub(lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = _tables_to_lists(text)
    text = html.escape(text, quote=False)
    text = LINK.sub(
        # The text is already escaped here; only a quote could still break the attribute.
        lambda m: keep(f'<a href="{m.group(2).replace(chr(34), "&quot;")}">{m.group(1)}</a>'),
        text,
    )
    text = HEADING.sub(r"<b>\1</b>", text)
    text = BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = ITALIC.sub(r"<i>\1</i>", text)
    text = BULLET.sub(r"\1• ", text)
    return re.sub("\x00(\\d+)\x00", lambda m: stash[int(m.group(1))], text)


def to_plain(text: str) -> str:
    """The fallback: the same text without Markdown marks, tables as lists."""
    return STRIP.sub("", _tables_to_lists(text))
