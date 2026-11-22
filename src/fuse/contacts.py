"""Who Fuse may send email to, and the check that enforces it.

The allowlist is: your approved contacts, plus everyone already in the email thread being
replied to. The check is plain code, run twice: when the draft is shown (to decide whether
a Send button appears) and again right before sending. Nothing the model writes, and
nothing in an email, can change it; only you can, with "Always allow" or /contacts.
"""

from email.utils import getaddresses
from typing import Protocol

from psycopg_pool import AsyncConnectionPool

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS contacts (
        address     TEXT PRIMARY KEY,
        source      TEXT        NOT NULL DEFAULT 'command',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
]


class InvalidRecipient(ValueError):
    pass


def parse_recipients(*fields: str) -> list[str]:
    """'Pat <Pat@Acme.com>, sam@corp.com' -> ['pat@acme.com', 'sam@corp.com'].

    Strict on purpose: anything that isn't a plain email address raises InvalidRecipient
    instead of being skipped, so what's checked against the allowlist is exactly what is
    sent. (The tool sends these normalised addresses, never the raw text.)
    """
    addresses: list[str] = []
    for text in fields:
        if not text or not text.strip():
            continue
        pieces = [p for p in text.split(",") if p.strip()]
        parsed = getaddresses([text])
        if len(parsed) != len(pieces):
            raise InvalidRecipient(f"{text!r} isn't a list of email addresses")
        for name, address in parsed:
            address = address.strip().lower()
            local, _, domain = address.partition("@")
            if (
                not local
                or "." not in domain
                or address.count("@") != 1
                or any(c.isspace() for c in address)
            ):
                raise InvalidRecipient(f"{(address or name or text)!r} isn't a valid email address")
            if address not in addresses:
                addresses.append(address)
    return addresses


def thread_participants(messages: list[dict[str, str]]) -> set[str]:
    found: set[str] = set()
    for message in messages:
        for field in ("from", "to", "cc"):
            try:
                found.update(parse_recipients(message.get(field, "")))
            except InvalidRecipient:
                continue  # an odd header in someone else's email doesn't allow anything
    return found


def blocked_recipients(
    recipients: list[str], contacts: set[str], participants: set[str]
) -> list[str]:
    return [r for r in recipients if r not in contacts and r not in participants]


class ContactStore(Protocol):
    async def add(self, address: str, source: str = "command") -> None: ...
    async def remove(self, address: str) -> bool: ...
    async def all(self) -> set[str]: ...


class PostgresContactStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def add(self, address: str, source: str = "command") -> None:
        (clean,) = parse_recipients(address)
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO contacts (address, source) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (clean, source),
            )

    async def remove(self, address: str) -> bool:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM contacts WHERE address = %s", (address.strip().lower(),)
            )
            return cur.rowcount > 0

    async def all(self) -> set[str]:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT address FROM contacts")
            return {r["address"] for r in await cur.fetchall()}


class InMemoryContactStore:
    def __init__(self, addresses: set[str] | None = None) -> None:
        self.addresses = set(addresses or set())

    async def add(self, address: str, source: str = "command") -> None:
        (clean,) = parse_recipients(address)
        self.addresses.add(clean)

    async def remove(self, address: str) -> bool:
        clean = address.strip().lower()
        if clean in self.addresses:
            self.addresses.discard(clean)
            return True
        return False

    async def all(self) -> set[str]:
        return set(self.addresses)


async def contacts_text(contacts: ContactStore, argument: str) -> str:
    """/contacts, /contacts add x@y.com, /contacts remove x@y.com"""
    words = argument.split()
    if not words:
        listed = sorted(await contacts.all())
        if not listed:
            return (
                "No approved contacts yet. Fuse can send only to people already in the "
                "thread, or to contacts you add: /contacts add name@example.com"
            )
        return "Fuse may send to:\n" + "\n".join(f"- {a}" for a in listed)
    command, rest = words[0].lower(), " ".join(words[1:])
    if command == "add" and rest:
        try:
            await contacts.add(rest)
        except (InvalidRecipient, ValueError):
            return f"{rest!r} doesn't look like one email address."
        return f"Added {rest.strip().lower()}. Fuse may now send email to them."
    if command == "remove" and rest:
        if await contacts.remove(rest):
            return f"Removed {rest.strip().lower()}."
        return f"{rest} wasn't in your contacts."
    return "Usage: /contacts · /contacts add name@example.com · /contacts remove name@example.com"
