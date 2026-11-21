"""Google sign-in: get your permission once, store the token encrypted, refresh it as needed.

Sign in (opens your browser):
    uv run python -m fuse.google.auth
"""

import asyncio
import json
import logging

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from psycopg_pool import AsyncConnectionPool

from fuse.compat import use_selector_event_loop_on_windows
from fuse.config import Settings, get_settings
from fuse.crypto import SecretBox
from fuse.db import open_pool

log = logging.getLogger("fuse.google")

# What Fuse may do with your Google account.
# gmail.compose is Google's narrowest permission that allows creating drafts. It would also
# allow sending, but Fuse has no send tool, and every draft needs your Approve first.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
PROVIDER = "google"
SIGN_IN_HINT = "Google is not connected. Run: uv run python -m fuse.google.auth"

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS credentials (
        provider    TEXT PRIMARY KEY,
        encrypted   BYTEA       NOT NULL,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
]


class GoogleNotConnected(RuntimeError):
    """Raised when there is no usable Google token; the message says how to fix it."""


class TokenStore:
    """Keeps OAuth tokens in Postgres, encrypted. The plaintext never touches the disk."""

    def __init__(self, pool: AsyncConnectionPool, box: SecretBox) -> None:
        self._pool = pool
        self._box = box

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def save(self, provider: str, token_json: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO credentials (provider, encrypted) VALUES (%s, %s)
                ON CONFLICT (provider)
                DO UPDATE SET encrypted = EXCLUDED.encrypted, updated_at = now()
                """,
                (provider, self._box.encrypt(token_json)),
            )

    async def load(self, provider: str) -> str | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT encrypted FROM credentials WHERE provider = %s", (provider,)
            )
            row = await cur.fetchone()
        return self._box.decrypt(bytes(row["encrypted"])) if row else None


async def get_credentials(store: TokenStore, provider: str = PROVIDER) -> Credentials:
    """Valid Google credentials, refreshed if needed. Raises GoogleNotConnected otherwise."""
    try:
        token_json = await store.load(provider)
    except RuntimeError as exc:  # stored with a different ENCRYPTION_KEY
        raise GoogleNotConnected(
            f"The saved Google sign-in can't be decrypted (was ENCRYPTION_KEY changed?). "
            f"{SIGN_IN_HINT}"
        ) from exc
    if token_json is None:
        raise GoogleNotConnected(SIGN_IN_HINT)

    info = json.loads(token_json)
    missing = set(SCOPES) - set(info.get("scopes") or [])
    if missing:  # signed in before Fuse needed these permissions
        raise GoogleNotConnected(f"Fuse needs updated Google permissions. {SIGN_IN_HINT}")

    creds = Credentials.from_authorized_user_info(info)
    if creds.valid:
        return creds
    if not creds.refresh_token:
        raise GoogleNotConnected(SIGN_IN_HINT)
    try:
        # The Google library is synchronous; run it in a thread so the bot doesn't freeze.
        await asyncio.to_thread(creds.refresh, Request())
    except RefreshError as exc:
        # Expired or revoked (Testing-mode sign-ins can expire after about a week).
        raise GoogleNotConnected(f"Google sign-in expired. {SIGN_IN_HINT}") from exc
    await store.save(provider, creds.to_json())
    return creds


def sign_in_with_browser(client_secrets_file: str) -> Credentials:
    """Open Google's consent page in your browser and wait for you to approve."""
    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
    return flow.run_local_server(port=0, prompt="consent")


def connected_email(creds: Credentials) -> str:
    gmail = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return gmail.users().getProfile(userId="me").execute()["emailAddress"]


async def _sign_in(settings: Settings) -> None:
    creds = sign_in_with_browser(settings.google_client_secrets_file)
    box = SecretBox(settings.encryption_key.get_secret_value())
    async with open_pool(settings.database_url.get_secret_value()) as pool:
        store = TokenStore(pool, box)
        await store.ensure_schema()
        await store.save(PROVIDER, creds.to_json())
        creds = await get_credentials(store)  # read it back: proves storage + decryption work
    email = await asyncio.to_thread(connected_email, creds)
    print(f"Connected Google account: {email}")


def main() -> None:
    use_selector_event_loop_on_windows()
    settings = get_settings()
    SecretBox(settings.encryption_key.get_secret_value())  # fail early if the key is missing
    asyncio.run(_sign_in(settings))


if __name__ == "__main__":
    main()
