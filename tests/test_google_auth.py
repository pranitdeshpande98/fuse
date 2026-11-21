"""Google sign-in storage. No real Google calls: the tokens here are fake."""

import json
import os

import pytest
from cryptography.fernet import Fernet
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from fuse.crypto import SecretBox
from fuse.db import open_pool
from fuse.google.auth import PROVIDER, GoogleNotConnected, TokenStore, get_credentials

DB_URL = os.environ.get("TEST_DATABASE_URL")


def fake_token(expiry: str) -> str:
    return json.dumps(
        {
            "token": "ya29.fake-access-token",
            "refresh_token": "1//fake-refresh-token",
            "client_id": "fake.apps.googleusercontent.com",
            "client_secret": "fake-secret",
            "token_uri": "https://oauth2.googleapis.com/token",
            "expiry": expiry,
        }
    )


class FakeStore:
    """Stands in for TokenStore in tests that don't need a database."""

    def __init__(self, token: str | None) -> None:
        self.token = token

    async def load(self, provider: str) -> str | None:
        return self.token

    async def save(self, provider: str, token_json: str) -> None:
        self.token = token_json


async def test_not_signed_in_says_how_to_fix_it():
    with pytest.raises(GoogleNotConnected, match="fuse.google.auth"):
        await get_credentials(FakeStore(None))


async def test_expired_sign_in_says_how_to_fix_it(monkeypatch):
    def refresh_fails(self, request):
        raise RefreshError("invalid_grant: Token has been expired or revoked.")

    monkeypatch.setattr(Credentials, "refresh", refresh_fails)
    store = FakeStore(fake_token(expiry="2000-01-01T00:00:00Z"))  # long expired
    with pytest.raises(GoogleNotConnected, match="expired"):
        await get_credentials(store)


@pytest.mark.integration
@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_token_is_stored_encrypted_and_loads_back():
    box = SecretBox(Fernet.generate_key().decode())
    async with open_pool(DB_URL) as pool:
        store = TokenStore(pool, box)
        await store.ensure_schema()
        await store.save(PROVIDER, fake_token(expiry="2999-01-01T00:00:00Z"))

        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT encrypted FROM credentials WHERE provider = %s", (PROVIDER,)
            )
            raw = bytes((await cur.fetchone())["encrypted"])
        assert b"fake-refresh-token" not in raw  # the database never sees the plaintext

        creds = await get_credentials(store)
    assert creds.valid
    assert creds.refresh_token == "1//fake-refresh-token"
