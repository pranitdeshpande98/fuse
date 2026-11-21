"""Saving drafts to Gmail, with a fake Gmail service: no network, no account needed."""

import base64
import email

import httplib2
import pytest
from googleapiclient.errors import HttpError

from fuse.google.auth import GoogleNotConnected
from fuse.google.drafts import GmailDraftSink, build_raw_message
from fuse.tools.drafts import DraftError


class FakeGmail:
    """Mimics the chain gmail.users().drafts().create(...).execute()."""

    def __init__(self) -> None:
        self.created: list[dict] = []

    def users(self):
        return self

    def drafts(self):
        return self

    def create(self, userId, body):
        self.created.append(body)
        return self

    def execute(self):
        return {"id": "draft-123"}


def sink_with(monkeypatch, gmail=None, error: Exception | None = None) -> GmailDraftSink:
    async def fake_gmail(self):
        if error:
            raise error
        return gmail

    monkeypatch.setattr(GmailDraftSink, "_gmail", fake_gmail)
    return GmailDraftSink(tokens=None)


def test_raw_message_is_a_valid_email():
    raw = build_raw_message("pat@example.com", "Re: Proposal", "Friday works.")
    parsed = email.message_from_bytes(base64.urlsafe_b64decode(raw))
    assert parsed["To"] == "pat@example.com"
    assert parsed["Subject"] == "Re: Proposal"
    assert parsed.get_payload().strip() == "Friday works."


async def test_draft_is_created_in_gmail(monkeypatch):
    gmail = FakeGmail()
    result = await sink_with(monkeypatch, gmail).save_draft("pat@example.com", "Hi", "Hello")
    assert "draft-123" in result and "not sent" in result
    assert len(gmail.created) == 1
    assert "raw" in gmail.created[0]["message"]


async def test_not_connected_becomes_a_clear_draft_error(monkeypatch):
    sink = sink_with(monkeypatch, error=GoogleNotConnected("Google is not connected. Run: ..."))
    with pytest.raises(DraftError, match="not connected"):
        await sink.save_draft("pat@example.com", "Hi", "Hello")


async def test_gmail_errors_become_a_clear_draft_error(monkeypatch):
    sink = sink_with(monkeypatch, error=HttpError(httplib2.Response({"status": 403}), b""))
    with pytest.raises(DraftError, match="403"):
        await sink.save_draft("pat@example.com", "Hi", "Hello")
