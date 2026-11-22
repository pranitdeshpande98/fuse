import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from fuse.jobalerts.client import JobAlertsClient
from fuse.jobalerts.fit import Fit, FitScorer, at_least, level_for, load_profile
from fuse.jobalerts.service import MatchWatcher
from fuse.proactive.events import InMemoryEventStore
from fuse.proactive.notify import RecordingNotifier
from fuse.proactive.schedule import FixedClock
from fuse.proactive.store import InMemoryCursorStore
from tests.fake_job_alerts import FakeJobAlerts

EXAMPLE = Path(__file__).parent.parent / "job_profile.example.md"


def test_levels_and_profile_parsing(tmp_path):
    assert (level_for(0.83), level_for(0.75), level_for(0.6)) == ("strong", "possible", "weak")
    assert at_least(Fit("strong", 0.9, "x"), "possible")
    assert not at_least(Fit("weak", 0.5, "x"), "possible")
    lines = load_profile(EXAMPLE)
    assert len(lines) == 6 and lines[3].startswith("SAP ABAP developer")
    mine = tmp_path / "job_profile.md"
    mine.write_text("# notes\n- Rust developer: systems\nnot a role line\n- \n")
    assert load_profile(mine) == ["Rust developer: systems"]


def job(key, title, minute):
    return {
        "job_key": key,
        "company": "RAND",
        "title": title,
        "url": f"https://x/{key}",
        "first_seen": f"2026-11-22T23:{minute:02d}:00+00:00",
    }


class FixedScorer:
    def __init__(self, by_title):
        self.by_title = by_title

    async def score(self, titles):
        return [self.by_title[t] for t in titles]


def watcher_with(fake, scorer, min_fit="weak"):
    client = JobAlertsClient("http://127.0.0.1:5000", fake.transport())
    notifier = RecordingNotifier()
    clock = FixedClock(datetime(2026, 11, 22, 17, 40, tzinfo=ZoneInfo("America/Chicago")))
    watcher = MatchWatcher(
        client,
        InMemoryCursorStore(),
        InMemoryEventStore(),
        notifier,
        clock,
        scorer=scorer,
        min_fit=min_fit,
    )
    return watcher, notifier


SCORES = {
    "Accountant": Fit("weak", 0.56, "Data engineer"),
    "SAP ABAP Developer": Fit("strong", 0.88, "SAP ABAP developer"),
    "Senior Frontend Engineer": Fit("possible", 0.75, "Backend software engineer"),
}


async def test_best_fits_first_with_the_reason():
    fake = FakeJobAlerts()
    watcher, notifier = watcher_with(fake, FixedScorer(SCORES))
    await watcher.check()
    for i, title in enumerate(SCORES):
        fake.add_job(job(f"n{i}", title, i))
    await watcher.check()
    text, buttons = notifier.sent[0]
    order = [text.index(t) for t in ("SAP ABAP Developer", "Senior Frontend", "Accountant")]
    assert order == sorted(order)
    assert "[strong fit: SAP ABAP developer]" in text and "[weak fit]" in text
    assert buttons[0].label == "Track 1"


async def test_minimum_fit_hides_weaker_ones_but_counts_them():
    fake = FakeJobAlerts()
    watcher, notifier = watcher_with(fake, FixedScorer(SCORES), min_fit="strong")
    await watcher.check()
    for i, title in enumerate(SCORES):
        fake.add_job(job(f"n{i}", title, i))
    await watcher.check()
    text, buttons = notifier.sent[0]
    assert "SAP ABAP Developer" in text and "Accountant" not in text
    assert "(2 weaker fits not shown" in text and len(buttons) == 2  # one job's row


async def test_all_hidden_still_tells_you_without_buttons():
    fake = FakeJobAlerts()
    watcher, notifier = watcher_with(fake, FixedScorer(SCORES), min_fit="strong")
    await watcher.check()
    fake.add_job(job("n1", "Accountant", 1))
    await watcher.check()
    text, buttons = notifier.sent[0]
    assert text.startswith("1 new job match") and buttons == []


@pytest.mark.skipif(os.getenv("FUSE_TEST_EMBEDDINGS") != "1", reason="set FUSE_TEST_EMBEDDINGS=1")
async def test_real_model_separates_fits():
    from fuse.config import get_settings
    from fuse.memory.embeddings import FastEmbedder

    s = get_settings()
    scorer = FitScorer(
        FastEmbedder(s.embedding_model, s.fastembed_cache_dir), load_profile(EXAMPLE)
    )
    strong, nearby, unrelated = await scorer.score(
        ["SAP ABAP Developer", "Senior Mobile iOS Software Developer", "Nurse Practitioner"]
    )
    assert strong.level == "strong" and strong.role == "SAP ABAP developer"
    assert nearby.level == "possible"
    assert unrelated.level == "weak"
