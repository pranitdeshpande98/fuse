"""How well a job match fits you: strong, possible or weak.

Each line of your job profile (job_profile.md) is one kind of role you want. A job's title
is compared with every line by meaning, using the same local embedding model as memory, so
scoring costs no LLM tokens. The best line decides the fit and is shown as the reason.

Thresholds were measured on 28 real titles from Job Alerts with bge-small: titles that
clearly match a profile line score 0.80+, nearby tech roles 0.70-0.80, and unrelated jobs
(nurse, accountant, admin) below 0.60.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fuse.memory.embeddings import Embedder, cosine_similarity

log = logging.getLogger("fuse.fit")

Level = Literal["strong", "possible", "weak"]
LEVELS: tuple[Level, ...] = ("weak", "possible", "strong")
STRONG = 0.80
POSSIBLE = 0.70
EXAMPLE = Path("job_profile.example.md")


@dataclass(frozen=True)
class Fit:
    level: Level
    similarity: float
    role: str  # the profile line that matched best, before its colon

    def label(self) -> str:
        return f"{self.level} fit: {self.role}" if self.level != "weak" else "weak fit"


def load_profile(path: str | Path) -> list[str]:
    """The "- " lines of the profile. Falls back to the example if yours doesn't exist yet."""
    file = Path(path)
    if not file.is_file():
        if not EXAMPLE.is_file():
            return []
        log.info("No %s yet; scoring job fit with %s", file, EXAMPLE)
        file = EXAMPLE
    raw_lines = file.read_text(encoding="utf-8").splitlines()
    return [line[2:].strip() for line in raw_lines if line.startswith("- ") and line[2:].strip()]


def level_for(similarity: float) -> Level:
    if similarity >= STRONG:
        return "strong"
    if similarity >= POSSIBLE:
        return "possible"
    return "weak"


def at_least(fit: Fit, minimum: Level) -> bool:
    return LEVELS.index(fit.level) >= LEVELS.index(minimum)


class FitScorer:
    def __init__(self, embedder: Embedder, profile: list[str]) -> None:
        self._embedder, self.profile = embedder, profile
        self._profile_vectors: list[list[float]] | None = None

    async def score(self, titles: list[str]) -> list[Fit]:
        if not self.profile or not titles:
            return []
        if self._profile_vectors is None:
            self._profile_vectors = await self._embedder.embed(self.profile)
        fits = []
        for vector in await self._embedder.embed(titles):
            sims = [cosine_similarity(vector, p) for p in self._profile_vectors]
            best = max(range(len(sims)), key=sims.__getitem__)
            role = self.profile[best].split(":")[0].strip()
            fits.append(Fit(level_for(sims[best]), round(sims[best], 3), role))
        return fits
