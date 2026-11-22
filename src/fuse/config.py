"""Settings, loaded from environment variables and a local .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(BaseModel):
    """One OpenAI-compatible endpoint."""

    name: str
    base_url: str
    api_key: SecretStr
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    user_timezone: str = "America/Chicago"

    # Database (step 4). Contains a password, so it is a secret too.
    database_url: SecretStr = SecretStr("postgresql://fuse:fuse@127.0.0.1:55432/fuse")

    # Telegram (step 5). Secrets use SecretStr: printing or logging shows ********** instead.
    telegram_bot_token: SecretStr = SecretStr("")
    # Only this Telegram chat may talk to Fuse. 0 means "not set": nobody gets in.
    telegram_allowed_chat_id: int = 0

    # LLM providers (step 6): any OpenAI-compatible API (Groq, Cerebras, OpenRouter, Ollama...)
    llm_primary_base_url: str = "https://api.groq.com/openai/v1"
    llm_primary_api_key: SecretStr = SecretStr("")
    llm_primary_model: str = "openai/gpt-oss-120b"

    llm_fallback_base_url: str = "https://api.cerebras.ai/v1"
    llm_fallback_api_key: SecretStr = SecretStr("")
    llm_fallback_model: str = "gpt-oss-120b"

    # A third provider, optional: tried after the fallback.
    llm_fallback2_base_url: str = ""
    llm_fallback2_api_key: SecretStr = SecretStr("")
    llm_fallback2_model: str = ""

    # Background work (email triage, the brief, memory consolidation) uses this smaller model
    # on the primary provider first, keeping the big model's daily quota for your chats.
    # Groq counts limits per model, so this is quota the big model doesn't use. Empty = off.
    llm_light_model: str = "openai/gpt-oss-20b"

    llm_timeout_seconds: float = 60.0

    # Agent (step 7): max model + tool steps in one turn, so a loop can't burn the free quota.
    agent_max_steps: int = 12
    # How many recent messages the model sees each turn (phase 2). Long-term memory holds
    # what matters from older ones; the full history is still kept in the database.
    agent_history_messages: int = 30

    # Google sign-in (step 10)
    google_client_secrets_file: str = "credentials.json"
    # Encrypts stored Google tokens. Generate once (see .env.example) and never change it.
    encryption_key: SecretStr = SecretStr("")

    # Memory (phase 2): a local embedding model, downloaded once into fastembed_cache_dir.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_cache_dir: str = ".cache/fastembed"

    # Proactive (phase 3): Fuse messages you first. Times are HH:MM in user_timezone.
    proactive_enabled: bool = True
    brief_time: str = "07:30"
    quiet_hours_start: str = "22:00"  # no alerts from here...
    quiet_hours_end: str = "07:00"  # ...until here; they wait for the morning brief
    gmail_poll_seconds: int = 120
    max_alerts_per_hour: int = 5
    meeting_reminder_minutes: int = 15
    consolidation_time: str = "03:00"  # nightly memory consolidation

    @field_validator("brief_time", "quiet_hours_start", "quiet_hours_end", "consolidation_time")
    @classmethod
    def _check_hhmm(cls, value: str) -> str:
        from fuse.proactive.schedule import parse_hhmm

        parse_hhmm(value)  # raises a clear error for anything that isn't HH:MM
        return value

    # Skills (phase 5): one folder per skill, each with a SKILL.md
    skills_dir: str = "skills"

    # Web search (phase 4): DuckDuckGo, no API key.
    web_search_enabled: bool = True

    # Background tasks (phase 4): limits for one task, run by `uv run fuse-worker`.
    task_max_steps: int = 40
    task_timeout_minutes: int = 15
    task_max_attempts: int = 3
    # Per run. Groq's free tier allows 200k tokens a day per model, so one task must not
    # be able to use all of it.
    task_max_tokens: int = 40_000

    # Tracing (step 11): send a record of every agent run to LangSmith for debugging.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "fuse"

    def llm_providers(self, tier: Literal["main", "light"] = "main") -> list[LLMProvider]:
        """Configured providers in the order to try them. A provider without an API key is
        skipped, and the same model on the same endpoint is only tried once.

        "main" is for chatting with you and background tasks. "light" tries the smaller
        model first, for high-volume background work, then falls back to the main chain.
        """
        main = [
            LLMProvider(
                name="primary",
                base_url=self.llm_primary_base_url,
                api_key=self.llm_primary_api_key,
                model=self.llm_primary_model,
            ),
            LLMProvider(
                name="fallback",
                base_url=self.llm_fallback_base_url,
                api_key=self.llm_fallback_api_key,
                model=self.llm_fallback_model,
            ),
            LLMProvider(
                name="fallback2",
                base_url=self.llm_fallback2_base_url,
                api_key=self.llm_fallback2_api_key,
                model=self.llm_fallback2_model,
            ),
        ]
        candidates = main
        if tier == "light" and self.llm_light_model:
            light = LLMProvider(
                name="light",
                base_url=self.llm_primary_base_url,
                api_key=self.llm_primary_api_key,
                model=self.llm_light_model,
            )
            candidates = [light, *main]
        chosen: list[LLMProvider] = []
        seen: set[tuple[str, str]] = set()
        for p in candidates:
            key = (p.base_url.rstrip("/"), p.model)
            if p.api_key.get_secret_value() and p.base_url and p.model and key not in seen:
                seen.add(key)
                chosen.append(p)
        return chosen


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and reuse them everywhere."""
    return Settings()
