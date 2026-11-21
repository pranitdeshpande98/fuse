"""Settings, loaded from environment variables and a local .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(BaseModel):
    """One OpenAI-compatible endpoint."""

    name: str
    base_url: str
    api_key: SecretStr
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    user_timezone: str = "America/Chicago"

    # Secrets use SecretStr: printing or logging the settings shows ********** instead.
    telegram_bot_token: SecretStr = SecretStr("")
    # Only this Telegram chat may talk to Fuse. 0 means "not set": nobody gets in.
    telegram_allowed_chat_id: int = 0
    database_url: SecretStr = SecretStr("postgresql://fuse:fuse@localhost:55432/fuse")
    # LLM providers: any OpenAI-compatible API (Groq, Cerebras, OpenRouter, local Ollama...)
    llm_primary_base_url: str = "https://api.groq.com/openai/v1"
    llm_primary_api_key: SecretStr = SecretStr("")
    llm_primary_model: str = "openai/gpt-oss-120b"

    llm_fallback_base_url: str = "https://api.cerebras.ai/v1"
    llm_fallback_api_key: SecretStr = SecretStr("")
    llm_fallback_model: str = "gpt-oss-120b"

    llm_timeout_seconds: float = 60.0
    # Max model + tool steps in one turn, so a looping model cannot burn the free quota.
    agent_max_steps: int = 12

    def llm_providers(self) -> list[LLMProvider]:
        """Configured providers in priority order. A provider without an API key is skipped."""
        candidates = [
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
        ]
        return [p for p in candidates if p.api_key.get_secret_value()]


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and reuse them everywhere."""
    return Settings()
