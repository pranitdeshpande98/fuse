"""Settings, loaded from environment variables and a local .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and reuse them everywhere."""
    return Settings()
