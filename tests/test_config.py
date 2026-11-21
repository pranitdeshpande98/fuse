import pytest
from pydantic import ValidationError

from fuse.config import Settings


def test_defaults_without_env_file(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    settings = Settings(_env_file=None)
    assert settings.app_env == "dev"
    assert settings.log_level == "INFO"


def test_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("USER_TIMEZONE", "Asia/Kolkata")
    settings = Settings(_env_file=None)
    assert settings.log_level == "DEBUG"
    assert settings.user_timezone == "Asia/Kolkata"


def test_secrets_are_hidden_when_printed():
    settings = Settings(_env_file=None, telegram_bot_token="123:secret-token")
    assert "secret-token" not in repr(settings)
    assert "secret-token" not in str(settings)
    assert settings.telegram_bot_token.get_secret_value() == "123:secret-token"


def test_bad_values_fail_fast():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="LOUD")
