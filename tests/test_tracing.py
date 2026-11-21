import os

import pytest

from fuse.config import Settings
from fuse.tracing import enable_tracing

TRACING_VARS = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]


@pytest.fixture(autouse=True)
def clean_environment():
    """Each test starts with no tracing variables, and they are removed again afterwards,
    so no other test ever tries to send real traces."""
    saved = {name: os.environ.pop(name) for name in TRACING_VARS if name in os.environ}
    yield
    for name in TRACING_VARS:
        os.environ.pop(name, None)
    os.environ.update(saved)


def test_off_by_default():
    assert enable_tracing(Settings(_env_file=None)) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_on_without_a_key_stays_off():
    settings = Settings(_env_file=None, langsmith_tracing=True)
    assert enable_tracing(settings) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_on_with_a_key_sets_the_environment():
    settings = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key="lsv2_test",
        langsmith_project="fuse-test",
    )
    assert enable_tracing(settings) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test"
    assert os.environ["LANGSMITH_PROJECT"] == "fuse-test"


def test_tracing_variables_do_not_leak_between_tests():
    assert "LANGSMITH_TRACING" not in os.environ
