import pytest

from tests.conftest import pytest_configure


def test_refuses_a_database_not_named_test(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://fuse:fuse@127.0.0.1:55432/fuse")
    with pytest.raises(pytest.UsageError, match="_test"):
        pytest_configure(None)


def test_accepts_a_test_database(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://fuse:fuse@127.0.0.1:55432/fuse_test")
    pytest_configure(None)  # no error
