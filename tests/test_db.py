import os

import psycopg
import pytest

from fuse.db import check_database


def test_unreachable_database_fails_fast():
    with pytest.raises(psycopg.OperationalError):
        check_database("postgresql://nobody:wrong@127.0.0.1:1/nothing")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_database_is_reachable_with_pgvector():
    info = check_database(os.environ["TEST_DATABASE_URL"])
    assert info["pgvector"] != "not installed"
