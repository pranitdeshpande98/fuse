"""Memory store behaviour, tested on both implementations with the same checks.

The in-memory store always runs. The Postgres store runs when TEST_DATABASE_URL is set.
Both use HashingEmbedder, so no model download is needed.
"""

import os
import uuid

import pytest

from fuse.db import open_pool
from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.store import InMemoryMemoryStore, PostgresMemoryStore

DB_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(params=["memory", "postgres"])
async def store(request):
    embedder = HashingEmbedder()
    if request.param == "memory":
        yield InMemoryMemoryStore(embedder)
        return
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    async with open_pool(DB_URL) as pool:
        store = PostgresMemoryStore(pool, embedder)
        await store.ensure_schema()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM memories")  # each test starts empty
        yield store


def unique(text: str) -> str:
    return f"{text} {uuid.uuid4().hex[:6]}"


async def test_add_and_search_finds_the_closest_memory(store):
    await store.add("fact", "Pat is the client contact for the Acme proposal")
    await store.add("fact", "The user's favourite colour is green")
    results = await store.search("who is the contact at Acme", limit=1)
    assert results[0].content.startswith("Pat is the client contact")
    assert results[0].similarity > 0


async def test_near_duplicates_are_not_stored_twice(store):
    first = await store.add("fact", "The user prefers afternoon meetings")
    second = await store.add("fact", "The user prefers afternoon meetings")
    assert second.id == first.id
    assert len(await store.list()) == 1


async def test_supersede_replaces_without_losing_history(store):
    old = await store.add("profile", "Prefers meetings in the afternoon")
    new = await store.supersede(old.id, "Prefers meetings in the morning")
    contents = [m.content for m in await store.list()]
    assert contents == ["Prefers meetings in the morning"]  # only the current belief shows
    assert await store.get(old.id) is None
    assert (await store.get(new.id)).kind == "profile"  # keeps the kind


async def test_list_shows_profile_first_and_filters_by_kind(store):
    await store.add("fact", "Works on the Fuse project")
    await store.add("profile", "Name is Pranit")
    everything = await store.list()
    assert everything[0].kind == "profile"
    assert [m.content for m in await store.list("fact")] == ["Works on the Fuse project"]


async def test_delete_forgets_the_memory_and_its_older_versions(store):
    old = await store.add("fact", unique("Lives in Chicago"))
    new = await store.supersede(old.id, "Lives in Baltimore")
    assert await store.delete(new.id) is True
    assert await store.list() == []
    assert await store.delete(new.id) is False  # nothing left to delete


async def test_min_similarity_filters_weak_matches(store):
    await store.add("fact", "The user's favourite colour is green")
    assert await store.search("quarterly tax filing deadline", min_similarity=0.5) == []


async def test_supersede_unknown_memory_is_an_error(store):
    with pytest.raises(KeyError):
        await store.supersede(999999, "anything")
