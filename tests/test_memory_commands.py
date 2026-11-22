from fuse.memory.commands import HELP, forget_text, list_memories_text
from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.store import InMemoryMemoryStore


async def make_store():
    store = InMemoryMemoryStore(HashingEmbedder())
    await store.add("fact", "Pat is the client contact at Acme")
    await store.add("profile", "Name is Pranit")
    return store


async def test_empty_list():
    store = InMemoryMemoryStore(HashingEmbedder())
    assert await list_memories_text(store) == "I don't remember anything about you yet."


async def test_list_shows_profile_then_facts_with_ids():
    text = await list_memories_text(await make_store())
    assert text.index("Profile:") < text.index("Other things I remember:")
    assert "[2] Name is Pranit" in text
    assert "[1] Pat is the client contact at Acme" in text
    assert text.endswith(HELP)


async def test_forget_deletes_by_id():
    store = await make_store()
    assert await forget_text(store, "1") == "Forgotten: [1] Pat is the client contact at Acme"
    assert [m.id for m in await store.list()] == [2]


async def test_forget_unknown_id():
    reply = await forget_text(await make_store(), "99")
    assert reply.startswith("There is no memory with id 99")


async def test_forget_without_a_number_shows_help():
    assert await forget_text(await make_store(), "") == HELP
    assert await forget_text(await make_store(), "pat") == HELP
