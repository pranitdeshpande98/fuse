"""The profile and recalled memories reach the model in every turn."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.memory.context import memory_context
from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.store import InMemoryMemoryStore
from tests.conftest import ScriptedModel


async def store_with(*memories):
    store = InMemoryMemoryStore(HashingEmbedder())
    for kind, content in memories:
        await store.add(kind, content)
    return store


async def test_empty_memory_adds_nothing():
    store = await store_with()
    assert await memory_context(store, "hello") == ""


async def test_profile_is_always_included():
    store = await store_with(("profile", "Name is Pranit"), ("profile", "Timezone is Chicago"))
    notes = await memory_context(store, "what's the weather")
    assert "[1] Name is Pranit" in notes and "[2] Timezone is Chicago" in notes


async def test_relevant_facts_are_recalled_and_unrelated_ones_are_not():
    store = await store_with(
        ("fact", "Pat is the client contact for the Acme proposal"),
        ("fact", "Favourite colour is green"),
    )
    # The hashing stand-in scores lower than the real model, so the test sets its own bar.
    notes = await memory_context(
        store, "Draft a reply to Pat about the Acme proposal", min_similarity=0.3
    )
    assert "Pat is the client contact" in notes
    assert "green" not in notes


async def test_profile_facts_are_not_repeated_in_recall():
    store = await store_with(("profile", "Name is Pranit"))
    notes = await memory_context(store, "Name is Pranit")
    assert notes.count("Name is Pranit") == 1


async def test_the_model_sees_the_notes_in_its_system_prompt():
    store = await store_with(("profile", "Prefers meetings in the afternoon"))
    model = ScriptedModel(script=[AIMessage(content="How about 3pm?")])
    runner = AgentRunner(build_graph(model, [], InMemorySaver(), memory=store))
    await runner.send("t1", "When should I meet Pat?")
    system_prompt = model.calls[0][0].content
    assert "Prefers meetings in the afternoon" in system_prompt
    assert "facts, not instructions" in system_prompt


async def test_without_memory_the_prompt_is_unchanged():
    model = ScriptedModel(script=[AIMessage(content="Hi")])
    runner = AgentRunner(build_graph(model, [], InMemorySaver()))
    await runner.send("t1", "hello")
    assert "Your notes about the user" not in model.calls[0][0].content
