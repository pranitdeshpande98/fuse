"""Memory tools, with the in-memory store and the hashing embedder (no model download)."""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fuse.actions import InMemoryActionStore
from fuse.agent.graph import build_graph
from fuse.agent.runner import AgentRunner
from fuse.memory.embeddings import HashingEmbedder
from fuse.memory.store import InMemoryMemoryStore
from fuse.memory.tools import make_memory_tools
from tests.conftest import ScriptedModel, tool_call


def setup(related_similarity: float = 0.8):
    store = InMemoryMemoryStore(HashingEmbedder())
    actions = InMemoryActionStore()
    tools = {t.name: t for t in make_memory_tools(store, actions, related_similarity)}
    return store, actions, tools


async def test_remember_saves_a_memory():
    store, _, tools = setup()
    out = await tools["remember"].ainvoke({"content": "Name is Pranit", "kind": "profile"})
    assert out.startswith("Remembered [1] (profile)")
    assert [m.content for m in await store.list("profile")] == ["Name is Pranit"]


async def test_remember_asks_before_saving_something_similar():
    store, _, tools = setup(related_similarity=0.6)
    await tools["remember"].ainvoke({"content": "Prefers afternoon meetings"})
    out = await tools["remember"].ainvoke({"content": "Prefers morning meetings"})
    assert out.startswith("Not saved yet")
    assert "update_memory" in out
    assert len(await store.list()) == 1


async def test_force_saves_a_separate_but_similar_fact():
    store, _, tools = setup(related_similarity=0.6)
    await tools["remember"].ainvoke({"content": "Prefers afternoon meetings"})
    await tools["remember"].ainvoke({"content": "Prefers morning meetings", "force": True})
    assert len(await store.list()) == 2


async def test_update_replaces_the_old_memory():
    store, _, tools = setup()
    await tools["remember"].ainvoke({"content": "Prefers afternoon meetings"})
    out = await tools["update_memory"].ainvoke(
        {"memory_id": 1, "new_content": "Prefers morning meetings"}
    )
    assert "replaced by [2]" in out
    assert [m.content for m in await store.list()] == ["Prefers morning meetings"]


async def test_update_unknown_id_is_explained():
    _, _, tools = setup()
    out = await tools["update_memory"].ainvoke({"memory_id": 42, "new_content": "x"})
    assert out == "There is no memory with id 42."


async def test_search_memory_finds_by_content():
    _, _, tools = setup()
    await tools["remember"].ainvoke({"content": "Pat is the client contact at Acme"})
    out = await tools["search_memory"].ainvoke({"query": "contact at Acme"})
    assert "Pat is the client contact at Acme" in out


async def test_forget_needs_approval_and_then_deletes():
    store, actions, _ = setup()
    await store.add("fact", "Lives in Chicago")
    model = ScriptedModel(
        script=[tool_call("forget_memory", {"memory_id": 1}), AIMessage(content="Forgotten.")]
    )
    runner = AgentRunner(build_graph(model, make_memory_tools(store, actions), InMemorySaver()))

    reply = await runner.send("t1", "Forget where I live")
    assert reply.kind == "approval"
    assert "Lives in Chicago" in reply.approval["preview"]
    assert len(await store.list()) == 1  # nothing deleted before approval

    reply = await runner.resume("t1", approved=True)
    assert reply.text == "Forgotten."
    assert await store.list() == []


async def test_rejected_forget_keeps_the_memory():
    store, actions, _ = setup()
    await store.add("fact", "Lives in Chicago")
    model = ScriptedModel(
        script=[tool_call("forget_memory", {"memory_id": 1}), AIMessage(content="Kept it.")]
    )
    runner = AgentRunner(build_graph(model, make_memory_tools(store, actions), InMemorySaver()))
    await runner.send("t1", "Forget where I live")
    await runner.resume("t1", approved=False)
    assert len(await store.list()) == 1
