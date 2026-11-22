"""What the /memories and /forget commands reply. Kept separate from Telegram so it's easy
to test, and so Slack or WhatsApp can reuse it later."""

from fuse.memory.store import MemoryStore

HELP = "To delete one: /forget <id>   (e.g. /forget 3)"


async def list_memories_text(store: MemoryStore) -> str:
    memories = await store.list()
    if not memories:
        return "I don't remember anything about you yet."
    profile = [m for m in memories if m.kind == "profile"]
    facts = [m for m in memories if m.kind == "fact"]
    parts = []
    if profile:
        parts.append("Profile:\n" + "\n".join(f"[{m.id}] {m.content}" for m in profile))
    if facts:
        parts.append(
            "Other things I remember:\n" + "\n".join(f"[{m.id}] {m.content}" for m in facts)
        )
    parts.append(HELP)
    return "\n\n".join(parts)


async def forget_text(store: MemoryStore, argument: str) -> str:
    """You typed the command yourself, so this deletes straight away (no extra approval)."""
    if not argument.strip().isdigit():
        return HELP
    memory_id = int(argument.strip())
    memory = await store.get(memory_id)
    if memory is None:
        return f"There is no memory with id {memory_id}. Send /memories to see them."
    await store.delete(memory_id)
    return f"Forgotten: [{memory_id}] {memory.content}"
