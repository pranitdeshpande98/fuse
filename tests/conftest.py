"""Shared test helpers."""

import asyncio
import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def pytest_asyncio_loop_factories(config, item):
    """Tell pytest-asyncio which event loop to create for async tests.

    psycopg's async mode can't run on Windows' default (Proactor) loop, so on Windows
    we hand pytest-asyncio the selector loop directly.
    """
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return {"default": asyncio.new_event_loop}


class ScriptedModel(BaseChatModel):
    """A fake chat model that returns pre-written replies in order and records its inputs.

    Lets us test the agent loop without calling a real LLM.
    """

    script: list[AIMessage]
    calls: list[list[Any]] = []
    position: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if self.position >= len(self.script):
            raise AssertionError("ScriptedModel ran out of scripted replies")
        message = self.script[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    """An AI message that asks to call one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
