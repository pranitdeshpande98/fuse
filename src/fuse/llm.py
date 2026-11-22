"""Chat model with a provider fallback chain.

Every provider speaks the OpenAI Chat Completions format, so switching provider only means
a different base URL and model id. If the primary fails (rate limit, outage, timeout), the
same request is retried on the fallback.
"""

import sys
from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from fuse.config import LLMProvider, Settings, get_settings
from fuse.usage import USAGE_CALLBACK


def build_chat_model(provider: LLMProvider, timeout: float) -> BaseChatModel:
    return ChatOpenAI(
        model=provider.model,
        base_url=provider.base_url,
        api_key=provider.api_key,
        temperature=0,
        timeout=timeout,
        max_retries=1,
        callbacks=[USAGE_CALLBACK],  # counts tokens per model per day (see fuse.usage)
    )


Tier = Literal["main", "light"]


def build_llm(settings: Settings, tools: Sequence[BaseTool] = (), tier: Tier = "main") -> Runnable:
    """The primary model, falling back to the next configured provider on any error.

    Tools are bound to each model before the fallback chain is built, so a fallback model
    can call the same tools as the primary.
    """
    providers = settings.llm_providers(tier)
    if not providers:
        raise RuntimeError("No LLM provider configured: set LLM_PRIMARY_API_KEY in .env")
    models = [build_chat_model(p, settings.llm_timeout_seconds) for p in providers]
    if tools:
        models = [m.bind_tools(list(tools)) for m in models]
    primary, *fallbacks = models
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def build_structured_llm(settings: Settings, schema: type, tier: Tier = "main") -> Runnable:
    """A model that answers with an instance of `schema` (a pydantic class), with the same
    fallback chain. Uses tool calling under the hood, which every provider here supports."""
    providers = settings.llm_providers(tier)
    if not providers:
        raise RuntimeError("No LLM provider configured: set LLM_PRIMARY_API_KEY in .env")
    models = [
        build_chat_model(p, settings.llm_timeout_seconds).with_structured_output(
            schema, method="function_calling"
        )
        for p in providers
    ]
    primary, *fallbacks = models
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Say hello in five words."
    reply = build_llm(get_settings()).invoke(question)
    print(reply.content)
    print(f"\n[model: {reply.response_metadata.get('model_name')}]")


if __name__ == "__main__":
    main()
