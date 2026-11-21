"""Chat model with a provider fallback chain.

Every provider speaks the OpenAI Chat Completions format, so switching provider only means
a different base URL and model id. If the primary fails (rate limit, outage, timeout), the
same request is retried on the fallback.
"""

import sys

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from fuse.config import LLMProvider, Settings, get_settings


def build_chat_model(provider: LLMProvider, timeout: float) -> BaseChatModel:
    return ChatOpenAI(
        model=provider.model,
        base_url=provider.base_url,
        api_key=provider.api_key,
        temperature=0,
        timeout=timeout,
        max_retries=1,
    )


def build_llm(settings: Settings) -> Runnable:
    """The primary model, falling back to the next configured provider on any error."""
    providers = settings.llm_providers()
    if not providers:
        raise RuntimeError("No LLM provider configured: set LLM_PRIMARY_API_KEY in .env")
    models = [build_chat_model(p, settings.llm_timeout_seconds) for p in providers]
    primary, *fallbacks = models
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Say hello in five words."
    reply = build_llm(get_settings()).invoke(question)
    print(reply.content)
    print(f"\n[model: {reply.response_metadata.get('model_name')}]")


if __name__ == "__main__":
    main()
