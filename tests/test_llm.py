import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from fuse import llm
from fuse.config import Settings


def test_no_provider_configured_is_a_clear_error():
    settings = Settings(_env_file=None, llm_primary_api_key="", llm_fallback_api_key="")
    with pytest.raises(RuntimeError, match="No LLM provider"):
        llm.build_llm(settings)


def test_providers_without_a_key_are_skipped():
    only_primary = Settings(_env_file=None, llm_primary_api_key="k1", llm_fallback_api_key="")
    assert [p.name for p in only_primary.llm_providers()] == ["primary"]

    both = Settings(_env_file=None, llm_primary_api_key="k1", llm_fallback_api_key="k2")
    assert [p.name for p in both.llm_providers()] == ["primary", "fallback"]


def test_api_keys_stay_hidden():
    settings = Settings(_env_file=None, llm_primary_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(settings.llm_providers())


def test_fallback_answers_when_primary_fails(monkeypatch):
    def broken(_):
        raise ConnectionError("primary is down")

    fakes = {
        "primary": RunnableLambda(broken),
        "fallback": FakeListChatModel(responses=["answer from fallback"]),
    }
    monkeypatch.setattr(llm, "build_chat_model", lambda provider, timeout: fakes[provider.name])

    settings = Settings(_env_file=None, llm_primary_api_key="k1", llm_fallback_api_key="k2")
    reply = llm.build_llm(settings).invoke("hi")
    assert reply.content == "answer from fallback"
