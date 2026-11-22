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


def test_up_to_three_providers_in_order_and_duplicates_once():
    groq = "https://api.groq.com/openai/v1"
    settings = Settings(
        _env_file=None,
        llm_primary_base_url=groq,
        llm_primary_api_key="g",
        llm_primary_model="openai/gpt-oss-120b",
        llm_fallback_base_url=groq,
        llm_fallback_api_key="g",
        llm_fallback_model="openai/gpt-oss-20b",
        llm_fallback2_base_url="https://openrouter.ai/api/v1",
        llm_fallback2_api_key="o",
        llm_fallback2_model="some/free-model",
    )
    assert [p.model for p in settings.llm_providers()] == [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "some/free-model",
    ]
    # light work tries the small model first; it's already the fallback, so it isn't repeated
    assert [p.model for p in settings.llm_providers("light")] == [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "some/free-model",
    ]


def test_light_tier_can_be_turned_off():
    settings = Settings(_env_file=None, llm_primary_api_key="k", llm_light_model="")
    assert [p.name for p in settings.llm_providers("light")] == ["primary"]


def test_task_budget_fits_inside_a_free_daily_quota():
    assert Settings(_env_file=None).task_max_tokens <= 50_000
