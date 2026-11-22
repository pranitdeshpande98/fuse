"""A token budget for one task: stops a task that would burn through the free quota.

Every model reply reports how many tokens it used. This callback adds them up and raises
once the task goes over its budget, which ends the run at the next model call.
"""

from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


class BudgetExceeded(Exception):
    pass


def tokens_used(response: LLMResult) -> int:
    total = 0
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) or {}
            total += int(usage.get("total_tokens", 0))
    if total == 0 and response.llm_output:  # some providers only report it here
        usage = response.llm_output.get("token_usage") or {}
        total = int(usage.get("total_tokens", 0))
    return total


class TokenBudget(AsyncCallbackHandler):
    raise_error = True  # without this, LangChain would log the exception and carry on

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.used += tokens_used(response)
        if self.used > self.limit:
            raise BudgetExceeded(f"used {self.used:,} tokens (budget {self.limit:,})")
