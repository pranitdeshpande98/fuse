"""Tracing: a record of every agent run (model calls, tool calls, timings) in LangSmith.

LangChain and LangGraph send traces automatically when certain environment variables are
set. Our settings come from .env through pydantic-settings, which does NOT put them into
the process environment, so this copies them across at startup.
"""

import logging
import os

from fuse.config import Settings

log = logging.getLogger("fuse.tracing")


def enable_tracing(settings: Settings) -> bool:
    """Turn LangSmith tracing on if configured. Returns whether it is on."""
    if not settings.langsmith_tracing:
        log.info("Tracing is off (set LANGSMITH_TRACING=true to turn it on)")
        return False
    if not settings.langsmith_api_key.get_secret_value():
        log.warning("LANGSMITH_TRACING is true but LANGSMITH_API_KEY is empty: tracing is off")
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    log.info("Tracing to LangSmith project %r", settings.langsmith_project)
    return True
