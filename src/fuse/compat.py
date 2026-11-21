"""Platform workarounds."""

import asyncio
import sys


def use_selector_event_loop_on_windows() -> None:
    """psycopg's async mode cannot run on Windows' default event loop (Proactor).

    Call this once, before any event loop starts.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
