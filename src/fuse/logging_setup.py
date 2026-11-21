"""Logging setup shared by every entry point."""

import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every request URL, and Telegram API URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
