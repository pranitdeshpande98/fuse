"""Logging setup shared by every entry point."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO", name: str | None = None) -> None:
    """Log to the console, and also to <FUSE_LOG_DIR>/<name>.log when FUSE_LOG_DIR is set
    (scripts\\start_all.bat sets it), so the bot's and the worker's logs can be read later.
    Each process gets its own file, rotated at 2 MB with 3 old copies kept."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_dir = os.environ.get("FUSE_LOG_DIR", "").strip()
    if log_dir and name:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                Path(log_dir) / f"{name}.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
        )
    logging.basicConfig(level=level, format=FORMAT, handlers=handlers, force=True)
    # httpx logs every request URL, and Telegram API URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
