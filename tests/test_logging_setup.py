import logging

from fuse.logging_setup import configure_logging


def test_logs_go_to_a_file_per_process_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSE_LOG_DIR", str(tmp_path / "logs"))
    configure_logging("INFO", "bot")
    logging.getLogger("fuse.test").info("hello from the bot")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello from the bot" in (tmp_path / "logs" / "bot.log").read_text()
    monkeypatch.delenv("FUSE_LOG_DIR")
    configure_logging("WARNING")  # back to console only, for the other tests
