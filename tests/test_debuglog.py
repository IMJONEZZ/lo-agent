"""The rotating debug log — the file a bug reporter can actually attach."""

import logging
import sys
import threading

import pytest

from local_harness import debuglog


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """Point the log at tmp_path and undo every global it touches."""
    monkeypatch.setenv("LO_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(debuglog, "_configured", False)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    hooks = (sys.excepthook, threading.excepthook)
    try:
        yield tmp_path
    finally:
        for h in list(root.handlers):
            if h not in handlers:
                root.removeHandler(h)
                h.close()
        root.setLevel(level)
        sys.excepthook, threading.excepthook = hooks


def test_setup_records_the_environment_a_bug_report_never_includes(isolated_logging):
    path = debuglog.setup("tui")
    assert path == isolated_logging / "lo.log"

    logging.getLogger("lo.events").warning("WAL unavailable for /Volumes/box/lo.db")
    text = debuglog.tail()
    assert "lo " in text and "tui" in text          # version + which command ran
    assert "sqlite" in text and "python" in text    # the builds involved
    assert "WAL unavailable for /Volumes/box/lo.db" in text


def test_setup_is_idempotent(isolated_logging):
    debuglog.setup("run")
    before = len(logging.getLogger().handlers)
    debuglog.setup("run")
    assert len(logging.getLogger().handlers) == before  # no duplicate lines


def test_logging_can_be_turned_off(isolated_logging, monkeypatch):
    monkeypatch.setenv("LO_LOG_LEVEL", "off")
    assert debuglog.setup("run") is None
    assert not (isolated_logging / "lo.log").exists()


def test_an_unwritable_log_dir_never_breaks_the_command(isolated_logging, monkeypatch):
    monkeypatch.setenv("LO_LOG_DIR", str(isolated_logging / "nope" / "logs"))
    monkeypatch.setattr(debuglog.Path, "mkdir",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError()))
    assert debuglog.setup("run") is None  # degraded silently, no traceback
