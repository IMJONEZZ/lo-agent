"""Always-on rotating debug log — the file to ask a bug reporter for.

The TUI owns the terminal, so nothing may print to stderr while it runs: a
diagnostic printed mid-session either corrupts the screen or scrolls away with
it. Everything worth knowing after the fact goes here instead, and every error
we explain to the user points at this path.

On by default at DEBUG (the log is small, rotated, and useless if it only turns
on after the bug happens). `LO_LOG_LEVEL=warning` quiets it, `LO_LOG_LEVEL=off`
disables it, `LO_LOG_DIR` moves it.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path

_MAX_BYTES = 2 * 1024 * 1024  # 2 MB × 3 — a long session, no unbounded growth
_BACKUPS = 3
_configured = False


def log_dir() -> Path:
    return Path(os.environ.get("LO_LOG_DIR") or "~/.lo/logs").expanduser()


def log_path() -> Path:
    return log_dir() / "lo.log"


def setup(command: str = "") -> Path | None:
    """Attach the rotating file handler (idempotent). Returns the log path, or
    None if logging is off or the directory isn't writable — never raises: a
    broken log must not break the command the user actually asked for."""
    global _configured
    if _configured:
        return log_path()
    level_name = (os.environ.get("LO_LOG_LEVEL") or "debug").strip().lower()
    if level_name in ("off", "none", "0"):
        return None
    level = getattr(logging, level_name.upper(), logging.DEBUG)
    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
        )
    except OSError:
        return None
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(min(level, root.level or level))
    root.addHandler(handler)
    # Third-party libraries at DEBUG bury our own lines (httpx logs every request
    # body chunk, urllib3 every socket read).
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "markdown_it", "PIL"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
    _configured = True
    _log_header(command)
    _install_excepthooks()
    return path


def _log_header(command: str) -> None:
    """One line per run with everything a bug report needs but never includes."""
    import platform
    import sqlite3

    log = logging.getLogger("lo")
    try:
        from . import __version__ as version
    except Exception:  # noqa: BLE001 — a version lookup must never abort startup
        version = "?"
    log.info(
        "--- lo %s · %s · python %s · sqlite %s · %s · cwd=%s · argv=%s",
        version,
        command or "?",
        platform.python_version(),
        sqlite3.sqlite_version,
        platform.platform(),
        os.getcwd(),
        " ".join(sys.argv[1:]),
    )


def _install_excepthooks() -> None:
    """Crashes reach the log even when the traceback scrolls past the user."""
    log = logging.getLogger("lo")
    prev = sys.excepthook

    def hook(exc_type, exc, tb):
        log.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        prev(exc_type, exc, tb)

    sys.excepthook = hook

    prev_thread = threading.excepthook

    def thread_hook(arg):
        log.critical(
            "uncaught exception in thread %s",
            arg.thread.name if arg.thread else "?",
            exc_info=(arg.exc_type, arg.exc_value, arg.exc_traceback),
        )
        prev_thread(arg)

    threading.excepthook = thread_hook


def tail(n: int = 40) -> str:
    """Last n lines of the log (for `lo doctor` and in-app error surfaces)."""
    try:
        lines = log_path().read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])
