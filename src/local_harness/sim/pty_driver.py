"""PtyTui: the scenario Driver over a REAL terminal — the TUI as a subprocess
in a pseudo-terminal, fed raw keystrokes, observed through its ANSI output.

This is the only driver that exercises the true terminal boundary (process
boot, key encoding, real rendering) and the one `lo simulate` and the live
tier use against real model endpoints. The PTY/marker machinery originated in
demos/tui_record.py (which is now a thin wrapper over this class); recording
an asciicast v2 is an optional side effect (`record_to=`).
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[\(\)][AB0]|\x1b\].*?\x07")


def _plain(b: bytes) -> str:
    return _ANSI.sub("", b.decode("utf-8", "replace"))


def tui_command(url: str, db: str, *extra: str, model: str = "") -> list[str]:
    """A `lo tui` invocation runnable without `uv run` overhead (same venv)."""
    cmd = [
        sys.executable,
        "-c",
        "from local_harness.cli.main import main; main()",
        "tui",
        "--url",
        url,
        "--db",
        db,
        *extra,
    ]
    if model:
        cmd += ["--model", model]
    return cmd


class PtyTui:
    """Launch `cmd` in a PTY and drive it as a scenario Driver.

    Marker semantics are byte-offset based (true append-only output stream):
    `mark()` is an offset, `wait_marker` searches output produced after it —
    exactly demos/tui_record.py's proven wait technique."""

    def __init__(
        self,
        cmd: list[str],
        *,
        env: dict | None = None,
        cwd: str | None = None,
        cols: int = 120,
        rows: int = 42,
        record_to: str | None = None,
        title: str = "local_harness user simulation",
    ):
        self.cols, self.rows = cols, rows
        self.record_to, self.title = record_to, title
        self.events: list = []  # asciicast v2 output events
        self.buf = bytearray()
        self._t0 = time.monotonic()
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        full_env = dict(
            os.environ,
            TERM="xterm-256color",
            COLORTERM="truecolor",
            COLUMNS=str(cols),
            LINES=str(rows),
        )
        full_env.update(env or {})
        self.proc = subprocess.Popen(
            cmd, stdin=slave, stdout=slave, stderr=slave, env=full_env,
            close_fds=True, cwd=cwd,
        )
        os.close(slave)
        self.master = master

    # ── raw pty plumbing (sync; scenarios run one TUI at a time) ────────────

    def _drain(self, timeout: float) -> int:
        r, _, _ = select.select([self.master], [], [], timeout)
        if not r:
            return 0
        try:
            data = os.read(self.master, 65536)
        except OSError:
            return -1
        if not data:
            return -1
        if self.record_to is not None:
            self.events.append(
                [round(time.monotonic() - self._t0, 3), "o",
                 data.decode("utf-8", "replace")]
            )
        self.buf.extend(data)
        return len(data)

    def _pump(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._drain(0.2) < 0:
                return

    # ── Driver protocol ─────────────────────────────────────────────────────

    async def send(self, keys: str) -> None:
        os.write(self.master, keys.encode())
        await asyncio.sleep(0)

    def mark(self) -> int:
        self._drain(0)
        return len(self.buf)

    def new_text(self, since: int) -> str:
        self._drain(0)
        return _plain(bytes(self.buf[since:]))

    async def wait_marker(self, marker: str, timeout: float, since: int) -> bool:
        deadline = time.monotonic() + timeout
        needle = marker.lower()
        while time.monotonic() < deadline:
            if self._drain(0.3) < 0:
                return needle in _plain(bytes(self.buf[since:])).lower()
            if needle in _plain(bytes(self.buf[since:])).lower():
                return True
        return False

    async def settle(self, seconds: float) -> None:
        self._pump(seconds)

    def dump(self) -> str:
        self._drain(0)
        return _plain(bytes(self.buf))[-4000:]

    async def answer_permission(self, allow: bool, timeout: float = 60.0) -> bool:
        """Wait for the permission modal to render, answer it with y/n."""
        since = max(0, len(self.buf) - 65536)
        if not await self.wait_marker("y allow (for this session)", timeout, since):
            return False
        os.write(self.master, b"y" if allow else b"n")
        self._pump(0.5)
        return True

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def boot(self, marker: str = "local_harness", timeout: float = 60.0) -> bool:
        """Wait for the TUI to be up (its banner rendered)."""
        return await self.wait_marker(marker, timeout, 0)

    def close(self) -> None:
        try:
            os.write(self.master, b"\x03\x03")  # ^C ^C — the honest quit gesture
            self._pump(1.5)
        except OSError:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        try:
            os.close(self.master)
        except OSError:
            pass
        if self.record_to is not None:
            header = {
                "version": 2, "width": self.cols, "height": self.rows,
                "timestamp": int(time.time()),
                "env": {"TERM": "xterm-256color"}, "title": self.title,
            }
            with open(self.record_to, "w") as f:
                f.write(json.dumps(header) + "\n")
                for ev in self.events:
                    f.write(json.dumps(ev) + "\n")

    def __enter__(self) -> "PtyTui":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
