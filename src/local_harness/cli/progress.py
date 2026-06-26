"""A Claude-Code-style compaction progress bar for the terminal.

Drives off the agent's `on_compact(phase, info)` callback. Renders a single
in-place line that fills while the summary call is in flight and, on completion,
reports the token reduction — mirroring the "Compacting conversation…" indicator
Claude Code shows when its auto-compact pass runs.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

_FILL = "█"
_EMPTY = "░"


def _human(n: int | None) -> str:
    if not n:
        return "0"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


class CompactionProgressBar:
    """Callable matching the agent's `on_compact` signature."""

    def __init__(self, stream: TextIO | None = None, width: int = 24):
        self.stream = stream or sys.stderr
        self.width = width
        self._active = False

    def __call__(self, phase: str, info: dict[str, Any]) -> None:
        if not self.stream.isatty() and phase != "done":
            return  # no live redraw when piped; still print the final summary line
        if phase == "start":
            self._active = True
            self._draw(0.0, info)
        elif phase == "tick":
            if self._active:
                self._draw(float(info.get("frac", 0.0)), info)
        elif phase == "done":
            self._finish(info)
            self._active = False

    def _draw(self, frac: float, info: dict[str, Any]) -> None:
        frac = max(0.0, min(1.0, frac))
        filled = int(round(frac * self.width))
        bar = _FILL * filled + _EMPTY * (self.width - filled)
        cw = info.get("context_window")
        ctx = f" · {_human(info.get('trigger_tokens'))}/{_human(cw)} ctx" if cw else ""
        self.stream.write(
            f"\r\033[2m⛁ Compacting conversation\033[0m  {bar}  {int(frac * 100):3d}%"
            f"\033[2m{ctx}\033[0m   "
        )
        self.stream.flush()

    def _finish(self, info: dict[str, Any]) -> None:
        before = info.get("before_tokens") or 0
        after = info.get("after_tokens") or 0
        pct = int(round((1 - after / before) * 100)) if before else 0
        method = info.get("method", "summarize")
        tag = "summarized" if method == "summarize" else f"mechanical·phase {info.get('phase', '?')}"
        if self.stream.isatty():
            self.stream.write("\r\033[2K")  # clear the in-place bar line
        self.stream.write(
            f"\033[2m⛁ Compacted ({tag}): {_human(before)} → {_human(after)} tokens "
            f"(−{pct}%)\033[0m\n"
        )
        self.stream.flush()
