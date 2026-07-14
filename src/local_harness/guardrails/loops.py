"""Doom-loop detection: a model that calls the SAME tool with the SAME arguments
over and over is stuck, not working. Stateful — one instance per run.

Complements the error budget in `errors.py`: that one counts consecutive tool
*failures* (any arguments); this one counts repeated *identical* calls even when
each one "succeeds" — e.g. re-reading the same file, re-running the same search,
re-submitting the same code. Fingerprint = tool name + truncated arguments.
"""

from __future__ import annotations

from ..inference.types import ToolCallRequest


class LoopDetector:
    def __init__(self, max_repeats: int = 3, hard_cap: int = 6, args_len: int = 200):
        # nudge once a fingerprint reaches `max_repeats` attempts; declare fatal at
        # `hard_cap`. Between the two the call still runs (a genuine repeat isn't
        # always a loop) — the cap is the backstop when the model ignores the nudge.
        self.max_repeats = max_repeats
        self.hard_cap = hard_cap
        self.args_len = args_len
        self.counts: dict[str, int] = {}
        self.nudged: set[str] = set()

    def _fingerprint(self, tc: ToolCallRequest) -> str:
        return f"{tc.name}:{(tc.arguments or '').strip()[: self.args_len]}"

    def inspect(
        self, tool_calls: list[ToolCallRequest]
    ) -> tuple[str, ToolCallRequest, int] | None:
        """Count this batch of proposed calls and return the first one that trips a
        threshold: ("fatal"|"nudge", offending_call, repeat_count), or None."""
        trip: tuple[str, ToolCallRequest, int] | None = None
        for tc in tool_calls:
            fp = self._fingerprint(tc)
            self.counts[fp] = self.counts.get(fp, 0) + 1
            n = self.counts[fp]
            if trip is not None:
                continue  # keep counting the rest of the batch, but one trip is enough
            if n >= self.hard_cap:
                trip = ("fatal", tc, n)
            elif n >= self.max_repeats and fp not in self.nudged:
                self.nudged.add(fp)
                trip = ("nudge", tc, n)
        return trip
