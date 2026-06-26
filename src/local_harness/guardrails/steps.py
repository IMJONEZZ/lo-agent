"""Step enforcement: required steps, terminal gating, prerequisites, with
escalating nudges (after forge, MIT). Stateful — one instance per run.

Adapted to this harness's loop shape: the "terminal" act is usually a plain
text answer (no tool call), so `check_finish()` gates that; `terminal_tools`
additionally gates tool-terminated workflows.
"""

from __future__ import annotations

from ..inference.types import ToolCallRequest
from .nudges import Nudge, prerequisite_nudge, step_nudge


class StepEnforcer:
    def __init__(
        self,
        required_steps: list[str] | None = None,
        terminal_tools: frozenset[str] | set[str] = frozenset(),
        prerequisites: dict[str, list[str]] | None = None,
        max_premature_attempts: int = 3,
    ):
        self.required = list(required_steps or [])
        self.terminal_tools = frozenset(terminal_tools)
        self.prerequisites = prerequisites or {}
        self.max_premature_attempts = max_premature_attempts
        self.completed: set[str] = set()
        self.premature_attempts = 0

    def pending(self) -> list[str]:
        return [s for s in self.required if s not in self.completed]

    def satisfied(self) -> bool:
        return not self.pending()

    def record(self, executed_tools: list[str]) -> None:
        self.completed.update(executed_tools)

    @property
    def exhausted(self) -> bool:
        return self.premature_attempts >= self.max_premature_attempts

    def check_tools(self, tool_calls: list[ToolCallRequest]) -> Nudge | None:
        """Gate a tool batch: premature terminal tools and prerequisites."""
        terminal = [tc for tc in tool_calls if tc.name in self.terminal_tools]
        if terminal and not self.satisfied():
            self.premature_attempts += 1
            return Nudge(
                role="user", kind="step",
                content=step_nudge(terminal[0].name, self.pending(),
                                   tier=self.premature_attempts),
            )
        for tc in tool_calls:
            missing = [p for p in self.prerequisites.get(tc.name, []) if p not in self.completed]
            if missing:
                return Nudge(
                    role="tool", kind="prerequisite", tool_call_id=tc.id,
                    content=prerequisite_nudge(tc.name, missing),
                )
        return None

    def check_finish(self) -> Nudge | None:
        """Gate a plain-text final answer when required steps remain."""
        if self.satisfied():
            return None
        self.premature_attempts += 1
        return Nudge(
            role="user", kind="step",
            content=step_nudge("a final answer", self.pending(),
                               tier=self.premature_attempts),
        )
