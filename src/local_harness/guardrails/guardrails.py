"""Bundled guardrails: forge's two-method API (check / record) adapted to
this harness. One instance per run.

    result = guardrails.check(message)        # after each model response
    guardrails.record(executed, had_errors)   # after executing tools
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..inference.types import Message, ToolCallRequest
from .errors import ErrorTracker
from .loops import LoopDetector
from .nudges import Nudge, doom_loop_nudge
from .steps import StepEnforcer
from .validator import ResponseValidator


@dataclass(frozen=True)
class CheckResult:
    """action:
    "execute"  — tool_calls are safe to run (possibly rescued from text)
    "final"    — legitimate plain-text answer; required steps satisfied
    "nudge"    — inject `nudge` with its role and re-prompt
    "fatal"    — an error budget is exhausted; stop the run
    """

    action: Literal["execute", "final", "nudge", "fatal"]
    tool_calls: list[ToolCallRequest] | None = None
    nudge: Nudge | None = None
    reason: str | None = None
    rescued: bool = False


class Guardrails:
    def __init__(
        self,
        tool_names: list[str],
        required_steps: list[str] | None = None,
        terminal_tools: frozenset[str] | set[str] = frozenset(),
        prerequisites: dict[str, list[str]] | None = None,
        rescue_enabled: bool = True,
        max_retries: int = 3,
        max_tool_errors: int = 2,
        max_premature_attempts: int = 3,
        max_repeats: int = 3,
        max_loop: int = 6,
    ):
        self.validator = ResponseValidator(tool_names, rescue_enabled=rescue_enabled)
        self.steps = StepEnforcer(required_steps, terminal_tools, prerequisites,
                                  max_premature_attempts)
        self.errors = ErrorTracker(max_retries, max_tool_errors)
        self.loops = LoopDetector(max_repeats=max_repeats, hard_cap=max_loop)

    def check(self, message: Message) -> CheckResult:
        validation = self.validator.validate(message)

        if validation.nudge is not None:
            self.errors.record_retry()
            if self.errors.retries_exhausted:
                return CheckResult("fatal", reason=(
                    f"retry budget exhausted after {self.errors.consecutive_retries} "
                    f"consecutive unusable responses (last: {validation.nudge.kind})"
                ))
            return CheckResult("nudge", nudge=validation.nudge)

        self.errors.reset_retries()

        if validation.final:
            nudge = self.steps.check_finish()
            if nudge is not None:
                if self.steps.exhausted:
                    return CheckResult("fatal", reason=(
                        f"required steps still pending after "
                        f"{self.steps.premature_attempts} premature finish attempts: "
                        f"{self.steps.pending()}"
                    ))
                return CheckResult("nudge", nudge=nudge)
            return CheckResult("final")

        nudge = self.steps.check_tools(validation.tool_calls or [])
        if nudge is not None:
            if nudge.kind == "step" and self.steps.exhausted:
                return CheckResult("fatal", reason=(
                    f"terminal tool blocked {self.steps.premature_attempts} times; "
                    f"pending steps: {self.steps.pending()}"
                ))
            return CheckResult("nudge", nudge=nudge)

        loop = self.loops.inspect(validation.tool_calls or [])
        if loop is not None:
            action, tc, n = loop
            if action == "fatal":
                return CheckResult("fatal", reason=(
                    f"doom loop: {tc.name} called {n} times with identical arguments"
                ))
            # role="tool" so the repeated call's channel stays paired (like the
            # prerequisite nudge) — the model reads it as that call's result.
            return CheckResult("nudge", nudge=Nudge(
                role="tool", kind="loop", tool_call_id=tc.id,
                content=doom_loop_nudge(tc.name, n)))

        return CheckResult("execute", tool_calls=validation.tool_calls,
                           rescued=validation.rescued)

    def record(self, executed_tools: list[str], had_errors: bool) -> str | None:
        """Record a completed tool batch. Returns a fatal reason if the tool
        error budget is exhausted, else None."""
        self.steps.record(executed_tools)
        if had_errors:
            self.errors.record_tool_error()
            if self.errors.tool_errors_exhausted:
                return (
                    f"tool error budget exhausted after "
                    f"{self.errors.consecutive_tool_errors} consecutive failing batches"
                )
        else:
            self.errors.reset_tool_errors()
        return None
