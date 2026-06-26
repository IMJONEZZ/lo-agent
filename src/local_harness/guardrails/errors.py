"""Error budgets: consecutive failures before the loop declares fatal
(after forge, MIT). Stateful — one instance per run."""

from __future__ import annotations


class ErrorTracker:
    def __init__(self, max_retries: int = 3, max_tool_errors: int = 2):
        self.max_retries = max_retries
        self.max_tool_errors = max_tool_errors
        self.consecutive_retries = 0
        self.consecutive_tool_errors = 0

    def record_retry(self) -> None:
        self.consecutive_retries += 1

    def reset_retries(self) -> None:
        self.consecutive_retries = 0

    def record_tool_error(self) -> None:
        self.consecutive_tool_errors += 1

    def reset_tool_errors(self) -> None:
        """Call after a fully clean tool batch (no errors at all)."""
        self.consecutive_tool_errors = 0

    @property
    def retries_exhausted(self) -> bool:
        return self.consecutive_retries > self.max_retries

    @property
    def tool_errors_exhausted(self) -> bool:
        return self.consecutive_tool_errors > self.max_tool_errors
