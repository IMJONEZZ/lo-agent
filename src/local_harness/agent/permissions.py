"""Tiered tool permissions — allow / ask / deny (the Claude Code model).

Rules are tool-name globs evaluated **deny → ask → allow, first match wins**; an
unmatched tool falls to `default` (ask, like Claude Code). `ask` calls an injected
approver (the TUI modal, say); approving once for the session is remembered.

NOTE: permissions are decided purely by deterministic policy. An earlier version
also escalated to `ask` when the model's mean token-logprob for a call was low.
That was removed: token-logprob is the probability of a *surface form*, not a
measure of whether an action is correct or safe, so gating permissions on it was
a category error (see docs/uncertainty-done-right.md). Confidence belongs to
verification — sample-consistency / semantic entropy — not to a single number.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Awaitable, Callable

# approver: (tool_name, arguments) -> approved? (may be sync or async)
Approver = Callable[[str, str], "bool | Awaitable[bool]"]


@dataclass
class Permissions:
    allow: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    default: str = "ask"                  # allow | ask | deny when nothing matches
    approver: Approver | None = None
    _session_allow: set[str] = field(default_factory=set)

    def decide(self, tool_name: str) -> str:
        for pat in self.deny:
            if fnmatch(tool_name, pat):
                return "deny"
        for pat in self.ask:
            if fnmatch(tool_name, pat):
                return "ask"
        for pat in self.allow:
            if fnmatch(tool_name, pat):
                return "allow"
        return self.default

    async def check(self, tool_name: str, arguments: str,
                    confidence: float | None = None) -> tuple[bool, str]:
        """Return (allowed, reason-if-denied). `confidence` is accepted and ignored
        for call-compatibility; permissions are a deterministic policy decision."""
        if tool_name in self._session_allow:
            return True, ""
        decision = self.decide(tool_name)
        if decision == "allow":
            return True, ""
        if decision == "deny":
            return False, f"{tool_name} is denied by policy"
        # ask
        if self.approver is None:
            return False, f"{tool_name} needs approval but no approver is configured"
        result = self.approver(tool_name, arguments)
        if inspect.isawaitable(result):
            result = await result
        if result:
            self._session_allow.add(tool_name)   # remember the approval for this session
            return True, ""
        return False, f"{tool_name} was not approved"


# Read-only tools run freely; anything that writes, runs shell, or hits the network asks.
def default_permissions(approver: Approver | None = None) -> Permissions:
    return Permissions(
        allow=["read_file", "list_dir", "calculator", "grep", "glob",
               "session_search", "memory"],
        default="ask",
        approver=approver,
    )
