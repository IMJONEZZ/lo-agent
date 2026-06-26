"""Agent presets (Opencode's model): a named agent = system prompt + sampling +
tool-permission profile + exposed toolset.

- build   — the default: full toolset; writes/shell/web ask, read-only runs free.
- plan    — investigate and produce a plan; read/search only (edits denied), precise.
- explore — read and search to understand; nothing else; denied by default.
- general — like build, a bit more exploratory sampling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..inference.types import SamplingParams
from .permissions import Permissions

_READ_TOOLS = ["read_file", "list_dir", "grep", "glob", "session_search", "calculator"]

_BUILD_PROMPT = (
    "You are a precise coding agent. Use the available tools to complete the task — "
    "read and search before editing, make minimal changes, and verify. When the task "
    "is complete, reply with the final answer and no tool calls."
)
_PLAN_PROMPT = (
    "You are in PLAN mode. Investigate the codebase by reading and searching, then "
    "produce a clear, concrete plan. Do NOT modify files or run state-changing "
    "commands — you have read and search tools only."
)
_EXPLORE_PROMPT = (
    "You are in EXPLORE mode. Read and search to understand the codebase and answer "
    "the question. Do not modify anything."
)
_REVIEW_PROMPT = (
    "You are a senior code reviewer. Review the provided diff for correctness, clarity, "
    "simplicity, and bugs. Use your read-only tools to read surrounding files for context. "
    "Report findings concisely, one per line, as:\n"
    "  <file>:<line> — <blocker|major|minor|nit> — <the issue and a suggested fix>\n"
    "Group nothing; just list them, most severe first. If the change looks good, say so "
    "plainly. Do NOT modify any files."
)
_SECURITY_PROMPT = (
    "You are a security reviewer. Review the provided diff for vulnerabilities: injection "
    "(SQL / command / path), secrets or credentials committed in code, missing "
    "authentication/authorization, unsafe deserialization or eval, SSRF, XSS, and unsafe "
    "handling of untrusted input. Use your read-only tools to read surrounding files for "
    "context. Report each finding as:\n"
    "  <file>:<line> — <critical|high|medium|low> — <the vulnerability and how to fix it>\n"
    "Most severe first. If you find nothing concerning, say so. Do NOT modify any files."
)


@dataclass
class AgentPreset:
    name: str
    system_prompt: str
    sampling: SamplingParams
    allow: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    default: str = "ask"
    tools: list[str] | None = None  # exposed toolset (None = all)

    def permissions(self, approver=None) -> Permissions:
        return Permissions(allow=list(self.allow), ask=list(self.ask), deny=list(self.deny),
                           default=self.default, approver=approver)

    def exposed(self) -> set[str] | None:
        return set(self.tools) if self.tools is not None else None


# No token budget by default: locally, output tokens are free, and a cap only ever
# truncates real output (a file body cut mid-JSON, a reasoning trace cut mid-thought).
# Let the server decide. A user who wants a cap sets HARNESS_MAX_TOKENS.
_CAP = os.environ.get("HARNESS_MAX_TOKENS")
_MAX = int(_CAP) if _CAP else None

PRESETS: dict[str, AgentPreset] = {
    "build": AgentPreset(
        "build", _BUILD_PROMPT, SamplingParams(temperature=0.2, max_tokens=_MAX),
        allow=_READ_TOOLS + ["memory"], default="ask"),
    "plan": AgentPreset(
        "plan", _PLAN_PROMPT, SamplingParams(temperature=0.1, max_tokens=_MAX),
        allow=_READ_TOOLS, deny=["write_file", "edit_file", "bash", "webfetch", "web_search"],
        default="deny", tools=_READ_TOOLS),
    "explore": AgentPreset(
        "explore", _EXPLORE_PROMPT, SamplingParams(temperature=0.2, max_tokens=_MAX),
        allow=_READ_TOOLS, default="deny", tools=_READ_TOOLS),
    "review": AgentPreset(
        "review", _REVIEW_PROMPT, SamplingParams(temperature=0.1, max_tokens=_MAX),
        allow=_READ_TOOLS, default="deny", tools=_READ_TOOLS),
    "security-review": AgentPreset(
        "security-review", _SECURITY_PROMPT, SamplingParams(temperature=0.1, max_tokens=_MAX),
        allow=_READ_TOOLS, default="deny", tools=_READ_TOOLS),
    "general": AgentPreset(
        "general", _BUILD_PROMPT, SamplingParams(temperature=0.3, max_tokens=_MAX),
        allow=_READ_TOOLS + ["memory"], default="ask"),
}


def get_preset(name: str) -> AgentPreset:
    return PRESETS.get((name or "build").lower(), PRESETS["build"])
