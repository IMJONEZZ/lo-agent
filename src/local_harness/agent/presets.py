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
from pathlib import Path

from ..inference.types import SamplingParams
from .frontmatter import split_frontmatter
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
    model: str | None = None  # per-agent model (recorded; switching not yet wired)

    def permissions(self, approver=None) -> Permissions:
        return Permissions(allow=list(self.allow), ask=list(self.ask), deny=list(self.deny),
                           default=self.default, approver=approver)

    def exposed(self) -> set[str] | None:
        return set(self.tools) if self.tools is not None else None


# No token budget by default: locally, output tokens are free, and a cap only ever
# truncates real output (a file body cut mid-JSON, a reasoning trace cut mid-thought).
# Let the server decide. A user who wants a cap sets LO_MAX_TOKENS.
_CAP = os.environ.get("LO_MAX_TOKENS")
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


# --- file-authored agents (the OpenCode pattern) -------------------------------
# A markdown file per agent; filename → preset name. Discovered from `.lo/agents/`
# (project), `~/.lo/agents/` (user), and read-only `.opencode/agents/` interop.
# Frontmatter → permissions/tools/sampling; body (or `prompt`) → system prompt.

_FILE_PRESETS: dict[str, AgentPreset] = {}
_PLAN_DENY = ["write_file", "edit_file", "bash", "webfetch", "web_search"]

# The trusted read-only safety presets: a file-authored agent must never be able
# to shadow these (an untrusted repo's `.lo/agents/plan.md` claiming write tools
# would silently defeat plan/review mode), so we skip these names at load time.
_RESERVED = frozenset({"plan", "explore", "review", "security-review"})

# OpenCode uses shorter tool names than lo; translate them so an imported
# `.opencode/agents` file (`tools: {write: true, bash: true}`) maps onto real tools.
_OPENCODE_TOOLS = {
    "write": "write_file",
    "edit": "edit_file",
    "read": "read_file",
    "list": "list_dir",
    "websearch": "web_search",
}


def agent_dirs() -> list[Path]:
    return [Path(".lo/agents"), Path.home() / ".lo" / "agents", Path(".opencode/agents")]


def _aslist(meta: dict, key: str) -> list[str]:
    v = meta.get(key)
    if v is None:
        return []
    if isinstance(v, dict):  # OpenCode's `tools: {write: true, edit: false}` map
        return [str(k) for k, on in v.items() if on]
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def _tool_name(name: str) -> str:
    n = name.strip()
    return _OPENCODE_TOOLS.get(n.lower(), n)


def _tools(meta: dict, key: str) -> list[str] | None:
    """Translated tool-name list, or None if the key is absent — so callers can
    tell an explicit empty (`tools: []` → expose nothing) from "not set" (→ default)."""
    if meta.get(key) is None:
        return None
    return [_tool_name(t) for t in _aslist(meta, key)]


def _float_or(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _preset_from_meta(name: str, meta: dict, body: str) -> AgentPreset:
    mode = str(meta.get("mode") or "build").lower()
    read_only = mode in ("plan", "explore", "review", "security-review")

    prompt = meta.get("prompt")
    if isinstance(prompt, str) and prompt.startswith("{file:") and prompt.endswith("}"):
        try:
            prompt = Path(prompt[len("{file:") : -1].strip()).read_text()
        except OSError:
            prompt = ""
    if not prompt:
        prompt = body.strip() or (_PLAN_PROMPT if read_only else _BUILD_PROMPT)

    # A non-numeric temperature falls back to the mode default rather than raising
    # (a single bad file must not blow away every other file-authored agent).
    sampling = SamplingParams(
        temperature=_float_or(meta.get("temperature"), 0.1 if read_only else 0.2),
        max_tokens=_MAX,
    )
    allow = _tools(meta, "allow")
    if allow is None:
        allow = list(_READ_TOOLS) if read_only else _READ_TOOLS + ["memory"]
    deny = _tools(meta, "deny")
    if deny is None:
        deny = _PLAN_DENY if mode == "plan" else []
    tools = _tools(meta, "tools")  # explicit `tools: []` → expose nothing; absent → default
    if tools is None:
        tools = list(_READ_TOOLS) if read_only else None
    default = str(meta.get("default") or ("deny" if read_only else "ask"))
    return AgentPreset(
        name, str(prompt), sampling,
        allow=allow, ask=(_tools(meta, "ask") or []), deny=deny,
        default=default, tools=tools,
        model=(str(meta["model"]) if meta.get("model") else None),
    )


def load_file_presets(dirs: list[Path] | None = None) -> dict[str, AgentPreset]:
    dirs = dirs if dirs is not None else agent_dirs()
    out: dict[str, AgentPreset] = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            name = p.stem.lower()
            if name in _RESERVED:  # never let a file shadow a trusted read-only preset
                continue
            if name in out:  # higher-priority dir already claimed the name
                continue
            try:
                meta, body = split_frontmatter(p.read_text())
            except OSError:
                continue
            out[name] = _preset_from_meta(name, meta, body)
    return out


def register_file_presets(dirs: list[Path] | None = None) -> dict[str, AgentPreset]:
    """(Re)load file-authored agents into the process-level cache so `get_preset`
    resolves them by name — including server-side, where presets arrive by name."""
    global _FILE_PRESETS
    try:
        _FILE_PRESETS = load_file_presets(dirs)
    except Exception:
        _FILE_PRESETS = {}
    return _FILE_PRESETS


def all_preset_names() -> list[str]:
    return sorted(set(PRESETS) | set(_FILE_PRESETS))


def get_preset(name: str) -> AgentPreset:
    key = (name or "build").lower()
    if key in _FILE_PRESETS:  # a file-authored agent overrides / adds to built-ins
        return _FILE_PRESETS[key]
    return PRESETS.get(key, PRESETS["build"])
