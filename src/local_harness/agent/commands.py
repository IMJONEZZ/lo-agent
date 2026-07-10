"""File-authored slash commands (the OpenCode / Claude-Code pattern).

A *command* is a saved prompt template in a markdown file; its filename becomes a
slash command (`fix.md` → `/fix`). Discovered from `.lo/commands/` (project) and
`~/.lo/commands/` (user), plus read-only `.opencode/commands/` for drop-in interop
with OpenCode command files. Project wins over user wins over `.opencode`.

Frontmatter (all optional): `description`, `agent` (preset to run under), `model`,
`subtask`. The body is the template, expanded at invocation time:

    $ARGUMENTS          the full argument string after the command
    $1 $2 … $9          positional arguments (shell-split)
    @path               embed the contents of a file (read-only, if it exists)
    !`cmd`              embed the output of a shell command (sandbox-routed)

`!`cmd`` runs through the session sandbox (never the raw host when a microVM is
active) and only when a sandbox is supplied — otherwise it expands to nothing.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from .frontmatter import split_frontmatter

_BACKTICK = re.compile(r"!`([^`]+)`")
_FILE = re.compile(r"@(\S+)")


@dataclass
class CustomCommand:
    name: str
    template: str
    description: str = ""
    agent: str | None = None
    model: str | None = None
    subtask: bool = False
    source: str = ""


def command_dirs() -> list[Path]:
    """Search order: project `.lo/` → user `~/.lo/` → project `.opencode/` interop."""
    return [
        Path(".lo/commands"),
        Path.home() / ".lo" / "commands",
        Path(".opencode/commands"),
    ]


def load_commands(dirs: list[Path] | None = None) -> dict[str, CustomCommand]:
    dirs = dirs if dirs is not None else command_dirs()
    out: dict[str, CustomCommand] = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            name = p.stem.lower()
            if name in out:  # earlier dir (higher priority) already claimed it
                continue
            try:
                meta, body = split_frontmatter(p.read_text())
            except OSError:
                continue
            out[name] = CustomCommand(
                name=name,
                template=body.strip(),
                description=str(meta.get("description") or ""),
                agent=(str(meta["agent"]) if meta.get("agent") else None),
                model=(str(meta["model"]) if meta.get("model") else None),
                subtask=bool(meta.get("subtask") or False),
                source=str(d),
            )
    return out


def _embed_files(text: str) -> str:
    def repl(m: re.Match) -> str:
        p = Path(m.group(1))
        if p.is_file():
            try:
                return f"\n```\n{p.read_text()[:8000]}\n```\n"
            except OSError:
                return m.group(0)
        return m.group(0)  # not a real file → leave the @token untouched

    return _FILE.sub(repl, text)


async def render_template(template: str, arg_string: str = "", *, sandbox=None) -> str:
    """Expand a command template. `!`cmd`` runs FIRST — on the raw, author-written
    template only — so a shell command can never be smuggled in through `$ARGUMENTS`
    or an embedded `@file`. Argument and file substitution happen afterwards; a
    positional can still supply a filename to a later `@file`."""
    text = await _expand_backticks(template, sandbox)

    try:
        args = shlex.split(arg_string) if arg_string else []
    except ValueError:  # unbalanced quote/apostrophe (e.g. "it's broken") → plain split
        args = arg_string.split()
    for i in range(9, 0, -1):  # high-to-low so $10 isn't mangled by $1
        text = text.replace(f"${i}", args[i - 1] if i - 1 < len(args) else "")
    text = text.replace("$ARGUMENTS", arg_string)
    text = _embed_files(text)
    return text.strip()


async def _expand_backticks(text: str, sandbox) -> str:
    """Run each `!`cmd`` through the sandbox (dropped when no sandbox is available)."""
    out_parts: list[str] = []
    last = 0
    for m in _BACKTICK.finditer(text):
        out_parts.append(text[last : m.start()])
        if sandbox is not None:
            try:
                res, _rc = await sandbox.exec(m.group(1), timeout=30)
                out_parts.append(res.strip())
            except Exception as e:  # a bad command must not break expansion
                out_parts.append(f"(shell error: {e})")
        last = m.end()
    out_parts.append(text[last:])
    return "".join(out_parts)
