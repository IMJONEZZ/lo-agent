"""Self-editing memory: MEMORY.md + USER.md (the Hermes pattern).

Frozen-injected into the system prompt at run start — stable within a run, so it
preserves the prefix cache and stays inside our determinism/replay model. The
`memory` tool edits the files (add/replace/remove); there is **no auto-compaction**
— a write that would exceed a file's limit errors, forcing the agent to consolidate
rather than silently dropping. The files persist across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tools import Tool


@dataclass
class MemoryFile:
    path: Path
    title: str
    limit: int  # character cap; overflow errors instead of auto-compacting

    def read(self) -> str:
        return self.path.read_text() if self.path.exists() else ""

    def _write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text.strip() + "\n")

    def add(self, line: str) -> str:
        cur = self.read().rstrip()
        line = line.strip()
        new = (cur + "\n" + line) if cur else line
        if len(new) > self.limit:
            return (f"error: {self.title} is full ({len(new)}/{self.limit} chars). "
                    "Remove or merge entries before adding.")
        self._write(new)
        return "ok"

    def replace(self, old: str, new: str) -> str:
        cur = self.read()
        if old not in cur:
            return f"error: text not found in {self.title}: {old!r}"
        updated = cur.replace(old, new)
        if len(updated) > self.limit:
            return f"error: {self.title} would exceed its limit ({len(updated)}/{self.limit})."
        self._write(updated)
        return "ok"

    def remove(self, text: str) -> str:
        cur = self.read()
        if text not in cur:
            return f"error: text not found in {self.title}: {text!r}"
        self._write(cur.replace(text, ""))
        return "ok"


class Notebook:
    """Three scopes: USER.md (identity, prefs) · MEMORY.md (global env facts,
    conventions, lessons) · PROJECT.md (this project's specifics). USER + MEMORY
    live in the shared memory dir; PROJECT lives in the project (cwd) so it travels
    with the codebase. PROJECT is absent when no project dir is given."""

    # AGENTS.md is read-only interop (someone else's file), capped so it can't
    # blow the prefix cache. We read AGENTS.md only; CLAUDE.md is picked up ONLY
    # when an AGENTS.md explicitly imports it with an `@CLAUDE.md` reference.
    REPO_INSTRUCTION_CAP = 4000

    def __init__(self, memory_dir: str | Path, project_dir: str | Path | None = None,
                 repo_dir: str | Path | None = None):
        d = Path(memory_dir)
        self.user = MemoryFile(d / "USER.md", "USER.md", limit=1375)        # ~500 tok
        self.memory = MemoryFile(d / "MEMORY.md", "MEMORY.md", limit=2200)  # ~800 tok
        self.project: MemoryFile | None = (
            MemoryFile(Path(project_dir) / "PROJECT.md", "PROJECT.md", limit=2200)
            if project_dir else None)
        # Where to look for AGENTS.md (defaults to cwd). None disables the scope.
        self.repo_dir: Path | None = Path(repo_dir) if repo_dir else Path.cwd()

    def _file(self, target: str) -> MemoryFile | None:
        return {"memory": self.memory, "user": self.user, "project": self.project}.get(target)

    def edit(self, action: str, target: str, text: str = "", old_text: str = "") -> str:
        f = self._file(target)
        if f is None:
            return f"error: unknown target {target!r} — use 'memory', 'user', or 'project'"
        if action == "add":
            return f.add(text)
        if action == "replace":
            return f.replace(old_text, text)
        if action == "remove":
            return f.remove(text)
        return f"error: unknown action {action!r} — use add/replace/remove"

    def repo_instructions(self) -> str:
        """Read-only repo instructions: AGENTS.md walking up from repo_dir, bounded
        to the enclosing git repo (never escaping to the filesystem root), plus a
        global ~/.lo/AGENTS.md. CLAUDE.md is included only where an AGENTS.md imports
        it with an `@CLAUDE.md` reference. Nearest-first, deduped, char-capped."""
        if self.repo_dir is None:
            return ""
        seen: set[Path] = set()
        chunks: list[str] = []

        def _add(path: Path) -> str:
            try:
                rp = path.resolve()
            except OSError:
                return ""
            if rp in seen or not path.is_file():
                return ""
            seen.add(rp)
            try:
                text = path.read_text().strip()
            except OSError:
                return ""
            if text:
                chunks.append(f"# {path.name} ({path.parent})\n{text}")
            return text

        # Bound the walk to the enclosing git repo: find the nearest ancestor
        # (inclusive) with a .git and only collect AGENTS.md within it. With no
        # repo, stay at repo_dir so we don't pull in unrelated ancestor files all
        # the way up to the filesystem root.
        cur = self.repo_dir.resolve()
        root = next((d for d in [cur, *cur.parents] if (d / ".git").exists()), None)
        walk = [cur]
        while root is not None and walk[-1] != root:
            walk.append(walk[-1].parent)
        for d in walk:
            agents = d / "AGENTS.md"
            text = _add(agents)
            # Follow a CLAUDE.md reference only if AGENTS.md explicitly imports it
            # (@CLAUDE.md) — a bare mention ("ignore any CLAUDE.md") must not pull it in.
            if text and "@CLAUDE.md" in text:
                _add(d / "CLAUDE.md")

        # Global personal instructions.
        _add(Path.home() / ".lo" / "AGENTS.md")

        if not chunks:
            return ""
        joined = "\n\n".join(chunks)
        if len(joined) > self.REPO_INSTRUCTION_CAP:
            joined = joined[: self.REPO_INSTRUCTION_CAP].rstrip() + "\n… (truncated)"
        return joined

    def system_block(self) -> str:
        """The frozen snapshot injected into the system prompt at run start —
        USER (who) → MEMORY (global) → PROJECT (this codebase) → AGENTS.md (repo)."""
        parts = []
        usr, mem = self.user.read().strip(), self.memory.read().strip()
        proj = self.project.read().strip() if self.project else ""
        repo = self.repo_instructions()
        if usr:
            parts.append(f"## What you know about the user (USER.md)\n{usr}")
        if mem:
            parts.append(f"## Your durable notes (MEMORY.md)\n{mem}")
        if proj:
            parts.append(f"## This project (PROJECT.md)\n{proj}")
        if repo:
            parts.append(f"## Repository instructions (AGENTS.md)\n{repo}")
        return "\n\n".join(parts)


def memory_tool(notebook: Notebook) -> Tool:
    def memory(action: str, text: str = "", target: str = "memory", old_text: str = "") -> str:
        return notebook.edit(action, target, text=text, old_text=old_text)

    targets = ["memory", "user"] + (["project"] if notebook.project is not None else [])
    return Tool(
        name="memory",
        description=("Edit your durable memory. action=add/replace/remove; "
                     "target='memory' (global facts/conventions/lessons), 'user' (user prefs), "
                     "or 'project' (specifics of THIS codebase). "
                     "No auto-compaction — if full, remove/merge first."),
        parameters={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"]},
            "text": {"type": "string"},
            "target": {"type": "string", "enum": targets},
            "old_text": {"type": "string", "description": "for replace/remove"}},
            "required": ["action", "text"]},
        fn=memory)


def session_search_tool(memory) -> Tool:
    """memory: an agent.memory.Memory (FTS5 over past sessions/lessons)."""
    def session_search(query: str, limit: int = 5) -> str:
        hits = memory.recall(query, limit=limit)
        if not hits:
            return "no matching past sessions"
        return "\n".join(f"[{h.kind}] {h.text}" for h in hits)

    return Tool(
        name="session_search",
        description="Search your past sessions and lessons (full-text) for relevant prior context.",
        parameters={"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"]},
        fn=session_search)
