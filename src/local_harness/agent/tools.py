"""Tool definitions and registry (OpenAI function-calling schema)."""

from __future__ import annotations

import ast
import inspect
import json
import operator
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

MAX_TOOL_OUTPUT = 16000  # cap any tool's output so it can't flood the context
TOOL_SEARCH_NAME = "tool_search"  # the meta-tool that surfaces deferred tools


def tool_search_schema(n_deferred: int) -> dict[str, Any]:
    """The synthesized `tool_search` tool the model sees when tools are deferred —
    a count only (no names), so the model searches by the capability it needs."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_SEARCH_NAME,
            "description": (
                f"{n_deferred} more tools are available but not loaded to save context. "
                "Search them by describing the capability you need (e.g. 'send a slack "
                "message', 'query a postgres database'); the best matches are loaded and "
                "become callable on your next step."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "what you want to do, in a few words"}},
                "required": ["query"],
            },
        },
    }


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    fn: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None, permissions=None):
        self._tools: dict[str, Tool] = {}
        self.permissions = permissions  # agent.permissions.Permissions | None
        # Tools that may be deferred behind tool_search when the total count is
        # large (set by registry_with_sources for MCP/UTCP tools). Core builtins
        # are never deferred.
        self._deferrable: set[str] = set()
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def set_deferrable(self, names) -> None:
        self._deferrable = {n for n in names if n in self._tools}

    def deferrable_names(self) -> set[str]:
        return set(self._deferrable)

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """BM25-rank the deferrable tools against `query` via an in-memory FTS5
        index over name+description (same approach as agent/memory.py). Returns
        [(name, description), …] best first."""
        names = [n for n in self._deferrable if n in self._tools]
        words = re.findall(r"[A-Za-z0-9_]+", query)
        if not names or not words:
            return []
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE VIRTUAL TABLE t USING fts5(name, description)")
            con.executemany("INSERT INTO t(name, description) VALUES (?, ?)",
                            [(n, self._tools[n].description) for n in names])
            match = " OR ".join(f'"{w}"' for w in words)
            rows = con.execute(
                "SELECT name, description FROM t WHERE t MATCH ? ORDER BY bm25(t) LIMIT ?",
                (match, limit)).fetchall()
        finally:
            con.close()
        return [(n, d) for n, d in rows]

    async def execute(self, name: str, arguments: str, confidence: float | None = None) -> str:
        """Run a tool from a raw JSON arguments string; errors become strings
        so the model sees them and can recover. Tool fns may be sync or async
        (MCP/UTCP/webfetch/sandboxed tools are async) — an awaitable result is
        awaited. `confidence` is accepted for call-compatibility and ignored."""
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        if self.permissions is not None:
            ok, reason = await self.permissions.check(name, arguments, confidence)
            if not ok:
                return f"error: {reason}"
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return f"error: invalid JSON arguments: {e}"
        try:
            result = tool.fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as e:  # noqa: BLE001 — tool failures are model feedback
            return f"error: {type(e).__name__}: {e}"


# --- built-in tools ---------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Arithmetic over +-*/%//** and parentheses, no names or calls."""
    return str(_safe_eval(ast.parse(expression, mode="eval")))


def _slice_lines(text: str, start_line: int, end_line: int) -> str:
    """1-indexed inclusive line range. 0/None on either bound means open-ended;
    both unset returns the text unchanged (whole-file behavior)."""
    if not start_line and not end_line:
        return text
    lines = text.splitlines(keepends=True)
    lo = max(1, start_line or 1)
    hi = end_line or len(lines)
    return "".join(lines[lo - 1:hi])


def read_file(path: str, start_line: int = 0, end_line: int = 0,
              max_bytes: int = 65536) -> str:
    # With a line range, read the whole file then slice (so the range is reachable
    # even past max_bytes) and bound the result; otherwise cap the raw read.
    if start_line or end_line:
        return _truncate_output(_slice_lines(Path(path).read_text(), start_line, end_line))
    return Path(path).read_text()[:max_bytes]


def list_dir(path: str = ".") -> str:
    return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in Path(path).iterdir()))


def _truncate_output(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    head = MAX_TOOL_OUTPUT - 60
    return text[:head] + f"\n… [truncated, {len(text) - head} more chars]"


def bash(command: str, timeout: int = 30) -> str:
    """Run a shell command via `bash -c`, returning combined stdout+stderr.

    A non-zero exit is prefixed with `[exit N]` so the model can see failures;
    output is capped and the call times out rather than hanging."""
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        out = f"[exit {proc.returncode}]\n{out}".strip()
    return _truncate_output(out) if out else "[no output]"


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", html).strip()


async def webfetch(url: str, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """Fetch a URL and return its readable text (HTML stripped), bounded."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, transport=transport) as c:
        resp = await c.get(url, headers={"User-Agent": "local_harness/0.1"})
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        text = _html_to_text(resp.text) if "html" in ctype or resp.text.lstrip().startswith("<") \
            else resp.text
    return _truncate_output(text)


async def _wikipedia_search(query: str, transport, limit: int = 6) -> str:
    """Keyless default search: Wikipedia's REST search API. Reliable and
    bot-friendly (unlike scraping a general engine), and good enough to find an
    article the agent can then `webfetch` in full. Returns a clean ranked list of
    title · URL · excerpt."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, transport=transport) as c:
        resp = await c.get(
            "https://en.wikipedia.org/w/rest.php/v1/search/page",
            params={"q": query, "limit": limit},
            headers={"User-Agent": "local_harness/0.1 (agent research)"})
        resp.raise_for_status()
        pages = resp.json().get("pages", [])
    if not pages:
        return f"no Wikipedia results for {query!r}. For broader web search, set LO_SEARCH_URL."
    lines = [f"web_search results for {query!r} (source: Wikipedia; set LO_SEARCH_URL "
             "for a general web provider):"]
    for p in pages:
        url = f"https://en.wikipedia.org/wiki/{p.get('key', '')}"
        excerpt = _html_to_text(p.get("excerpt") or p.get("description") or "")
        lines.append(f"- {p.get('title')} — {url}\n  {excerpt}")
    return _truncate_output("\n".join(lines))


async def web_search(query: str, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """Search the web. Defaults to a keyless Wikipedia search; point at any other
    provider by setting $LO_SEARCH_URL (it receives ?q=<query>). Follow up by
    calling webfetch on a result URL to read the full page."""
    url = os.environ.get("LO_SEARCH_URL")
    if not url:
        return await _wikipedia_search(query, transport)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, transport=transport) as c:
        resp = await c.get(url, params={"q": query}, headers={"User-Agent": "local_harness/0.1"})
        resp.raise_for_status()
    return _truncate_output(resp.text)


def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace `old_string` with `new_string`. `old_string` must occur exactly
    once — otherwise the edit errors (add surrounding context to disambiguate)."""
    p = Path(path)
    if not p.exists():
        return f"error: no such file: {path}"
    text = p.read_text()
    n = text.count(old_string)
    if n == 0:
        return f"error: old_string not found in {path}"
    if n > 1:
        return f"error: old_string is not unique in {path} ({n} matches) — add context"
    p.write_text(text.replace(old_string, new_string))
    return f"edited {path}"


def _parse_patch(patch: str) -> tuple[str | None, list[tuple[list[str], list[str]]]]:
    """Parse a unified diff into (path-from-+++-header, [(before, after), ...]).
    before/after are the removed+context and added+context line lists per hunk."""
    path = None
    hunks: list[tuple[list[str], list[str]]] = []
    before: list[str] | None = None
    after: list[str] | None = None
    for ln in patch.splitlines():
        if ln.startswith("--- "):
            continue
        if ln.startswith("+++ "):
            p = ln[4:].strip().split("\t")[0]
            path = p[2:] if p.startswith(("a/", "b/")) else p
            continue
        if ln.startswith("@@"):
            if before is not None:
                hunks.append((before, after))
            before, after = [], []
            continue
        if before is None:
            continue  # preamble before the first hunk
        if ln.startswith("\\"):
            continue  # "\ No newline at end of file"
        tag, content = (ln[0], ln[1:]) if ln[:1] in " +-" else (" ", ln)
        if tag == " ":
            before.append(content)
            after.append(content)
        elif tag == "-":
            before.append(content)
        elif tag == "+":
            after.append(content)
    if before is not None:
        hunks.append((before, after))
    return path, hunks


def _find_block(lines: list[str], block: list[str], start: int) -> int:
    """Index of the first contiguous occurrence of `block` in `lines` at/after
    `start`, or -1. Empty block never matches."""
    if not block:
        return -1
    for i in range(start, len(lines) - len(block) + 1):
        if lines[i:i + len(block)] == block:
            return i
    return -1


def _apply_unified_diff(text: str, patch: str) -> tuple[str | None, str | None]:
    """Apply a unified diff by locating each hunk's context+removed block by
    CONTENT (robust to line-number drift) and swapping in its context+added
    block. Returns (new_text, None) or (None, error). Never partially writes."""
    _, hunks = _parse_patch(patch)
    if not hunks:
        return None, "no hunks found in patch (expected @@ ... @@ sections)"
    lines = text.splitlines()
    cursor = 0
    for before, after in hunks:
        if not before:
            return None, "a hunk has no context or removed lines to locate; add context lines"
        idx = _find_block(lines, before, cursor)
        if idx < 0:
            idx = _find_block(lines, before, 0)  # hunks may be out of order
        if idx < 0:
            return None, f"hunk did not apply — context not found near: {before[0][:60]!r}"
        lines[idx:idx + len(before)] = after
        cursor = idx + len(after)
    trailing = "\n" if text.endswith("\n") or not text else ""
    return "\n".join(lines) + trailing, None


def apply_patch(patch: str, path: str = "") -> str:
    """Apply a unified-diff `patch` to a file. `path` is optional if the patch has
    a `+++ b/<path>` header. Hunks are located by content, so line numbers need not
    be exact; if any hunk fails to match, nothing is written and an error explains."""
    target = path or _parse_patch(patch)[0]
    if not target:
        return "error: no path given and no +++ header in patch"
    p = Path(target)
    if not p.exists():
        return f"error: no such file: {target}"
    new_text, err = _apply_unified_diff(p.read_text(), patch)
    if err is not None:
        return f"error: {err}"
    p.write_text(new_text)
    return f"patched {target}"


_GREP_IGNORE = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
_IGNORE_FILES = (".gitignore", ".ignore")


def _ignore_rule(pat: str) -> tuple[re.Pattern, bool, bool] | None:
    """Compile one gitignore line to (regex, negate, dir_only). Returns None for
    blanks/comments. Regex matches a POSIX path relative to the ignore file's root.
    Unanchored names match at any depth; a leading/internal slash anchors to root."""
    pat = pat.rstrip()
    if not pat or pat.startswith("#"):
        return None
    negate = pat.startswith("!")
    if negate:
        pat = pat[1:]
    pat = pat.replace("\\ ", " ")
    dir_only = pat.endswith("/")
    pat = pat.rstrip("/")
    anchored = pat.startswith("/") or ("/" in pat)
    pat = pat.lstrip("/")
    if not pat:
        return None
    # translate the glob body segment-by-segment (git globs don't cross '/' on *)
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if pat[i:i + 3] == "**/":
            out.append("(?:.*/)?"); i += 3
        elif pat[i:i + 2] == "**":
            out.append(".*"); i += 2
        elif c == "*":
            out.append("[^/]*"); i += 1
        elif c == "?":
            out.append("[^/]"); i += 1
        else:
            out.append(re.escape(c)); i += 1
    body = "".join(out)
    prefix = "^" if anchored else "(?:^|.*/)"
    # match the path itself and anything beneath it (so a dir rule ignores its files)
    regex = re.compile(prefix + body + r"(?:/.*)?$")
    return regex, negate, dir_only


def _load_ignorer(root: Path):
    """Build an `is_ignored(relpath)` predicate from root-level ignore files
    (.gitignore, .ignore, .git/info/exclude). Root-level only — nested ignore
    files are not honored (documented limitation). Last matching rule wins, so
    `!negations` work."""
    rules: list[tuple[re.Pattern, bool, bool]] = []
    sources = [root / f for f in _IGNORE_FILES] + [root / ".git" / "info" / "exclude"]
    for src in sources:
        try:
            for line in src.read_text(errors="ignore").splitlines():
                rule = _ignore_rule(line)
                if rule is not None:
                    rules.append(rule)
        except OSError:
            continue
    if not rules:
        return None

    def is_ignored(relpath: str) -> bool:
        ignored = False
        for regex, negate, _dir_only in rules:
            if regex.match(relpath):
                ignored = not negate
        return ignored

    return is_ignored


def _walk_files(root: Path):
    """Yield files under root, pruning the always-ignore dirs and anything the
    root ignore files (.gitignore/.ignore) match."""
    is_ignored = _load_ignorer(root)
    for f in root.rglob("*"):
        if not f.is_file() or (_GREP_IGNORE & set(f.parts)):
            continue
        if is_ignored is not None:
            try:
                rel = f.relative_to(root).as_posix()
            except ValueError:
                rel = f.as_posix()
            if is_ignored(rel):
                continue
        yield f


def grep(pattern: str, path: str = ".", max_results: int = 200) -> str:
    """Search file contents for a regex; returns `path:line:text`, ripgrep-style.
    Skips the default ignore dirs and paths matched by root .gitignore/.ignore."""
    rx = re.compile(pattern)
    root = Path(path)
    files = [root] if root.is_file() else _walk_files(root)
    out: list[str] = []
    for f in files:
        try:
            for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{f}:{i}:{line.strip()}")
                    if len(out) >= max_results:
                        return _truncate_output("\n".join(out))
        except (OSError, ValueError):
            continue
    return _truncate_output("\n".join(out)) if out else "no matches"


def glob(pattern: str, path: str = ".") -> str:
    """Find files matching a glob (supports ** for recursive). Skips the default
    ignore dirs and paths matched by root .gitignore/.ignore."""
    root = Path(path)
    is_ignored = _load_ignorer(root)
    matches = []
    for p in root.glob(pattern):
        if not p.is_file() or (_GREP_IGNORE & set(p.parts)):
            continue
        if is_ignored is not None:
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = p.as_posix()
            if is_ignored(rel):
                continue
        matches.append(str(p))
    matches.sort()
    return _truncate_output("\n".join(matches)) if matches else "no matches"


# --- NotebookEdit: edit Jupyter .ipynb cells without corrupting the JSON ------

def _apply_notebook_edit(text: str, cell_index: int, source: str,
                         cell_type: str, action: str) -> tuple[str, str]:
    nb = json.loads(text)
    cells = nb.setdefault("cells", [])
    i = int(cell_index)

    def _new_cell() -> dict:
        c: dict[str, Any] = {"cell_type": cell_type, "metadata": {},
                             "source": source.splitlines(keepends=True)}
        if cell_type == "code":
            c["outputs"] = []
            c["execution_count"] = None
        return c

    if action == "delete":
        if not 0 <= i < len(cells):
            raise IndexError(f"cell {i} out of range (0..{len(cells) - 1})")
        cells.pop(i)
    elif action == "insert":
        i = max(0, min(i, len(cells)))
        cells.insert(i, _new_cell())
    elif action == "replace":
        if not 0 <= i < len(cells):
            raise IndexError(f"cell {i} out of range (0..{len(cells) - 1})")
        old, new = cells[i], _new_cell()
        if old.get("cell_type") == "code" and cell_type == "code":  # keep prior outputs
            new["outputs"] = old.get("outputs", [])
            new["execution_count"] = old.get("execution_count")
        cells[i] = new
    else:
        raise ValueError(f"unknown action {action!r} — use replace/insert/delete")
    return json.dumps(nb, indent=1) + "\n", f"{action}d cell {i} in the notebook ({len(cells)} cells)"


def notebook_edit(path: str, cell_index: int = 0, source: str = "",
                  cell_type: str = "code", action: str = "replace") -> str:
    p = Path(path)
    if not p.exists():
        return f"error: no such notebook: {path}"
    try:
        new_text, msg = _apply_notebook_edit(p.read_text(), cell_index, source, cell_type, action)
    except (ValueError, IndexError) as e:
        return f"error: {e}"
    p.write_text(new_text)
    return msg


# --- REPL: a persistent Python session (state across calls) -------------------

class _ReplSessions:
    """Per-session persistent namespaces, in-process. A code block is exec'd in
    the session's namespace (so variables/imports carry across calls); a trailing
    bare expression has its value echoed, REPL-style. State lives for the harness
    process (re-run cells to rebuild after a restart)."""

    def __init__(self) -> None:
        self._ns: dict[str, dict] = {}

    def run(self, code: str, session: str = "default", reset: bool = False) -> str:
        import contextlib
        import io as _io
        import traceback
        if reset or session not in self._ns:
            self._ns[session] = {"__name__": "__repl__"}
        ns = self._ns[session]
        buf = _io.StringIO()
        try:
            tree = ast.parse(code)
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(tree, "<repl>", "exec"), ns)  # noqa: S102 — local dev REPL
                if last_expr is not None:
                    value = eval(compile(last_expr, "<repl>", "eval"), ns)  # noqa: S307
                    if value is not None:
                        buf.write(repr(value) + "\n")
        except Exception:
            buf.write(traceback.format_exc())
        out = buf.getvalue()
        return _truncate_output(out) if out.strip() else "[no output]"


_REPL = _ReplSessions()


def repl(code: str, session: str = "default", reset: bool = False) -> str:
    return _REPL.run(code, session=session, reset=reset)


def builtin_tools(sandbox=None) -> list[Tool]:
    # When a non-host sandbox is supplied, the whole dangerous surface — bash AND
    # the file tools — runs INSIDE it (e.g. a microVM), confined to the workdir.
    # Host default is unchanged.
    bash_fn, read_file_fn, write_file_fn = bash, read_file, write_file
    edit_file_fn, list_dir_fn, grep_fn, glob_fn = edit_file, list_dir, grep, glob
    apply_patch_fn = apply_patch
    notebook_edit_fn, repl_fn = notebook_edit, repl
    if sandbox is not None and getattr(sandbox, "kind", "host") != "host":
        import shlex

        async def notebook_edit_fn(path: str, cell_index: int = 0, source: str = "",  # noqa: F811
                                   cell_type: str = "code", action: str = "replace") -> str:
            try:
                text = await sandbox.read_file(path)
            except Exception as e:
                return f"error: {e}"
            try:
                new_text, msg = _apply_notebook_edit(text, cell_index, source, cell_type, action)
            except (ValueError, IndexError) as e:
                return f"error: {e}"
            await sandbox.write_file(path, new_text)
            return msg

        def repl_fn(code: str, session: str = "default", reset: bool = False) -> str:  # noqa: F811
            # the REPL execs in-process and cannot be confined to the microVM —
            # fall back to bash (which runs inside the sandbox) for isolated code.
            return ("error: repl is disabled under the microVM sandbox (it runs in-process). "
                    "Use bash to run code inside the sandbox instead.")

        async def bash_fn(command: str, timeout: int = 30) -> str:  # noqa: F811
            out, code = await sandbox.exec(command, timeout)
            out = (out or "").strip()
            if code != 0:
                out = f"[exit {code}]\n{out}".strip()
            return _truncate_output(out) if out else "[no output]"

        async def read_file_fn(path: str, start_line: int = 0, end_line: int = 0,  # noqa: F811
                               max_bytes: int = 65536) -> str:
            try:
                if start_line or end_line:  # read enough to reach the range, then slice
                    text = await sandbox.read_file(path, max(max_bytes, 1 << 21))
                    return _truncate_output(_slice_lines(text, start_line, end_line))
                return _truncate_output(await sandbox.read_file(path, max_bytes))
            except Exception as e:
                return f"error: {e}"

        async def write_file_fn(path: str, content: str) -> str:  # noqa: F811
            try:
                return await sandbox.write_file(path, content)
            except Exception as e:
                return f"error: {e}"

        async def list_dir_fn(path: str = ".") -> str:  # noqa: F811
            try:
                return await sandbox.list_dir(path)
            except Exception as e:
                return f"error: {e}"

        async def edit_file_fn(path: str, old_string: str, new_string: str) -> str:  # noqa: F811
            try:
                content = await sandbox.read_file(path)
            except Exception as e:
                return f"error: {e}"
            n = content.count(old_string)
            if n == 0:
                return "error: old_string not found"
            if n > 1:
                return f"error: old_string is not unique ({n} matches)"
            await sandbox.write_file(path, content.replace(old_string, new_string))
            return f"edited {path}"

        async def apply_patch_fn(patch: str, path: str = "") -> str:  # noqa: F811
            target = path or _parse_patch(patch)[0]
            if not target:
                return "error: no path given and no +++ header in patch"
            try:
                text = await sandbox.read_file(target)
            except Exception as e:
                return f"error: {e}"
            new_text, err = _apply_unified_diff(text, patch)
            if err is not None:
                return f"error: {err}"
            await sandbox.write_file(target, new_text)
            return f"patched {target}"

        _excl = " ".join(f"--exclude-dir={shlex.quote(d)}" for d in _GREP_IGNORE)
        _prune = " -o ".join(f"-name {shlex.quote(d)}" for d in _GREP_IGNORE)

        async def grep_fn(pattern: str, path: str = ".", max_results: int = 200) -> str:  # noqa: F811
            out, _ = await sandbox.exec(
                f"grep -rnI {_excl} -- {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null "
                f"| head -n {int(max_results)}")
            return out.strip() or "[no matches]"

        async def glob_fn(pattern: str, path: str = ".") -> str:  # noqa: F811
            # prune the same ignore dirs the host glob skips, then match by name
            out, _ = await sandbox.exec(
                f"find {shlex.quote(path)} \\( {_prune} \\) -prune -o "
                f"-name {shlex.quote(pattern)} -print 2>/dev/null | head -n 200")
            return out.strip() or "[no matches]"
    return [
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression (numbers, + - * / % // ** and parentheses).",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            fn=calculator,
        ),
        Tool(
            name="read_file",
            description=("Read a text file. Optional start_line/end_line (1-indexed, "
                         "inclusive) return just that range."),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer",
                                   "description": "first line to read (1-indexed); omit for start of file"},
                    "end_line": {"type": "integer",
                                 "description": "last line to read (inclusive); omit for end of file"},
                },
                "required": ["path"],
            },
            fn=read_file_fn,
        ),
        Tool(
            name="list_dir",
            description="List the entries of a directory.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
            fn=list_dir_fn,
        ),
        Tool(
            name="apply_patch",
            description=("Apply a unified diff (git-style @@ hunks) to a file. Hunks "
                         "are matched by content, so line numbers need not be exact. "
                         "Path comes from the +++ header unless you pass `path`. All "
                         "or nothing: if a hunk fails to match, nothing is written."),
            parameters={
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": "the unified diff text"},
                    "path": {"type": "string",
                             "description": "target file; optional if the patch has a +++ header"},
                },
                "required": ["patch"],
            },
            fn=apply_patch_fn,
        ),
        Tool(
            name="bash",
            description="Run a shell command and return its combined stdout/stderr.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "seconds (default 30)"},
                },
                "required": ["command"],
            },
            fn=bash_fn,
        ),
        Tool(
            name="webfetch",
            description="Fetch a URL and return its readable text content.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            fn=webfetch,
        ),
        Tool(
            name="web_search",
            description="Search the web for a query and return results.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            fn=web_search,
        ),
        Tool(
            name="write_file",
            description="Write (create or overwrite) a file with the given content.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            fn=write_file_fn,
        ),
        Tool(
            name="edit_file",
            description="Replace a unique old_string with new_string in a file.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"},
                               "old_string": {"type": "string"},
                               "new_string": {"type": "string"}},
                "required": ["path", "old_string", "new_string"],
            },
            fn=edit_file_fn,
        ),
        Tool(
            name="grep",
            description="Search file contents for a regex pattern (path:line:text).",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"],
            },
            fn=grep_fn,
        ),
        Tool(
            name="glob",
            description="Find files matching a glob pattern (** for recursive).",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"],
            },
            fn=glob_fn,
        ),
        Tool(
            name="notebook_edit",
            description="Edit a Jupyter .ipynb cell without corrupting the JSON. "
                        "action=replace|insert|delete; cell_index is 0-based; "
                        "cell_type=code|markdown (outputs are preserved on replace).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "cell_index": {"type": "integer"},
                    "source": {"type": "string", "description": "new cell content (replace/insert)"},
                    "cell_type": {"type": "string", "enum": ["code", "markdown"]},
                    "action": {"type": "string", "enum": ["replace", "insert", "delete"]},
                },
                "required": ["path", "cell_index", "action"],
            },
            fn=notebook_edit_fn,
        ),
        Tool(
            name="repl",
            description="Run Python in a persistent session — variables, imports and "
                        "state carry across calls (great for iterative data work). A "
                        "trailing bare expression echoes its value. session names an "
                        "interpreter; reset=true clears it.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "session": {"type": "string", "description": "interpreter name (default 'default')"},
                    "reset": {"type": "boolean", "description": "clear the session first"},
                },
                "required": ["code"],
            },
            fn=repl_fn,
        ),
    ]
