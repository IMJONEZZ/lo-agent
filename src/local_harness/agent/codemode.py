"""Code-mode: one `run_code` tool the model writes Python against, instead of N
per-tool schemas. The model chains many tools in a single code block (one
round-trip, far fewer tokens) — playing to models' code strength. MCP/UTCP tools
stay hookable: they're already in the registry, just called from code.

Execution follows the sandbox:
  host (default)  → restricted in-process exec (only the tools + safe builtins).
  microvm         → the code runs INSIDE the libkrun VM (hard isolation); tool
                    calls bridge back to the host over the shared workdir mount,
                    so the model's code can't touch the host except through tools.

The registry's own permission/exposed-tool policy is enforced on every call, so
code-mode can't reach a tool a preset (plan/explore) hides.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import textwrap
import traceback
from typing import Any

from .tools import ToolRegistry

RUN_CODE_NAME = "run_code"

# Pure-computation stdlib the model may import in-process (no I/O, no exec).
# Models reflexively write `import re` / `import math` — refusing these buys no
# safety and costs a failed round-trip; anything with I/O still goes via tools.
_SAFE_MODULES = {
    name: __import__(name)
    for name in ("json", "re", "math", "statistics", "collections", "itertools",
                 "functools", "textwrap", "datetime", "random", "string",
                 "difflib", "asyncio")
}


def _safe_import(name, *args, **kwargs):
    """__import__ restricted to _SAFE_MODULES; anything else fails with an
    error that TEACHES (the real failure mode is a model retrying `import os`
    forever against an opaque 'ImportError: __import__ not found')."""
    root = name.partition(".")[0]
    if root not in _SAFE_MODULES:
        raise ImportError(
            f"import {root!r} isn't available in code mode. Files, shell, and "
            f"network go through the tools API instead — e.g. "
            f"`await tools.list_dir('.')`, `await tools.read_file(path)`, "
            f"`await tools.bash(command)`. "
            f"Importable here: {', '.join(sorted(_SAFE_MODULES))}."
        )
    return __import__(name, *args, **kwargs)


# Builtins the model's code may use — no open/eval/exec/compile; imports are
# whitelisted through _safe_import.
_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
              "int", "len", "list", "map", "max", "min", "print", "range", "repr",
              "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
              "True", "False", "None", "isinstance", "issubclass", "getattr",
              "hasattr", "type", "divmod", "pow", "ord", "chr", "hex", "oct",
              "bin", "frozenset", "bytes", "bytearray", "iter", "next", "slice",
              "format", "Exception", "BaseException", "ValueError", "TypeError",
              "KeyError", "IndexError", "AttributeError", "RuntimeError",
              "NameError", "ImportError", "StopIteration", "StopAsyncIteration",
              "ZeroDivisionError", "ArithmeticError", "LookupError", "OSError")
    if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))
}
_SAFE_BUILTINS["__import__"] = _safe_import


def api_reference(registry: ToolRegistry, exposed: set[str] | None) -> str:
    """A compact Python API doc for the available tools (one line each). Replaces
    N full JSON schemas with one readable reference."""
    lines = []
    for schema in registry.schemas():
        fn = schema["function"]
        name = fn["name"]
        if exposed is not None and name not in exposed:
            continue
        params = ", ".join((fn.get("parameters") or {}).get("properties", {}).keys())
        desc = (fn.get("description") or "").split("\n")[0][:100]
        ref = f"tools.{name}({params})" if "." not in name else f'call("{name}", {params})'
        lines.append(f"  {ref} — {desc}")
    return "\n".join(lines)


def run_code_schema(reference: str) -> dict[str, Any]:
    """The single tool the model sees in code-mode: write Python, get the result."""
    return {
        "type": "function",
        "function": {
            "name": RUN_CODE_NAME,
            "description": (
                "Run Python to do the work. You have these tools (await every call):\n"
                f"{reference}\n\n"
                "Rules: `await` every tools.* call; positional or keyword args both work; "
                "chain as many as you like in one block; "
                "`print(...)` for logs; end with `return <value>` to report the result. "
                "Namespaced tools: `await tools.ns.name(...)` or `await call(\"ns.name\", ...)`. "
                "Pure-Python imports are available (json, re, math, collections, "
                "itertools, functools, textwrap, datetime, statistics, random, "
                "string, difflib); everything with I/O (os, subprocess, open, "
                "pathlib, requests, …) is NOT importable — use the tools for "
                "files, shell, and network."),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "the Python to run"}},
                "required": ["code"],
            },
        },
    }


def _params_of(registry: ToolRegistry, exposed: set[str] | None) -> dict[str, list[str]]:
    """Ordered parameter names per tool, so code-mode can bind positional args to
    them (the model naturally writes `tools.calculator("2+2")`, not keyword-only)."""
    out: dict[str, list[str]] = {}
    for schema in registry.schemas():
        fn = schema["function"]
        name = fn["name"]
        if exposed is not None and name not in exposed:
            continue
        out[name] = list((fn.get("parameters") or {}).get("properties", {}).keys())
    return out


def _bind(name: str, params: dict[str, list[str]], args: tuple, kwargs: dict) -> dict:
    """Merge positional args into kwargs using the tool's declared parameter order,
    so both `tool(x, y)` and `tool(a=x, b=y)` work (models write both)."""
    if not args:
        return kwargs
    names = params.get(name) or []
    merged = dict(kwargs)
    for i, val in enumerate(args):
        if i >= len(names):
            raise TypeError(
                f"{name}() takes {len(names)} positional argument(s)"
                f" [{', '.join(names) or 'none'}] but {len(args)} were given")
        if names[i] in merged:
            raise TypeError(f"{name}() got multiple values for argument '{names[i]}'")
        merged[names[i]] = val
    return merged


class _Callable:
    """`tools.web_search(...)` and dotted `tools.github.get_pr(...)` → execute().
    Accepts positional OR keyword args (bound via the tool's schema)."""

    def __init__(self, execute, name: str, params: dict[str, list[str]]):
        self._execute, self._name, self._params = execute, name, params

    async def __call__(self, *args, **kwargs):
        return await self._execute(self._name, _bind(self._name, self._params, args, kwargs))

    def __getattr__(self, part: str) -> "_Callable":
        return _Callable(self._execute, f"{self._name}.{part}", self._params)


class _ToolNS:
    def __init__(self, execute, params: dict[str, list[str]]):
        self._execute, self._params = execute, params

    def __getattr__(self, name: str) -> _Callable:
        return _Callable(self._execute, name, self._params)


def _model_traceback() -> str:
    """The current exception's traceback with harness-internal frames stripped:
    the model should see ITS code failing, not codemode.py plumbing (which it
    misreads as harness breakage and retries the same code against)."""
    out: list[str] = []
    skipping = False
    for ln in traceback.format_exc().splitlines():
        if ln.startswith("  File "):
            skipping = "<code-mode>" not in ln
            if skipping:
                continue
        elif ln[:1].isspace():
            if skipping:
                continue
        else:  # header / exception message — always shown
            skipping = False
        out.append(ln)
    return "\n".join(out)


def _format(result: Any, logs: str) -> str:
    out = ""
    if logs.strip():
        out += f"[logs]\n{logs.rstrip()}\n"
    if result is not None:
        try:
            shown = json.dumps(result, default=str, indent=2)
        except (TypeError, ValueError):
            shown = str(result)
        out += f"[result]\n{shown}"
    return out.strip() or "[no result]"


class CodeMode:
    def __init__(self, registry: ToolRegistry, *, exposed: set[str] | None = None,
                 sandbox=None, timeout: int = 60):
        self.registry = registry
        self.exposed = exposed
        self.sandbox = sandbox
        self.timeout = timeout

    async def _call_tool(self, name: str, kwargs: dict) -> str:
        if self.exposed is not None and name not in self.exposed:
            return f"error: tool {name!r} isn't available in this mode"
        return await self.registry.execute(name, json.dumps(kwargs))

    async def run(self, code: str) -> str:
        if self.sandbox is not None and getattr(self.sandbox, "kind", "host") == "microvm":
            return await self._run_microvm(code)
        return await self._run_inprocess(code)

    # --- restricted in-process backend (host / default) ------------------

    async def _run_inprocess(self, code: str) -> str:
        params = _params_of(self.registry, self.exposed)
        tools = _ToolNS(self._call_tool, params)

        async def _call(name, *args, **kwargs):  # the dotted-name escape hatch
            return await self._call_tool(name, _bind(name, params, args, kwargs))

        g: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS, "tools": tools, "call": _call,
            **_SAFE_MODULES,
        }
        src = "async def __codemode__():\n" + textwrap.indent(code or "pass", "    ")
        buf = io.StringIO()
        try:
            exec(compile(src, "<code-mode>", "exec"), g)  # noqa: S102 — restricted ns
            with contextlib.redirect_stdout(buf):
                result = await asyncio.wait_for(g["__codemode__"](), self.timeout)
        except asyncio.TimeoutError:
            return f"error: code timed out after {self.timeout}s\n{_format(None, buf.getvalue())}"
        except Exception:
            # "error:" prefix — the loop's tool-error budget keys on it, so a
            # model stuck in a crash loop gets stopped instead of spinning.
            return (f"error: your code raised:\n{_model_traceback()}\n"
                    f"{_format(None, buf.getvalue())}")
        return _format(result, buf.getvalue())

    # --- microVM-bridged backend (hard isolation) ------------------------

    async def _run_microvm(self, code: str) -> str:
        """Run the code inside the VM; bridge each tool call back to the host over
        the shared workdir mount (host sees the VM's files because the workdir is
        bind-mounted). The model's code thus runs isolated, reaching the host only
        through the registry."""
        import shutil
        import uuid
        from pathlib import Path
        # A FRESH dir per run — reusing main.py confuses the VM's virtio-fs cache
        # (a host unlink+recreate can leave a stale "file gone" entry in the VM).
        run_dir = Path(self.sandbox.workdir) / ".codemode" / uuid.uuid4().hex[:12]
        run_dir.mkdir(parents=True, exist_ok=True)
        params = _params_of(self.registry, self.exposed)
        params_line = "_PARAMS = json.loads(%r)\n" % json.dumps(params)
        (run_dir / "main.py").write_text(_VM_RUNTIME + "\n" + params_line
                                         + "\n# --- user code ---\n"
                                         + "async def __user__():\n"
                                         + textwrap.indent(code or "pass", "    ")
                                         + "\n_run(__user__)\n")
        rel = f".codemode/{run_dir.name}/main.py"

        done = asyncio.Event()

        async def _bridge() -> None:  # host side: serve tool-call requests
            seen: set[str] = set()
            while not done.is_set():
                for req in sorted(run_dir.glob("req-*.json")):
                    if req.name in seen:
                        continue
                    seen.add(req.name)
                    try:
                        payload = json.loads(req.read_text())
                    except (OSError, ValueError):
                        continue
                    result = await self._call_tool(payload.get("tool", ""),
                                                   payload.get("args") or {})
                    resp = run_dir / req.name.replace("req-", "resp-")
                    resp.write_text(json.dumps({"result": result}))
                await asyncio.sleep(0.01)

        bridge = asyncio.ensure_future(_bridge())
        try:
            out, _code = await self.sandbox.exec(f"python3 -u {rel}", timeout=self.timeout)
        finally:
            done.set()
            with contextlib.suppress(Exception):
                await bridge
            with contextlib.suppress(Exception):
                shutil.rmtree(run_dir)
        # the in-VM runtime prints the formatted [logs]/[result]; pass it through
        return out.strip() or "[no result]"


# Runs inside the VM. Does file-RPC over /workspace/.codemode for each tool call.
_VM_RUNTIME = '''\
import asyncio, json, os, time, traceback
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
_n = 0
_PARAMS = {}  # tool -> [param names]; overwritten by an injected line below

def _rpc(name, kwargs):
    global _n; _n += 1; rid = _n
    req = os.path.join(_DIR, "req-%d.json" % rid)
    resp = os.path.join(_DIR, "resp-%d.json" % rid)
    tmp = req + ".tmp"
    with open(tmp, "w") as f: json.dump({"tool": name, "args": kwargs}, f)
    os.replace(tmp, req)
    for _ in range(12000):  # up to ~120s
        if os.path.exists(resp):
            with open(resp) as f: r = json.load(f)
            return r.get("result")
        time.sleep(0.01)
    raise TimeoutError("tool %s timed out" % name)

def _bind(name, args, kw):
    if not args: return kw
    names = _PARAMS.get(name) or []
    merged = dict(kw)
    for i, val in enumerate(args):
        if i >= len(names):
            raise TypeError("%s() takes %d positional arg(s) but %d were given"
                            % (name, len(names), len(args)))
        merged[names[i]] = val
    return merged

class _Callable:
    def __init__(self, name): self._name = name
    async def __call__(self, *a, **kw): return _rpc(self._name, _bind(self._name, a, kw))
    def __getattr__(self, p): return _Callable(self._name + "." + p)

class _NS:
    def __getattr__(self, n): return _Callable(n)

tools = _NS()
async def call(name, *a, **kw): return _rpc(name, _bind(name, a, kw))

def _run(user):
    buf = []
    _p = print
    def _print(*a, **k):
        import io as _io
        s = _io.StringIO(); _p(*a, file=s, **k); buf.append(s.getvalue())
    import builtins; builtins.print = _print
    try:
        result = asyncio.run(user())
    except Exception:
        builtins.print = _p
        _p("error: your code raised:\\n" + traceback.format_exc())
        if buf: _p("[logs]\\n" + "".join(buf).rstrip())
        return
    builtins.print = _p
    if buf: _p("[logs]\\n" + "".join(buf).rstrip())
    if result is not None:
        try: shown = json.dumps(result, default=str, indent=2)
        except Exception: shown = str(result)
        _p("[result]\\n" + shown)
'''
