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

# Builtins the model's code may use — no __import__/open/eval/exec/compile.
_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
              "int", "len", "list", "map", "max", "min", "print", "range", "repr",
              "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
              "True", "False", "None", "isinstance", "Exception")
    if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))
}


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
                "Rules: `await` every tools.* call; chain as many as you like in one block; "
                "`print(...)` for logs; end with `return <value>` to report the result. "
                "Namespaced tools: `await tools.ns.name(...)` or `await call(\"ns.name\", ...)`. "
                "No imports/file access except through the tools."),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "the Python to run"}},
                "required": ["code"],
            },
        },
    }


class _Callable:
    """`tools.web_search(...)` and dotted `tools.github.get_pr(...)` → execute()."""

    def __init__(self, execute, name: str):
        self._execute, self._name = execute, name

    async def __call__(self, **kwargs):
        return await self._execute(self._name, kwargs)

    def __getattr__(self, part: str) -> "_Callable":
        return _Callable(self._execute, f"{self._name}.{part}")


class _ToolNS:
    def __init__(self, execute):
        self._execute = execute

    def __getattr__(self, name: str) -> _Callable:
        return _Callable(self._execute, name)


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
        tools = _ToolNS(self._call_tool)

        async def _call(name, **kwargs):  # the dotted-name escape hatch
            return await self._call_tool(name, kwargs)

        g: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS, "tools": tools, "call": _call,
            "json": json, "asyncio": asyncio,
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
            return f"error while running your code:\n{traceback.format_exc()}\n{_format(None, buf.getvalue())}"
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
        (run_dir / "main.py").write_text(_VM_RUNTIME + "\n\n# --- user code ---\n"
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

class _Callable:
    def __init__(self, name): self._name = name
    async def __call__(self, **kw): return _rpc(self._name, kw)
    def __getattr__(self, p): return _Callable(self._name + "." + p)

class _NS:
    def __getattr__(self, n): return _Callable(n)

tools = _NS()
async def call(name, **kw): return _rpc(name, kw)

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
        _p("error while running your code:\\n" + traceback.format_exc())
        if buf: _p("[logs]\\n" + "".join(buf).rstrip())
        return
    builtins.print = _p
    if buf: _p("[logs]\\n" + "".join(buf).rstrip())
    if result is not None:
        try: shown = json.dumps(result, default=str, indent=2)
        except Exception: shown = str(result)
        _p("[result]\\n" + shown)
'''
