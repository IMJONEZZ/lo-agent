"""Support for user-simulation (e2e) tests: the Pilot driver, the real-server
fixture, per-scenario mock model behaviors, and a real-socket mock upstream.

The backbone path under test is the one a real `lo tui` user takes: headless
Textual HarnessApp in SERVER mode → HTTP POST /session on a real uvicorn
instance of the session server → SSE stream back → screen. The model behind
the server is MockLlamaCpp, spliced in through _build_session_app's
client_factory seam.
"""

from __future__ import annotations

import html
import json
import re
import threading
import time
from time import monotonic
from types import SimpleNamespace

from local_harness.inference.client import OpenAICompatClient
from local_harness.sandbox import make_sandbox
from local_harness.sim.scenario import Scenario
from local_harness.skills.skill import BUILTIN_SKILLS_DIR
from local_harness.tui.app import HarnessApp, PermissionModal

from mocks import MockLlamaCpp, chat_response

SKILLS_DIR = str(BUILTIN_SKILLS_DIR)
UPSTREAM_URL = "http://upstream"  # must match on server AND TUI client, or
UPSTREAM_MODEL = "test-model"  # _probe_server rebuilds the client sans mock


# ── the real session server, on a real localhost socket ────────────────────


def delayed_transport(mock: MockLlamaCpp, delay: float = 0.4):
    """The mock, but each chat call takes `delay` seconds of *async* time —
    real-model latency. Needed for flows that depend on the TUI's SSE
    subscription being up before the first tool call (permission modals):
    an instant mock answers before the client has even subscribed, and the
    server's fail-safe denies unheard permission requests."""
    import asyncio

    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            await asyncio.sleep(delay)
        return mock.handler(request)

    return httpx.MockTransport(handler)


def start_server(
    tmp_path,
    mock: MockLlamaCpp,
    *,
    interactive_permissions: bool = True,
    preset: str = "build",
    chat_delay: float = 0.0,
):
    """Run the real Starlette session server on a uvicorn daemon thread, its
    upstream client wired to `mock`. Returns (server_url, uvicorn_server, db)."""
    import uvicorn

    from local_harness.cli.main import _build_session_app, _free_port

    db = str(tmp_path / "lo.db")
    args = SimpleNamespace(
        url=UPSTREAM_URL,
        model=UPSTREAM_MODEL,
        db=db,
        preset=preset,
        max_steps=8,
        context_budget=None,
        compact_fraction=0.85,
        no_guardrails=False,
        required_steps="",
        terminal_tool="",
        memory_dir=str(tmp_path / "mem"),
    )
    app = _build_session_app(
        args,
        make_sandbox("host", str(tmp_path)),
        interactive_permissions=interactive_permissions,
        client_factory=lambda url, model: OpenAICompatClient(
            url,
            model,
            transport=delayed_transport(mock, chat_delay)
            if chat_delay
            else mock.transport(),
        ),
    )
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):  # ≤10s for the socket to accept
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("embedded test server never started")
    return f"http://127.0.0.1:{port}", server, db


def stop_server(server) -> None:
    server.should_exit = True


def start_mock_upstream(mock: MockLlamaCpp):
    """MockLlamaCpp behind a REAL localhost socket, for TUI subprocesses
    (a PTY child can't use httpx.MockTransport). Returns (url, server)."""
    import httpx
    import uvicorn

    from local_harness.cli.main import _free_port

    async def asgi(scope, receive, send):
        if scope["type"] != "http":
            return
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        url = "http://upstream" + scope["path"]
        if scope.get("query_string"):
            url += "?" + scope["query_string"].decode()
        try:
            resp = mock.handler(httpx.Request(scope["method"], url, content=body))
        except Exception:
            await send({"type": "http.response.start", "status": 502, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        headers = [
            (k.encode(), v.encode())
            for k, v in resp.headers.items()
            if k.lower() != "content-length"
        ]
        await send(
            {"type": "http.response.start", "status": resp.status_code, "headers": headers}
        )
        await send({"type": "http.response.body", "body": resp.content})

    port = _free_port()
    config = uvicorn.Config(asgi, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("mock upstream server never started")
    return f"http://127.0.0.1:{port}", server


def make_server_app(db: str, mock: MockLlamaCpp, server_url: str) -> HarnessApp:
    """The server-mode twin of test_tui.make_app: a TUI that is a thin client
    of `server_url`, sharing its db (the default `lo tui` architecture)."""
    client = OpenAICompatClient(UPSTREAM_URL, UPSTREAM_MODEL, transport=mock.transport())
    return HarnessApp(client, db, server=server_url, shared_db=True, skills_dir=SKILLS_DIR)


# ── PilotDriver: the scenario Driver over Textual's run_test pilot ──────────

_KEYMAP = {
    "\r": "enter",
    "\n": "enter",
    "\t": "tab",
    "\x1b": "escape",
    "\x14": "ctrl+t",
    "\x04": "ctrl+d",
    "\x07": "ctrl+g",
    "\x15": "ctrl+u",
    " ": "space",
}
_SEQS = {"\x1b[3~": "delete"}


def keys_to_presses(keys: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(keys):
        for seq, name in _SEQS.items():
            if keys.startswith(seq, i):
                out.append(name)
                i += len(seq)
                break
        else:
            ch = keys[i]
            out.append(_KEYMAP.get(ch, ch))
            i += 1
    return out


_TEXT_RUN = re.compile(r"<text([^>]*)>([^<]*)</text>")
_ATTR = {a: re.compile(a + r'="([-\d.e]+)"') for a in ("x", "y")}


def _svg_lines(svg: str) -> list[str]:
    """Reconstruct plain screen lines from an export_screenshot() SVG: group
    text runs by y, order by x, unescape entities."""
    rows: dict[float, list[tuple[float, str]]] = {}
    for idx, m in enumerate(_TEXT_RUN.finditer(svg)):
        attrs, content = m.group(1), m.group(2)
        ym = _ATTR["y"].search(attrs)
        xm = _ATTR["x"].search(attrs)
        y = float(ym.group(1)) if ym else -1.0
        x = float(xm.group(1)) if xm else float(idx)
        text = html.unescape(content).replace("\xa0", " ").replace("\n", "")
        if text:
            rows.setdefault(y, []).append((x, text))
    return [
        "".join(t for _, t in sorted(runs)).rstrip()
        for y, runs in sorted(rows.items())
    ]


class PilotDriver:
    """Drives a HarnessApp under app.run_test(). Screen text is an append-only
    stream: new screen lines (diffed between polls) plus every notification
    (toasts don't appear in export_screenshot, so app.notify is hooked)."""

    def __init__(self, app: HarnessApp, pilot):
        self.app, self.pilot = app, pilot
        self._buffer: list[str] = []
        self._last_screen: set[str] = set()
        orig_notify = app.notify

        def hooked_notify(message, *a, **kw):
            self._buffer.append(f"[notify] {message}")
            return orig_notify(message, *a, **kw)

        app.notify = hooked_notify

    def _sync(self) -> None:
        try:
            lines = _svg_lines(self.app.export_screenshot())
        except Exception:
            return
        current = set(lines)
        for line in lines:
            if line not in self._last_screen:
                self._buffer.append(line)
        self._last_screen = current

    async def send(self, keys: str) -> None:
        for key in keys_to_presses(keys):
            await self.pilot.press(key)
        await self.pilot.pause()

    def mark(self) -> int:
        self._sync()
        return len(self._buffer)

    def new_text(self, since: int) -> str:
        self._sync()
        return "\n".join(self._buffer[since:])

    async def wait_marker(self, marker: str, timeout: float, since: int) -> bool:
        deadline = monotonic() + timeout
        needle = marker.lower()
        while monotonic() < deadline:
            if needle in self.new_text(since).lower():
                return True
            await self.pilot.pause(0.1)
        return False

    async def settle(self, seconds: float) -> None:
        await self.pilot.pause(seconds)
        self._sync()

    def dump(self) -> str:
        self._sync()
        screen = "\n".join(_line for _line in sorted(self._last_screen) if _line)
        tail = "\n".join(self._buffer[-40:])
        return f"{screen}\n--- recent output ---\n{tail}"

    async def answer_permission(self, allow: bool, timeout: float = 15.0) -> bool:
        """Wait for the PermissionModal (raised by the server over SSE) and
        answer it with the y/n key — the real user gesture."""
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if isinstance(self.app.screen, PermissionModal):
                await self.pilot.press("y" if allow else "n")
                await self.pilot.pause()
                return True
            await self.pilot.pause(0.1)
        return False


# ── per-scenario mock model behavior ────────────────────────────────────────
# One chat_fn per journey, returning EXACTLY the controlled answers the
# scenario's live prompts instruct a real model to construct — so the same
# Scenario passes against either backend.


def _final(text: str) -> dict:
    return chat_response(content=text)


def assert_ux_invariants(db: str) -> None:
    """The UX floor every journey must clear, regardless of what it tests:
    the harness may never show a user the same failing tool result more than
    the error budget allows (3 = budget of 2 + the batch that trips it).
    This is the invariant the real session 46eade64 violated — six identical
    ImportErrors scrolled past before the user gave up and hit Esc."""
    from local_harness.events.log import TOOL_CALL, EventLog

    log = EventLog(db)
    for run in log.runs():
        streak, last = 0, None
        for e in log.events(run.run_id, type=TOOL_CALL):
            res = e.payload.get("result") or ""
            if res.startswith("error"):
                streak = streak + 1 if res == last else 1
                last = res
            else:
                streak, last = 0, None
            assert streak <= 3, (
                f"run {run.run_id[:8]}: the same failing tool result reached the "
                f"screen {streak}× — a retry loop leaked to the user:\n{res[:300]}"
            )


def _is_probe(body: dict) -> bool:
    """The server's capability probe also issues chat calls (seeded 'rivers' /
    'mountains' prompts); scenario scripting must not consume them."""
    text = " ".join(str(m.get("content") or "") for m in body.get("messages", []))
    return "sentence about rivers" in text or "sentences about mountains" in text


def agent_bodies(mock: MockLlamaCpp) -> list[dict]:
    """The chat requests made by actual agent turns (probe traffic filtered)."""
    return [b for b in mock.chat_bodies if not _is_probe(b)]


def _skip_probe(fn):
    def handler(body: dict) -> dict:
        if _is_probe(body):
            return chat_response(content=f"deterministic-{body.get('seed', 0)}")
        return fn(body)

    return handler


_BAD_IMPORT_CODE = "import os\nprint(os.listdir('.'))\nreturn 'unreachable'"
_GOOD_TOOLS_CODE = "files = await tools.list_dir('.')\nreturn files"


def _run_code_call(code: str) -> dict:
    return chat_response(tool_calls=[("c1", "run_code", json.dumps({"code": code}))])


def scripted_chat(scenario: Scenario, tmp_path=None):
    """A chat_fn for MockLlamaCpp matching `scenario`'s markers."""
    name = scenario.name

    if name == "codemode-import-recovery":
        # A REALISTIC model: reaches for `import os` first, then does what the
        # error message says. This is the journey distilled from the real
        # failed session 46eade64 (six blind retries against an opaque error).
        def recover(body: dict) -> dict:
            last = str(body["messages"][-1].get("content") or "")
            if "isn't available in code mode" in last:
                return _run_code_call(_GOOD_TOOLS_CODE)  # it taught → correct
            if "[result]" in last or "[logs]" in last:
                return _final("Project listed. RECOVERED-OK")
            return _run_code_call(_BAD_IMPORT_CODE)

        return _skip_probe(recover)

    if name == "codemode-crash-loop-breaker":
        # A model that never learns: identical broken import every turn. The
        # loop's tool-error budget must cut it off, visibly.
        return _skip_probe(lambda body: _run_code_call(_BAD_IMPORT_CODE))

    if name in ("permission-allow", "permission-deny"):
        state = {"called": False}

        def perm(body: dict) -> dict:
            if not state["called"]:
                state["called"] = True
                return chat_response(
                    tool_calls=[
                        (
                            "c1",
                            "write_file",
                            '{"path": "perm-test.txt", "content": "PERMOK"}',
                        )
                    ]
                )
            last = str(body["messages"][-1].get("content", "")).lower()
            denied = "not approved" in last or "denied" in last
            return _final("PERM-REFUSED" if denied else "PERM-GRANTED")

        return _skip_probe(perm)

    if name == "plan-approve-build":

        def plan(body: dict) -> dict:
            sys_and_user = " ".join(
                str(m.get("content") or "") for m in body["messages"]
            )
            if "approved. Implement it now" in sys_and_user:
                return _final("All steps applied. FINISHED-OK")
            return _final(
                "## Plan\n1. Create hello.txt containing hello.\n"
                "2. Reply with the concatenation of FIN and ISHED-OK."
            )

        return _skip_probe(plan)

    if name in ("rewind-picker", "history-filter-rename"):

        def rewind(body: dict) -> dict:
            user = " ".join(
                str(m.get("content") or "")
                for m in body["messages"]
                if m.get("role") == "user"
            )
            if "SEC and OND-ANSWER" in user:
                return _final("SECOND-ANSWER")
            return _final("BANANA")

        return _skip_probe(rewind)

    # Default: every turn answers the standard chat task.
    return _skip_probe(lambda body: _final("BANANA"))
