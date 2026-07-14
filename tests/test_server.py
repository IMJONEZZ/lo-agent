"""Session server: a real agent run driven onto the bus, streamed to clients.

Locks the OpenCode-parity contract end-to-end — start a session, subscribe, and
see persisted events (model/tool/completed) interleaved with ephemeral token and
tool-progress deltas, with the ephemerals never landing in the replayable log.
Plus the HTTP surface over ASGI.
"""

import asyncio
import json

import httpx
import pytest

from local_harness.agent.loop import Agent
from local_harness.agent.permissions import Permissions
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.bus import (
    EventBus, PERMISSION_REQUEST, TOKEN_DELTA, TOOL_PROGRESS, TERMINAL,
)
from local_harness.events.log import (
    EventLog, MODEL_CALL, TOOL_CALL, RUN_COMPLETED,
)
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.server.sessions import SessionManager
from local_harness.server.app import create_server_app

from mocks import MockLlamaCpp, chat_response

CAPS = Capabilities(server="llama.cpp", seed=True, logprobs=True)
SCRIPT = {
    1: chat_response(tool_calls=[("c1", "calculator", '{"expression": "2+3"}')]),
    2: chat_response(content="The answer is 5."),
    424242: chat_response(content="probe"),
}


def make_manager(tmp_path, script=SCRIPT):
    log = EventLog(tmp_path / "e.db")
    bus = EventBus(log)

    def factory(on_token, on_tool, on_notice, preset=None):
        client = OpenAICompatClient(
            "http://t", "test-model", transport=MockLlamaCpp(script=script).transport())
        sysprompt = None
        exposed = None
        if preset:
            from local_harness.agent.presets import get_preset
            p = get_preset(preset)
            sysprompt, exposed = p.system_prompt, p.exposed()
        kw = {"system_prompt": sysprompt} if sysprompt else {}
        return Agent(client, ToolRegistry(builtin_tools()), log, capabilities=CAPS,
                     base_seed=1, exposed_tools=exposed,
                     on_token=on_token, on_tool=on_tool, on_notice=on_notice, **kw)

    return SessionManager(bus, factory), bus


async def test_session_streams_persistent_and_ephemeral(tmp_path):
    mgr, bus = make_manager(tmp_path)
    run_id = mgr.start("compute 2+3")
    events = [ev async for ev in mgr.stream(run_id, stop_on=TERMINAL)]
    types = [e.type for e in events]

    assert "run_started" in types          # catch-up
    assert MODEL_CALL in types and TOOL_CALL in types
    assert TOOL_PROGRESS in types           # ephemeral: ⚙ running calculator…
    assert TOKEN_DELTA in types             # ephemeral: streamed answer tokens
    assert types[-1] == RUN_COMPLETED

    # ephemeral deltas must NOT be in the replayable log
    persisted = [e.type for e in bus.log.events(run_id)]
    assert TOKEN_DELTA not in persisted and TOOL_PROGRESS not in persisted
    assert persisted[-1] == RUN_COMPLETED


async def test_two_clients_observe_one_session(tmp_path):
    import asyncio
    mgr, _ = make_manager(tmp_path)
    run_id = mgr.start("compute 2+3")
    a, b = await asyncio.gather(
        _collect(mgr, run_id), _collect(mgr, run_id))
    # both clients see the same persisted spine
    spine_a = [e.type for e in a if e.seq >= 0]
    spine_b = [e.type for e in b if e.seq >= 0]
    assert spine_a == spine_b
    assert spine_a[-1] == RUN_COMPLETED


async def _collect(mgr, run_id):
    return [ev async for ev in mgr.stream(run_id, stop_on=TERMINAL)]


async def test_interrupt_emits_terminal(tmp_path):
    """A cancelled run still produces a terminal event so subscribers don't hang.
    Interrupt synchronously right after start — no await has let the agent task run
    yet, so it's still pending and the cancellation lands at its first await."""
    mgr, bus = make_manager(tmp_path)
    run_id = mgr.start("compute 2+3")
    assert mgr.interrupt(run_id) is True
    events = [ev async for ev in mgr.stream(run_id, stop_on=TERMINAL)]
    assert events[-1].type in TERMINAL
    # the terminal is the interrupt failure
    assert events[-1].payload.get("error") == "interrupted"


# --- HTTP surface ---------------------------------------------------------

async def test_http_start_list_health(tmp_path):
    mgr, _ = make_manager(tmp_path)
    app = create_server_app(mgr, health={"status": "ok", "tier": 3})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://srv") as c:
        r = await c.post("/session", json={"task": "compute 2+3"})
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        h = await c.get("/health")
        assert h.json()["tier"] == 3

        # missing task -> 400
        assert (await c.post("/session", json={})).status_code == 400

        # the session shows up in the list
        sessions = (await c.get("/sessions")).json()
        assert any(s["run_id"] == run_id for s in sessions)


def _capturing_manager(tmp_path):
    """A manager whose factory records the preset it was built with, so we can
    assert the preset actually reaches the agent build (the bug this fixes)."""
    log = EventLog(tmp_path / "e.db")
    bus = EventBus(log)
    seen: list[str | None] = []

    def factory(on_token, on_tool, on_notice, preset=None):
        seen.append(preset)
        client = OpenAICompatClient(
            "http://t", "test-model", transport=MockLlamaCpp(script=SCRIPT).transport())
        return Agent(client, ToolRegistry(builtin_tools()), log, capabilities=CAPS,
                     base_seed=1, on_token=on_token, on_tool=on_tool, on_notice=on_notice)

    return SessionManager(bus, factory), seen


async def test_preset_reaches_factory_via_manager(tmp_path):
    mgr, seen = _capturing_manager(tmp_path)
    run_id = mgr.start("make a plan", preset="plan")
    [_ async for _ in mgr.stream(run_id, stop_on=TERMINAL)]
    assert "plan" in seen  # the plan preset was applied server-side, not ignored


async def test_preset_flows_through_http_body(tmp_path):
    mgr, seen = _capturing_manager(tmp_path)
    app = create_server_app(mgr)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://srv") as c:
        run_id = (await c.post("/session",
                               json={"task": "plan it", "preset": "explore"})).json()["run_id"]
        [_ async for _ in mgr.stream(run_id, stop_on=TERMINAL)]
    assert "explore" in seen


def test_server_factory_applies_plan_preset(tmp_path):
    """The real cli factory path: a plan-preset agent must expose only read tools
    and use the plan system prompt — i.e. it can't write."""
    mgr, _ = make_manager(tmp_path)  # make_manager's factory now applies presets
    plan_agent = mgr._agent_for("r", preset="plan")
    exposed = plan_agent.exposed_tools
    assert exposed is not None and "write_file" not in exposed and "read_file" in exposed
    assert "PLAN mode" in plan_agent.system_prompt
    build_agent = mgr._agent_for("r", preset="build")
    assert build_agent.exposed_tools is None  # build sees everything


# --- interactive tool approval over the bus -------------------------------

async def _await_permission_request(mgr, run_id):
    async for ev in mgr.bus.subscribe(run_id, replay=False):
        if ev.type == PERMISSION_REQUEST:
            return ev.payload["request_id"], ev.payload


async def _ask_with_subscriber(mgr, run_id, tool="write_file", args="{}"):
    """Register a subscriber, then issue a permission request once it's live.
    Returns (asker_task, request_id, payload)."""
    watcher = asyncio.ensure_future(_await_permission_request(mgr, run_id))
    for _ in range(200):  # wait until the watcher's queue is registered
        if mgr.bus.subscriber_count(run_id) > 0:
            break
        await asyncio.sleep(0.005)
    asker = asyncio.ensure_future(mgr.request_permission(run_id, tool, args))
    request_id, payload = await watcher
    return asker, request_id, payload


async def test_permission_request_allow(tmp_path):
    mgr, _ = _capturing_manager(tmp_path)
    run_id = mgr.bus.create_run("t")
    asker, req_id, payload = await _ask_with_subscriber(mgr, run_id)
    assert payload["tool"] == "write_file"
    assert mgr.resolve_permission(req_id, True) is True
    assert await asker is True  # approved → tool allowed


async def test_permission_request_deny(tmp_path):
    mgr, _ = _capturing_manager(tmp_path)
    run_id = mgr.bus.create_run("t")
    asker, req_id, _ = await _ask_with_subscriber(mgr, run_id)
    assert mgr.resolve_permission(req_id, False) is True
    assert await asker is False  # denied


async def test_permission_denied_with_no_subscriber(tmp_path):
    mgr, _ = _capturing_manager(tmp_path)  # non-interactive → instant deny (headless)
    run_id = mgr.bus.create_run("t")
    assert await mgr.request_permission(run_id, "bash", "{}") is False  # nobody to ask


async def test_interactive_permission_waits_for_late_subscriber(tmp_path):
    """Regression: an ask-tier tool that fires while the client's SSE subscription
    is momentarily detached must NOT auto-deny. An interactive manager waits for a
    subscriber to (re)attach, then publishes the request and resolves it — instead
    of the instant deny that produced the code-mode denial cascade."""
    log = EventLog(tmp_path / "e.db")
    bus = EventBus(log)
    mgr = SessionManager(bus, lambda *a, **k: None,
                         interactive_permissions=True, permission_timeout=5.0)
    run_id = bus.create_run("t")
    assert bus.subscriber_count(run_id) == 0  # nobody attached at ask time

    asker = asyncio.ensure_future(mgr.request_permission(run_id, "bash", "{}"))
    await asyncio.sleep(0.2)
    assert not asker.done()  # waiting for a subscriber, not denying

    # client (re)connects; the request is published to it and resolves normally
    watcher = asyncio.ensure_future(_await_permission_request(mgr, run_id))
    for _ in range(200):
        if bus.subscriber_count(run_id) > 0:
            break
        await asyncio.sleep(0.005)
    req_id, _ = await watcher
    assert mgr.resolve_permission(req_id, True) is True
    assert await asker is True


async def test_interactive_permission_denies_if_no_subscriber_appears(tmp_path):
    """Interactive, but no client ever attaches: deny after the timeout so the run
    fails safe instead of hanging forever."""
    log = EventLog(tmp_path / "e.db")
    bus = EventBus(log)
    mgr = SessionManager(bus, lambda *a, **k: None,
                         interactive_permissions=True, permission_timeout=0.2)
    run_id = bus.create_run("t")
    assert await mgr.request_permission(run_id, "bash", "{}") is False


def test_resolve_unknown_permission_is_false(tmp_path):
    mgr, _ = _capturing_manager(tmp_path)
    assert mgr.resolve_permission("nope", True) is False


def test_interactive_overrides_factory_approver(tmp_path):
    """interactive_permissions swaps the factory's approver for the bus one; a
    non-interactive manager leaves it untouched (headless auto-approve)."""
    log = EventLog(tmp_path / "e.db")
    bus = EventBus(log)
    sentinel = lambda _t, _a: True  # the factory's own approver

    def factory(on_token, on_tool, on_notice, preset=None):
        client = OpenAICompatClient(
            "http://t", "test-model", transport=MockLlamaCpp(script=SCRIPT).transport())
        tools = ToolRegistry(builtin_tools())
        tools.permissions = Permissions(ask=["*"], approver=sentinel)
        return Agent(client, tools, log, capabilities=CAPS, base_seed=1,
                     on_token=on_token, on_tool=on_tool, on_notice=on_notice)

    interactive = SessionManager(bus, factory, interactive_permissions=True)
    assert interactive._agent_for("r").tools.permissions.approver is not sentinel

    headless = SessionManager(bus, factory, interactive_permissions=False)
    assert headless._agent_for("r").tools.permissions.approver is sentinel


async def test_http_permission_route_resolves(tmp_path):
    mgr, _ = _capturing_manager(tmp_path)
    app = create_server_app(mgr)
    transport = httpx.ASGITransport(app=app)
    run_id = mgr.bus.create_run("t")
    asker, req_id, _ = await _ask_with_subscriber(mgr, run_id)
    async with httpx.AsyncClient(transport=transport, base_url="http://srv") as c:
        r = await c.post(f"/session/{run_id}/permission",
                         json={"request_id": req_id, "approved": True})
        assert r.json()["resolved"] is True
    assert await asker is True


async def test_http_sse_stream(tmp_path):
    mgr, _ = make_manager(tmp_path)
    app = create_server_app(mgr)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://srv") as c:
        run_id = (await c.post("/session", json={"task": "compute 2+3"})).json()["run_id"]
        seen = []
        async with c.stream("GET", f"/session/{run_id}/events?once=1") as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    seen.append(line.split(":", 1)[1].strip())
                if "run_completed" in seen:
                    break
        assert "run_started" in seen
        assert "run_completed" in seen


# --- POST /connect: repoint the upstream without restarting the server -------


async def test_connect_route_501_when_unsupported(tmp_path):
    mgr, _ = make_manager(tmp_path)
    app = create_server_app(mgr, health={"status": "ok"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://srv") as c:
        assert (await c.post("/connect", json={"url": "http://x"})).status_code == 501


def _session_app(tmp_path, url="http://down.invalid:9"):
    from local_harness.cli import main as cli
    from local_harness.sandbox import make_sandbox

    args = cli.build_parser().parse_args(
        ["serve", "--url", url, "--model", "m", "--db", str(tmp_path / "h.db")]
    )
    return cli._build_session_app(args, make_sandbox("host", str(tmp_path)))


async def test_health_reports_upstream_and_connect_repoints(tmp_path, monkeypatch):
    from local_harness.cli import main as cli

    app = _session_app(tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://srv") as c,
        transport.app.router.lifespan_context(transport.app),
    ):
        h = (await c.get("/health")).json()
        assert h["status"] == "degraded"
        assert h["upstream"] == "http://down.invalid:9"
        assert h["error"]

        # repoint to another unreachable host: still degraded, upstream updated
        h = (await c.post("/connect", json={"url": "http://also.invalid:9"})).json()
        assert h["status"] == "degraded"
        assert h["upstream"] == "http://also.invalid:9"

        # repoint to a "good" upstream (mocked client+probe): health flips to ok
        class FakeClient:
            def __init__(self, url, model, **kw):
                self.base_url, self.model = url, model

            async def get(self, path):
                return None  # reachable

            async def aclose(self):
                pass

        async def fake_probe(client):
            return CAPS

        monkeypatch.setattr(cli, "OpenAICompatClient", FakeClient)
        monkeypatch.setattr(cli, "probe", fake_probe)
        h = (await c.post("/connect", json={"url": "http://pedrogpt:8080"})).json()
        assert h["status"] == "ok"
        assert h["upstream"] == "http://pedrogpt:8080"
        assert h["capabilities"]["server"] == "llama.cpp"


async def test_session_after_failed_startup_fails_with_upstream_hint(tmp_path):
    from local_harness.events.bus import TERMINAL as _TERMINAL

    app = _session_app(tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://srv") as c,
        transport.app.router.lifespan_context(transport.app),
    ):
        r = await c.post("/session", json={"task": "hello"})
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        # follow the stream to its terminal event: a RUN_FAILED naming the upstream
        events = []
        async with c.stream(
            "GET", f"/session/{run_id}/events", params={"once": "1"}
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    events.append(line.split(": ", 1)[1])
                if line.startswith("data: ") and events and events[-1] == "run_failed":
                    payload = json.loads(line.split(": ", 1)[1])
                    err = payload["payload"]["error"]
                    assert "down.invalid" in err and "connect" in err.lower()
                    break
        assert events[-1] == "run_failed"
