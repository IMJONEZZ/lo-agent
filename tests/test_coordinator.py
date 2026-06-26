"""Phase 3 Coordinator: spawn_agents fan-out + gather, inter-agent messaging
(inbox delivery + loop injection), and the daemon guard."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from local_harness.agent.loop import Agent
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.bus import EventBus, TERMINAL
from local_harness.events.log import AGENT_SPAWNED, EventLog, RUN_COMPLETED, USER_MESSAGE
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.server.sessions import SessionManager

from mocks import MockLlamaCpp, chat_response

CAPS = Capabilities(server="llama.cpp", seed=True, logprobs=True)


def _coordinator_handler():
    """A lead that fans out two subtasks then synthesizes; workers answer directly.
    Returns plain JSON (the test factory disables streaming)."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        if path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {}})
        if path != "/v1/chat/completions":
            return httpx.Response(404)
        body = json.loads(request.content)
        msgs = body["messages"]
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "") or ""
        last = msgs[-1]
        if first_user.startswith("subtask"):                       # a worker
            return httpx.Response(200, json=chat_response(content=f"finished {first_user}"))
        if last["role"] == "tool" and "worker" in (last.get("content") or ""):  # lead, post-gather
            return httpx.Response(200, json=chat_response(content="Synthesized: both workers done."))
        return httpx.Response(200, json=chat_response(  # lead, first step: fan out
            tool_calls=[("c1", "spawn_agents", '{"tasks": ["subtask A", "subtask B"]}')]))
    return handler


def _coordinator_manager(tmp_path):
    log = EventLog(tmp_path / "e.db")
    bus = EventBus(log)
    transport = httpx.MockTransport(_coordinator_handler())

    def factory(on_token, on_tool, on_notice, preset=None):
        client = OpenAICompatClient("http://t", "test-model", transport=transport)
        return Agent(client, ToolRegistry(builtin_tools()), log, capabilities=CAPS,
                     base_seed=1, on_token=None, on_tool=on_tool, on_notice=on_notice)

    return SessionManager(bus, factory), bus


async def test_coordinator_fans_out_and_gathers(tmp_path):
    mgr, bus = _coordinator_manager(tmp_path)
    run_id = mgr.start("coordinate the audit")
    [_ async for _ in mgr.stream(run_id, stop_on=TERMINAL)]
    await mgr.drain()  # let the worker tasks settle

    # the lead spawned exactly two workers, recorded on its own log
    spawned = bus.log.events(run_id, type=AGENT_SPAWNED)
    assert len(spawned) == 2
    child_ids = [s.payload["child_run_id"] for s in spawned]
    assert [s.payload["task"] for s in spawned] == ["subtask A", "subtask B"]

    # each worker is its own completed, event-sourced run
    for cid in child_ids:
        answers = [e.payload["answer"] for e in bus.log.events(cid, type=RUN_COMPLETED)]
        assert answers and answers[-1].startswith("finished subtask")
        assert mgr._depth[cid] == 1  # workers are depth 1 (can't recurse)

    # the lead synthesized after gathering the workers' results
    lead_answer = [e.payload["answer"] for e in bus.log.events(run_id, type=RUN_COMPLETED)][-1]
    assert "Synthesized" in lead_answer


async def test_inbox_delivery_and_resolution(tmp_path):
    mgr, bus = _coordinator_manager(tmp_path)
    rid = bus.create_run("worker")
    mgr._tasks[rid] = asyncio.ensure_future(asyncio.sleep(5))  # mark it "running"
    try:
        assert mgr.deliver_message(rid[:8], "abc12345", "found the root cause") is True  # prefix ok
        drained = mgr._drain_inbox(rid)
        assert drained and "found the root cause" in drained[0] and "abc12345" in drained[0]
        assert mgr._drain_inbox(rid) == []                       # drained once
        assert mgr.deliver_message("nonexistent", "x", "y") is False
    finally:
        mgr._tasks[rid].cancel()


async def test_loop_injects_inbox_messages():
    client = OpenAICompatClient(
        "http://x", "test-model",
        transport=MockLlamaCpp(script={1: chat_response(content="ack")}).transport())
    log = EventLog(":memory:")
    agent = Agent(client, ToolRegistry([]), log, capabilities=Capabilities(),
                  base_seed=1, max_steps=3)
    box = ["peer says hi"]
    agent.inbox = lambda: ([box.pop(0)] if box else [])
    result = await agent.run("do a thing")
    assert result.status == "completed"
    injected = [e.payload["content"] for e in log.events(result.run_id, type=USER_MESSAGE)]
    assert "peer says hi" in injected


def test_daemon_requires_tmux(monkeypatch):
    import argparse

    from local_harness.cli import main as cli
    monkeypatch.setattr("shutil.which", lambda _name: None)  # tmux absent
    args = argparse.Namespace(action="start", host="127.0.0.1", port=8099, db="x.db",
                              url="http://localhost:8080", model="", approval="auto")
    with pytest.raises(SystemExit):
        cli.cmd_daemon(args)
