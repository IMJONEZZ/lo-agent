"""Phase 4: the web client route and autonomous-action drafts."""

from __future__ import annotations

import httpx
import pytest

from local_harness.background import propose_actions
from local_harness.events.bus import EventBus
from local_harness.events.log import EventLog, RUN_COMPLETED, RUN_FAILED
from local_harness.inference.client import OpenAICompatClient
from local_harness.server.app import create_server_app
from local_harness.server.sessions import SessionManager

from mocks import MockLlamaCpp, chat_response


async def test_web_client_served_at_root(tmp_path):
    log = EventLog(tmp_path / "e.db")
    mgr = SessionManager(EventBus(log), lambda *a, **k: None)
    app = create_server_app(mgr, health={"status": "ok", "model": "qwen", "capabilities": {"tier": 3}})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://srv") as c:
        r = await c.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        html = r.text
        for marker in ("local_harness", "EventSource", "/session", "permission_request", "--jade"):
            assert marker in html


async def test_propose_actions_drafts_for_failed_runs(tmp_path):
    log = EventLog(tmp_path / "e.db")
    rid = log.create_run("build the thing")
    log.append(rid, RUN_FAILED, {"error": "max_steps exceeded"})
    ok = log.create_run("a fine run")
    log.append(ok, RUN_COMPLETED, {"answer": "done"})  # completed → no proposal

    client = OpenAICompatClient(
        "http://x", "test-model",
        transport=MockLlamaCpp(script={1: chat_response(content="Retry with a smaller scope.")}).transport())
    drafts = tmp_path / "drafts"
    written = await propose_actions(log, client, drafts, limit=5)
    assert len(written) == 1  # only the failed run
    text = (drafts / f"proposed-{rid[:8]}.md").read_text()
    assert "Proposed next action" in text and "Retry with a smaller scope" in text
    assert "NOT executed" in text  # the safety framing
    # idempotent: a second pass writes nothing new
    assert await propose_actions(log, client, drafts, limit=5) == []
