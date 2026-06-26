"""Phase 2 pieces 3-7: NotebookEdit, REPL, Snip, /review presets, MCP http+auth."""

from __future__ import annotations

import json

import httpx
import pytest

from local_harness.agent.loop import Agent
from local_harness.agent.tools import (
    _apply_notebook_edit, builtin_tools, repl, ToolRegistry,
)
from local_harness.events.log import (
    EventLog, MESSAGE_SNIPPED, MODEL_CALL, RUN_COMPLETED, TOOL_CALL,
)
from local_harness.inference.capabilities import Capabilities
from local_harness.integrations.load import registry_with_sources
from local_harness.integrations.mcp import MCPClient, MCPError, http_transport


# --- NotebookEdit ----------------------------------------------------------

def _nb(cells):
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def test_notebook_replace_preserves_outputs():
    nb = _nb([{"cell_type": "code", "source": ["print(1)"],
               "outputs": [{"x": 1}], "execution_count": 3}])
    text, msg = _apply_notebook_edit(nb, 0, "print(2)", "code", "replace")
    cell = json.loads(text)["cells"][0]
    assert cell["source"] == ["print(2)"] and cell["outputs"] == [{"x": 1}]
    assert "replace" in msg


def test_notebook_insert_and_delete():
    nb = _nb([{"cell_type": "code", "source": ["a"]}])
    text, _ = _apply_notebook_edit(nb, 1, "# header", "markdown", "insert")
    cells = json.loads(text)["cells"]
    assert len(cells) == 2 and cells[1]["cell_type"] == "markdown"
    text2, _ = _apply_notebook_edit(text, 0, "", "code", "delete")
    assert len(json.loads(text2)["cells"]) == 1


def test_notebook_out_of_range_errors():
    import pytest as _p
    with _p.raises(IndexError):
        _apply_notebook_edit(_nb([]), 5, "x", "code", "replace")


# --- REPL ------------------------------------------------------------------

def test_repl_persists_state_and_resets():
    assert repl("a = 10", session="t") == "[no output]"
    assert repl("a * 2", session="t") == "20\n"          # state persisted, value echoed
    assert "NameError" in repl("a", session="t", reset=True)  # reset cleared it
    assert repl("1 + 1", session="other") == "2\n"        # sessions are independent


def test_repl_and_notebook_are_builtins():
    names = {t.name for t in builtin_tools()}
    assert {"repl", "notebook_edit"} <= names


# --- Snip ------------------------------------------------------------------

def test_snip_collapses_content_in_reconstruct():
    log = EventLog(":memory:")
    rid = log.create_run("task")
    log.append(rid, MODEL_CALL, {"call_index": 0, "response": {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"}}]}}]}})
    tc_seq = log.append(rid, TOOL_CALL, {"tool_call_id": "c1", "name": "read_file",
                                         "result": "HUGE FILE BODY " * 100})
    log.append(rid, MODEL_CALL, {"call_index": 1, "response": {"choices": [
        {"message": {"role": "assistant", "content": "done"}}]}})

    agent = Agent(None, ToolRegistry([]), log, capabilities=Capabilities())
    before, _, _ = agent._reconstruct(rid, "task")
    assert "HUGE FILE BODY" in [m.content for m in before if m.role == "tool"][0]

    log.append(rid, MESSAGE_SNIPPED, {"seq": tc_seq})
    after, _, _ = agent._reconstruct(rid, "task")
    tool_msg = [m for m in after if m.role == "tool"][0]
    assert tool_msg.content == "[snipped to free context]"   # collapsed
    # lossless: the original event is still in the log
    assert any("HUGE FILE BODY" in (e.payload.get("result") or "")
               for e in log.events(rid, type=TOOL_CALL))


# --- /review presets -------------------------------------------------------

def test_review_presets_are_read_only():
    from local_harness.agent.presets import get_preset
    for name in ("review", "security-review"):
        p = get_preset(name)
        assert p.name == name
        exposed = p.exposed()
        assert exposed is not None and "write_file" not in exposed and "read_file" in exposed
    assert "code reviewer" in get_preset("review").system_prompt
    assert "security" in get_preset("security-review").system_prompt.lower()


# --- MCP http transport + auth ---------------------------------------------

def _mcp_handler(token: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if token is not None and request.headers.get("Authorization") != f"Bearer {token}":
            return httpx.Response(401, json={"error": "unauthorized"})
        body = json.loads(request.content)
        rid, method = body["id"], body["method"]
        if method == "tools/list":
            result = {"tools": [{"name": "echo", "description": "Echo back text",
                                 "inputSchema": {"type": "object",
                                                 "properties": {"text": {"type": "string"}}}}]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": "echoed"}]}
        else:
            result = {}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": rid, "result": result})
    return handler


@pytest.mark.asyncio
async def test_http_transport_sends_bearer_and_lists_tools():
    send = http_transport("http://mcp", token="secret",
                          transport=httpx.MockTransport(_mcp_handler(token="secret")))
    client = MCPClient(send)
    await client.initialize()
    assert (await client.list_tools())[0]["name"] == "echo"
    assert await client.call_tool("echo", {"text": "hi"}) == "echoed"


@pytest.mark.asyncio
async def test_http_transport_401_without_token():
    send = http_transport("http://mcp", token=None,
                          transport=httpx.MockTransport(_mcp_handler(token="secret")))
    with pytest.raises(MCPError):
        await MCPClient(send).initialize()


@pytest.mark.asyncio
async def test_load_routes_http_mcp_and_resolves_token(monkeypatch):
    captured = {}

    def fake_http_transport(url, token=None, transport=None):
        captured["url"], captured["token"] = url, token
        return http_transport(url, token=token, transport=httpx.MockTransport(_mcp_handler()))

    monkeypatch.setattr("local_harness.integrations.load.http_transport", fake_http_transport)
    monkeypatch.setenv("MY_MCP_TOKEN", "secret")
    reg = await registry_with_sources(
        {"mcp": [{"url": "http://mcp", "token_env": "MY_MCP_TOKEN", "namespace": "svc"}]})
    assert captured == {"url": "http://mcp", "token": "secret"}
    names = {t["function"]["name"] for t in reg.schemas()}
    assert "svc.echo" in names
    assert "svc.echo" in reg.deferrable_names()  # external tools are deferrable


# --- vim word motion -------------------------------------------------------

def test_vim_word_motion():
    from local_harness.tui.app import HarnessApp
    w = HarnessApp._vim_word
    # forward: jump to the start of the next word
    assert w("hello world", 0, 1) == 6
    assert w("hello world", 6, 1) == 11
    # backward: jump to the start of the previous word
    assert w("hello world", 11, -1) == 6
    assert w("hello world", 6, -1) == 0
