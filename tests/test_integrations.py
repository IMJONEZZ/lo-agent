"""UTCP (default) + MCP tool protocols: tools register into the ToolRegistry
and execute through the normal (now async) tool path."""

import json

import httpx

from local_harness.agent.tools import ToolRegistry
from local_harness.integrations.load import registry_with_sources
from local_harness.integrations.mcp import MCPClient, register_mcp
from local_harness.integrations.utcp import UTCPClient, register_utcp


# --- UTCP --------------------------------------------------------------------

async def test_utcp_http_tool_calls_its_endpoint():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, text="sunny, 21C")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manual = {"tools": [{
        "name": "get_weather", "description": "weather by city",
        "inputs": {"type": "object", "properties": {"city": {"type": "string"}}},
        "tool_call_template": {"call_template_type": "http",
                               "url": "http://api/weather", "http_method": "GET"}}]}
    reg = ToolRegistry()
    names = register_utcp(reg, manual, http=http)

    assert names == ["get_weather"]
    out = await reg.execute("get_weather", '{"city": "Paris"}')
    assert out == "sunny, 21C"                       # the real endpoint response
    assert seen["params"] == {"city": "Paris"}       # args went through as query params
    await http.aclose()


def test_utcp_schema_is_openai_shaped():
    manual = {"tools": [{"name": "t", "description": "d",
                         "inputs": {"type": "object", "properties": {"x": {"type": "string"}}},
                         "tool_call_template": {"call_template_type": "http",
                                                "url": "http://x", "http_method": "POST"}}]}
    reg = ToolRegistry()
    register_utcp(reg, manual, namespace="weather")
    schema = reg.schemas()[0]
    assert schema["function"]["name"] == "weather.t"   # namespaced, collision-safe
    assert schema["function"]["parameters"]["properties"] == {"x": {"type": "string"}}


async def test_registry_with_sources_includes_builtins_and_utcp():
    manual = {"namespace": "weather", "tools": [{
        "name": "now", "description": "current weather",
        "inputs": {"type": "object", "properties": {}},
        "tool_call_template": {"call_template_type": "http", "url": "http://x"}}]}
    reg = await registry_with_sources({"utcp": [manual]})
    names = {s["function"]["name"] for s in reg.schemas()}
    assert "calculator" in names and "bash" in names  # builtins still present
    assert "weather.now" in names                       # UTCP tool loaded


# --- MCP ---------------------------------------------------------------------

def _fake_mcp_server():
    """A minimal in-process MCP server speaking JSON-RPC over the transport."""
    async def send(request):
        method = request["method"]
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": request["id"], "result": {"protocolVersion": "2024-11-05"}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request["id"], "result": {"tools": [
                {"name": "add", "description": "add two ints",
                 "inputSchema": {"type": "object",
                                 "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}]}}
        if method == "tools/call":
            args = request["params"]["arguments"]
            total = args["a"] + args["b"]
            return {"jsonrpc": "2.0", "id": request["id"],
                    "result": {"content": [{"type": "text", "text": str(total)}]}}
        return {"jsonrpc": "2.0", "id": request["id"], "error": {"message": f"no {method}"}}
    return send


async def test_mcp_tool_round_trips_and_executes():
    reg = ToolRegistry()
    names = await register_mcp(reg, MCPClient(_fake_mcp_server()))
    assert names == ["add"]
    assert await reg.execute("add", '{"a": 2, "b": 5}') == "7"   # JSON-RPC tools/call result


async def test_mcp_error_surfaces_to_model():
    async def send(request):
        if request["method"] in ("initialize", "tools/list"):
            return await _fake_mcp_server()(request)
        return {"jsonrpc": "2.0", "id": request["id"],
                "result": {"isError": True, "content": [{"type": "text", "text": "bad args"}]}}
    reg = ToolRegistry()
    await register_mcp(reg, MCPClient(send))
    out = await reg.execute("add", '{"a": 1, "b": 2}')
    assert out.startswith("error:") and "bad args" in out
