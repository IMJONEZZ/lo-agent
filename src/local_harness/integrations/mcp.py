"""Model Context Protocol (MCP) — supported alongside UTCP.

An MCP server exposes tools over JSON-RPC 2.0 (commonly stdio). We initialize,
list its tools, and register each as a harness Tool whose async fn issues a
`tools/call`. The transport is injectable (an async `send(request) -> response`)
so it's testable without a subprocess; `stdio_transport()` provides the real one.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from ..agent.tools import Tool

Transport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class MCPError(RuntimeError):
    pass


class MCPClient:
    def __init__(self, send: Transport):
        self._send = send
        self._id = 0

    async def _rpc(self, method: str, params: dict | None = None) -> dict[str, Any]:
        self._id += 1
        resp = await self._send(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        )
        if resp.get("error"):
            raise MCPError(resp["error"].get("message", str(resp["error"])))
        return resp.get("result", {}) or {}

    async def initialize(self) -> dict[str, Any]:
        return await self._rpc("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "local_harness", "version": "0.1"}})

    async def list_tools(self) -> list[dict[str, Any]]:
        return (await self._rpc("tools/list")).get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        return _content_text(result)


def _content_text(result: dict[str, Any]) -> str:
    parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    text = "\n".join(p for p in parts if p)
    if result.get("isError"):
        return f"error: {text or json.dumps(result)}"
    return text or json.dumps(result)


async def register_mcp(registry, client: MCPClient, *, namespace: str = "") -> list[str]:
    """Initialize an MCP client, list its tools, and register them. Returns names."""
    await client.initialize()
    names: list[str] = []
    for spec in await client.list_tools():
        tool_name = spec["name"]

        def make_fn(_name: str):
            async def fn(**kwargs):
                return await client.call_tool(_name, kwargs)
            return fn

        reg_name = f"{namespace}.{tool_name}" if namespace else tool_name
        registry.register(Tool(
            name=reg_name,
            description=spec.get("description", ""),
            parameters=spec.get("inputSchema") or {"type": "object", "properties": {}},
            fn=make_fn(tool_name),
        ))
        names.append(reg_name)
    return names


def http_transport(url: str, token: str | None = None, transport=None) -> Transport:
    """JSON-RPC over HTTP POST — for remote/HTTP MCP servers. An optional bearer
    `token` authenticates to OAuth/token-gated servers; a 401 raises a clear error
    pointing at the config. (Full interactive OAuth is a future enhancement.)"""
    import httpx

    headers = {"content-type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async def send(request: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30, transport=transport) as c:
            resp = await c.post(url, json=request, headers=headers)
            if resp.status_code == 401:
                raise MCPError(
                    f"MCP server {url} requires authentication (401). Provide a token via "
                    "the 'token' or 'token_env' field in your --tools config.")
            resp.raise_for_status()
            return resp.json()

    return send


def stdio_transport(command: list[str]) -> Transport:
    """Real transport: spawn an MCP server and exchange newline-delimited
    JSON-RPC over its stdio. One subprocess per client."""
    state: dict[str, Any] = {}

    async def send(request: dict[str, Any]) -> dict[str, Any]:
        proc = state.get("proc")
        if proc is None:
            proc = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
            state["proc"] = proc
        proc.stdin.write((json.dumps(request) + "\n").encode())
        await proc.stdin.drain()
        line = await proc.stdout.readline()
        return json.loads(line.decode())

    return send
