"""Universal Tool-Calling Protocol (UTCP) — the harness's default tool protocol.

A UTCP *manual* describes tools and how to call each one *directly* via its
native transport (HTTP, CLI, ...) — no wrapper server in the loop, unlike MCP.
We read a manual and turn each tool into a harness Tool whose async fn performs
the real call. HTTP and CLI transports are supported.

Manual shape (a pragmatic subset of the UTCP spec)::

    {"tools": [
      {"name": "get_weather",
       "description": "...",
       "inputs": {"type": "object", "properties": {"city": {"type": "string"}}},
       "tool_call_template": {"call_template_type": "http",
                              "url": "https://api/weather", "http_method": "GET"}}]}
"""

from __future__ import annotations

import json
import shlex
from typing import Any

import httpx

from ..agent.tools import Tool


class UTCPClient:
    def __init__(self, http: httpx.AsyncClient | None = None, timeout: float = 30.0):
        self._http = http or httpx.AsyncClient(timeout=timeout)

    def tools_from_manual(self, manual: dict[str, Any], namespace: str = "") -> list[Tool]:
        out: list[Tool] = []
        for spec in manual.get("tools", []):
            name = spec["name"]
            tmpl = spec.get("tool_call_template") or spec.get("call_template") or {}
            out.append(Tool(
                name=f"{namespace}.{name}" if namespace else name,
                description=spec.get("description", ""),
                parameters=spec.get("inputs") or {"type": "object", "properties": {}},
                fn=self._make_caller(tmpl),
            ))
        return out

    def _make_caller(self, tmpl: dict[str, Any]):
        kind = tmpl.get("call_template_type") or tmpl.get("type")
        if kind == "http":
            url = tmpl["url"]
            method = (tmpl.get("http_method") or "POST").upper()

            async def call_http(**kwargs):
                if method == "GET":
                    resp = await self._http.request(method, url, params=kwargs)
                else:
                    resp = await self._http.request(method, url, json=kwargs)
                resp.raise_for_status()
                return resp.text

            return call_http
        if kind == "cli":
            base = tmpl["command"]

            async def call_cli(**kwargs):
                import asyncio

                cmd = base if isinstance(base, list) else shlex.split(base)
                cmd = [*cmd, json.dumps(kwargs)]
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                out, _ = await proc.communicate()
                return out.decode().strip()

            return call_cli
        raise ValueError(f"unsupported UTCP call template: {kind!r}")

    async def aclose(self) -> None:
        await self._http.aclose()


def register_utcp(registry, manual: dict[str, Any], *, namespace: str = "",
                  http: httpx.AsyncClient | None = None) -> list[str]:
    """Load a UTCP manual and register its tools. Returns the registered names."""
    client = UTCPClient(http=http)
    tools = client.tools_from_manual(manual, namespace=namespace)
    for t in tools:
        registry.register(t)
    return [t.name for t in tools]
