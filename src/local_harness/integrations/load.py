"""Build a ToolRegistry from builtins plus configured UTCP manuals and MCP servers.

Config shape (e.g. a tools.json)::

    {"utcp": ["./weather_manual.json", {<inline manual>}],
     "mcp":  [{"command": ["uvx", "mcp-server-git"], "namespace": "git"}]}

UTCP is the default protocol; MCP servers are supported alongside.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from ..agent.tools import ToolRegistry, builtin_tools
from .mcp import MCPClient, http_transport, register_mcp, stdio_transport
from .utcp import register_utcp


async def registry_with_sources(
    config: dict[str, Any] | None = None,
    *,
    http: httpx.AsyncClient | None = None,
    sandbox=None,
) -> ToolRegistry:
    registry = ToolRegistry(builtin_tools(sandbox=sandbox))
    core = set(registry.deferrable_names()) | {t["function"]["name"] for t in registry.schemas()}
    config = config or {}
    for manual in config.get("utcp", []):
        if isinstance(manual, (str, Path)):
            manual = json.loads(Path(manual).read_text())
        register_utcp(registry, manual, namespace=manual.get("namespace", ""), http=http)
    for server in config.get("mcp", []):
        if server.get("url"):  # remote/HTTP MCP server (optionally token-authed)
            import os
            token = server.get("token") or (
                os.environ.get(server["token_env"]) if server.get("token_env") else None)
            transport = http_transport(server["url"], token=token)
        else:
            transport = stdio_transport(server["command"])
        await register_mcp(registry, MCPClient(transport), namespace=server.get("namespace", ""))
    # everything added from external sources (MCP/UTCP) is deferrable; builtins aren't
    registry.set_deferrable({t["function"]["name"] for t in registry.schemas()} - core)
    return registry


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}
