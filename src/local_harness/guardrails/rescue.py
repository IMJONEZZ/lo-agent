"""Rescue parsing: recover tool calls from free-text model output.

Local models routinely emit a tool call as prose-wrapped or code-fenced JSON
instead of the native tool_calls channel. Rather than burn a retry, scan the
text for JSON objects shaped like a tool call and execute them.

Accepted shapes (forge's format plus the OpenAI-style one):
    {"tool": "name", "args": {...}}
    {"name": "name", "arguments": {...}}     # arguments may also be a JSON string

Design (after forge, MIT): strip code fences, brace-scan for balanced JSON
candidates, validate each against the known tool names.
"""

from __future__ import annotations

import json
import uuid

from ..inference.types import ToolCallRequest


def _try_parse(candidate: str, tool_names: list[str]) -> ToolCallRequest | None:
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    name = obj.get("tool") or obj.get("name")
    if not isinstance(name, str) or name not in tool_names:
        return None
    args = obj.get("args") if "tool" in obj else obj.get("arguments")
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return ToolCallRequest(
        id=f"rescued-{uuid.uuid4().hex[:8]}", name=name, arguments=json.dumps(args)
    )


def rescue_tool_calls(text: str, tool_names: list[str]) -> list[ToolCallRequest]:
    """Extract every valid tool call embedded in free text (may be empty)."""
    cleaned = text.replace("```json", "```").replace("```", "")
    found: list[ToolCallRequest] = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        end = None
        for j in range(i, len(cleaned)):
            ch = cleaned[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            i += 1
            continue
        call = _try_parse(cleaned[i : end + 1], tool_names)
        if call is not None:
            found.append(call)
            i = end + 1
        else:
            i += 1
    return found
