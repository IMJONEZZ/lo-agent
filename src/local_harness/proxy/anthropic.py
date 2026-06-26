"""Anthropic Messages API <-> OpenAI chat-completions translation.

Lets Anthropic-native clients (Claude Code among them) talk to any local
model through the proxy: /v1/messages requests are translated to OpenAI
shape, run through the engine (pipeline + guardrails), and translated back —
including a buffered Anthropic SSE stream for stream=true.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator


def anthropic_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"model": body.get("model", "")}
    messages: list[dict[str, Any]] = []

    system = body.get("system")
    if system:
        if isinstance(system, list):  # list of text blocks
            system = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                texts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block["id"], "type": "function",
                    "function": {"name": block["name"],
                                 "arguments": json.dumps(block.get("input", {}))},
                })
            elif btype == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, list):
                    rc = "\n".join(b.get("text", "") for b in rc if isinstance(b, dict))
                messages.append({"role": "tool", "content": rc,
                                 "tool_call_id": block.get("tool_use_id", "")})
            # thinking blocks from prior turns are dropped: local models
            # regenerate their own reasoning
        if role == "assistant" and tool_calls:
            messages.append({"role": "assistant",
                             "content": "\n".join(texts) or None,
                             "tool_calls": tool_calls})
        elif texts:
            messages.append({"role": role, "content": "\n".join(texts)})

    out["messages"] = messages
    if body.get("tools"):
        out["tools"] = [
            {"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("input_schema", {"type": "object"})}}
            for t in body["tools"]
        ]
    for src, dst in (("max_tokens", "max_tokens"), ("temperature", "temperature"),
                     ("top_p", "top_p"), ("stop_sequences", "stop")):
        if src in body:
            out[dst] = body[src]
    if "harness" in body:
        out["harness"] = body["harness"]
    return out


_STOP_REASON = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


def openai_to_anthropic(resp: dict[str, Any], requested_model: str) -> dict[str, Any]:
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    content: list[dict[str, Any]] = []

    if message.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": message["reasoning_content"],
                        "signature": ""})
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append({"type": "tool_use", "id": tc.get("id", uuid.uuid4().hex[:12]),
                        "name": fn.get("name", ""), "input": args})

    usage = resp.get("usage") or {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": _STOP_REASON.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def anthropic_sse(message: dict[str, Any]) -> Iterator[str]:
    """Buffered Anthropic SSE: the full response emitted as a valid event
    sequence (one delta per content block)."""
    skeleton = {**message, "content": []}
    yield _event("message_start", {"type": "message_start", "message": skeleton})
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            yield _event("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "text", "text": ""}})
            yield _event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": block["text"]}})
        elif block["type"] == "thinking":
            yield _event("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "thinking", "thinking": ""}})
            yield _event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "thinking_delta", "thinking": block["thinking"]}})
        else:  # tool_use
            yield _event("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "tool_use", "id": block["id"],
                                  "name": block["name"], "input": {}}})
            yield _event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(block["input"])}})
        yield _event("content_block_stop", {"type": "content_block_stop", "index": index})
    yield _event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
        "usage": {"output_tokens": message["usage"]["output_tokens"]}})
    yield _event("message_stop", {"type": "message_stop"})


def openai_sse(resp: dict[str, Any]) -> Iterator[str]:
    """Buffered OpenAI SSE: role chunk, one content/tool_calls delta, finish."""
    base = {"id": resp.get("id", "harness"), "object": "chat.completion.chunk",
            "created": resp.get("created", int(time.time())), "model": resp.get("model", "")}
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})

    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    delta: dict[str, Any] = {}
    if message.get("content"):
        delta["content"] = message["content"]
    if message.get("reasoning_content"):
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": i, **tc} for i, tc in enumerate(message["tool_calls"])
        ]
    if delta:
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice.get('finish_reason', 'stop')}]})}\n\n"
    yield "data: [DONE]\n\n"
