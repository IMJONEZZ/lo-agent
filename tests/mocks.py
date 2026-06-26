"""Mock OpenAI-compatible servers built on httpx.MockTransport.

MockLlamaCpp imitates a llama.cpp server (Tier 3 surface: /props, /slots,
seeded determinism, logprobs, raw completion). Responses are a pure function
of the request seed, so probe determinism checks and replay verification
behave exactly as they would against a real Tier-1+ server.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


def chat_response(
    content: str | None = None,
    tool_calls: list[tuple[str, str, str]] | None = None,  # (id, name, arguments)
    with_logprobs: bool = True,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": tc_id, "type": "function", "function": {"name": name, "arguments": args}}
            for tc_id, name, args in tool_calls
        ]
    choice: dict[str, Any] = {
        "index": 0,
        "message": msg,
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }
    if with_logprobs:
        choice["logprobs"] = {
            "content": [
                {"token": "x", "logprob": -0.05, "top_logprobs": [{"token": "x", "logprob": -0.05}]},
                {"token": "y", "logprob": -0.50, "top_logprobs": [{"token": "y", "logprob": -0.50}]},
            ]
        }
    return {
        "id": "cmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def _sse_from_chat(resp: dict[str, Any], with_logprobs: bool = True) -> str:
    """Turn a chat_response dict into an OpenAI-style SSE stream, so the mock
    can exercise the client's streaming path (chat_body_stream). Logprobs are
    streamed only when the request asked for them (with_logprobs), mirroring a
    real server — so a tool-calling turn (which strips logprobs) yields none."""
    choice = resp["choices"][0]
    msg = choice["message"]
    events = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
    content = msg.get("content") or ""
    for i in range(0, len(content), 4):
        events.append({"choices": [{"index": 0, "delta": {"content": content[i:i + 4]}}]})
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        events.append({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": tc["id"], "type": "function",
             "function": {"name": fn["name"], "arguments": fn["arguments"]}}]}}]})
    lp = (choice.get("logprobs") or {}).get("content")
    if with_logprobs and lp:
        events.append({"choices": [{"index": 0, "delta": {}, "logprobs": {"content": lp}}]})
    events.append({"choices": [{"index": 0, "delta": {},
                               "finish_reason": choice.get("finish_reason", "stop")}],
                   "usage": resp.get("usage", {})})
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return body + "data: [DONE]\n\n"


def mock_token_id(text: str) -> int:
    """Deterministic fake tokenizer: first 'token' id of a string."""
    return 1000 + sum(text.lstrip().encode()[:4])


class MockLlamaCpp:
    def __init__(
        self,
        script: dict[int, dict[str, Any]] | None = None,
        fail_after: int | None = None,
        completion_fn=None,  # (prompt: str, body: dict) -> (text, finish_reason)
        slot_save_enabled: bool = True,
    ):
        self.script = script  # seed -> chat response; None = echo the seed
        self.fail_after = fail_after  # raise ConnectError once this many chat calls succeeded
        self.chat_calls = 0
        self.completion_calls = 0
        self.completion_fn = completion_fn
        self.slot_save_enabled = slot_save_enabled
        self.saved_slots: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model", "owned_by": "llamacpp"}]})
        if path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {}})
        if path == "/slots":
            return httpx.Response(200, json=[{"id": 0}])
        if path.startswith("/slots/"):
            if not self.slot_save_enabled:
                return httpx.Response(400, json={"error": "no --slot-save-path"})
            body = json.loads(request.content)
            self.saved_slots.append(body.get("filename", ""))
            return httpx.Response(200, json={"id_slot": 0})
        if path == "/tokenize":
            content = json.loads(request.content).get("content", "")
            return httpx.Response(200, json={"tokens": [mock_token_id(content)]})
        if path == "/apply-template":
            msgs = json.loads(request.content).get("messages", [])
            prompt = "".join(f"<{m['role']}>{m.get('content') or ''}" for m in msgs) + "<assistant>"
            return httpx.Response(200, json={"prompt": prompt})
        if path == "/v1/completions":
            self.completion_calls += 1
            body = json.loads(request.content)
            if self.completion_fn is not None:
                text, finish = self.completion_fn(body.get("prompt", ""), body)
            else:
                text, finish = " upon", "stop"
            return httpx.Response(200, json={
                "choices": [{"text": text, "finish_reason": finish}],
                "usage": {"completion_tokens": max(1, len(text.split()))},
            })
        if path == "/v1/chat/completions":
            if self.fail_after is not None and self.chat_calls >= self.fail_after:
                raise httpx.ConnectError("mock server crashed")
            self.chat_calls += 1
            body = json.loads(request.content)
            seed = body.get("seed", 0)
            resp = self.script[seed] if self.script is not None \
                else chat_response(content=f"deterministic-{seed}")
            if body.get("stream"):
                return httpx.Response(
                    200, text=_sse_from_chat(resp, with_logprobs=bool(body.get("logprobs"))),
                    headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=resp)
        return httpx.Response(404)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class MockGeneric:
    """Tier-0 endpoint: no extensions, ignores seed, no logprobs."""

    def __init__(self):
        self.counter = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model", "owned_by": "acme"}]})
        if path == "/v1/chat/completions":
            self.counter += 1
            return httpx.Response(
                200, json=chat_response(content=f"nondeterministic-{self.counter}", with_logprobs=False)
            )
        return httpx.Response(404)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
