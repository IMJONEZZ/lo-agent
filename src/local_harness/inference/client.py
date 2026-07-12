"""Async OpenAI-compatible client.

One client serves every backend; server-specific behavior lives in adapters
(which contribute static capabilities and request-body tweaks), not in
subclasses of this client.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

from .types import GenerationRequest, GenerationResponse


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # When the server exposes token logprobs only via /v1/responses (e.g. LM
        # Studio) and not on chat-completions, the prober sets this so logprob
        # requests transparently route there. See capabilities.probe().
        self.logprobs_via_responses = False
        # Rung 6: a paired lens service URL (set from config/env by callers); when
        # present, probe() reports activation/intervention capability (Tier 4).
        self.lens_url: str | None = None
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # A reasoning model can generate — and pre-fill a large context — for many
        # minutes; a non-streamed call returns no bytes until it finishes. So the
        # READ timeout is None by default: a long, healthy generation must never be
        # cut off (the failure that looked like a hang on slow 27B reasoning runs).
        # The connect timeout stays short so an unreachable server fails fast. Pass
        # a float to cap read time explicitly.
        to = httpx.Timeout(timeout, connect=15.0) if timeout is not None \
            else httpx.Timeout(None, connect=15.0)
        self._http = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=to, transport=transport
        )

    # Grammar/guided-decoding params the server applies on chat-completions; the
    # Responses logprobs path can't carry them, so a request using one stays on chat.
    _GRAMMAR_KEYS = ("grammar", "guided_grammar", "guided_json", "guided_regex",
                     "json_schema", "response_format")

    async def chat(self, request: GenerationRequest) -> GenerationResponse:
        extra = request.sampling.extra or {}
        if (self.logprobs_via_responses and request.sampling.logprobs
                and not request.tools
                and not any(k in extra for k in self._GRAMMAR_KEYS)):
            return await self._responses_chat(request)
        body = request.to_body(self.model)
        return await self.chat_body(body)

    async def _responses_chat(self, request: GenerationRequest) -> GenerationResponse:
        """Fetch a completion *with logprobs* via /v1/responses for servers that
        only expose them there (LM Studio). Reasoning models spend the budget on
        thinking first, so give enough headroom for a message block (which carries
        the logprobs) to actually appear."""
        sp = request.sampling
        # Reasoning models spend tokens thinking *before* the message block that carries
        # the logprobs. Don't suppress the reasoning — tokens are free locally — just give
        # it a big budget so the message (and its logprobs) always lands. With no cap
        # requested, give generous headroom rather than starving the message block.
        budget = max(sp.max_tokens or 4096, 64) + 4096
        body: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": m.role, "content": m.content or ""} for m in request.messages],
            "include": ["message.output_text.logprobs"],
            "top_logprobs": sp.top_logprobs,
            "max_output_tokens": budget,
        }
        if sp.temperature is not None:
            body["temperature"] = sp.temperature
        if sp.top_p is not None:
            body["top_p"] = sp.top_p
        if sp.seed is not None:
            body["seed"] = sp.seed
        start = time.monotonic()
        resp = await self._http.post("/v1/responses", json=body)
        resp.raise_for_status()
        timing_ms = (time.monotonic() - start) * 1000
        return GenerationResponse.from_responses_api(resp.json(), timing_ms)

    async def chat_body(self, body: dict[str, Any]) -> GenerationResponse:
        """Issue a chat completion from a prebuilt body.

        Replay (events/replay.py) uses this to re-send logged requests verbatim.
        """
        start = time.monotonic()
        resp = await self._http.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        timing_ms = (time.monotonic() - start) * 1000
        return GenerationResponse.from_chat_response(resp.json(), timing_ms)

    async def chat_body_stream(
        self, body: dict[str, Any], on_delta: Callable[[str, str], None]
    ) -> GenerationResponse:
        """Stream a chat completion, calling on_delta(kind, text) for each
        content/reasoning token, and return the fully assembled response.

        The assembled response matches the non-streaming shape exactly, so the
        event log and replay are identical whether a turn was streamed or not
        (replay re-sends the logged body, which never carries stream=true)."""
        start = time.monotonic()
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        logprobs_content: list[dict] = []
        finish: str | None = None
        usage: dict[str, Any] = {}
        send = {**body, "stream": True}
        async with self._http.stream("POST", "/v1/chat/completions", json=send) as resp:
            if resp.status_code >= 400:
                await resp.aread()  # read the body so the exception carries the server's reason
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                ch = (chunk.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                    on_delta("content", delta["content"])
                # reasoning models stream their chain-of-thought in `reasoning_content`
                # (llama.cpp, most vLLM) or `reasoning` (vLLM stepfun Step parser).
                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning_delta:
                    reasoning.append(reasoning_delta)
                    on_delta("reasoning", reasoning_delta)
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                lp = ch.get("logprobs")
                if lp and lp.get("content"):
                    logprobs_content.extend(lp["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
        timing_ms = (time.monotonic() - start) * 1000
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            msg["tool_calls"] = [
                {"id": s["id"], "type": "function",
                 "function": {"name": s["name"], "arguments": s["args"]}}
                for _, s in sorted(tool_calls.items())
            ]
        choice: dict[str, Any] = {"index": 0, "message": msg, "finish_reason": finish}
        if logprobs_content:
            choice["logprobs"] = {"content": logprobs_content}
        raw = {"choices": [choice], "usage": usage}
        return GenerationResponse.from_chat_response(raw, timing_ms)

    async def complete_raw(self, prompt: str, body_extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Raw /v1/completions — prefill, FIM, budget forcing (Tier 2 features)."""
        body: dict[str, Any] = {"model": self.model, "prompt": prompt}
        if body_extra:
            body.update(body_extra)
        resp = await self._http.post("/v1/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    async def responses(
        self,
        input: Any,
        include: list[str] | None = None,
        top_logprobs: int | None = 4,
        max_output_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call the Open Responses endpoint (/v1/responses) — LM Studio surfaces
        token logprobs here. `input` is a string or a messages array."""
        body: dict[str, Any] = {"model": self.model, "input": input}
        if include:
            body["include"] = include
        if top_logprobs is not None:
            body["top_logprobs"] = top_logprobs
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if extra:
            body.update(extra)
        resp = await self._http.post("/v1/responses", json=body)
        resp.raise_for_status()
        return resp.json()

    async def get(self, path: str) -> httpx.Response:
        """GET helper used by the capability prober (e.g. /props, /slots)."""
        return await self._http.get(path)

    async def post(self, path: str, json: dict[str, Any]) -> httpx.Response:
        return await self._http.post(path, json=json)

    async def list_models(self) -> list[str]:
        resp = await self._http.get("/v1/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "OpenAICompatClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
