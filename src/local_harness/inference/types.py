"""Shared request/response types for the inference layer.

Everything here serializes to plain JSON dicts so the event log (pillar A)
can store and replay requests byte-for-byte.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def canonical_text(message: dict[str, Any]) -> str:
    """Canonical comparable view of a raw assistant message dict: content,
    parsed-out reasoning (reasoning models), and tool calls. Used by the
    capability prober's determinism check and by replay verification.

    Reasoning parsers disagree on the field name: llama.cpp/Qwen use
    `reasoning_content`; vLLM's parser for some models (e.g. stepfun Step)
    uses `reasoning`. Read both so a short probe that lands entirely in the
    thinking channel isn't mistaken for an empty (non-deterministic) response."""
    parts = [
        message.get("content") or "",
        message.get("reasoning_content") or message.get("reasoning") or "",
    ]
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        parts.append(f"{fn.get('name')}({fn.get('arguments')})")
    return "|".join(parts)


@dataclass
class ToolCallRequest:
    """A tool call the model asked for."""

    id: str
    name: str
    arguments: str  # raw JSON string, as returned by the server

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCallRequest":
        fn = d.get("function", {})
        return cls(id=d.get("id", ""), name=fn.get("name", ""), arguments=fn.get("arguments", ""))


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role=tool messages
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        # OpenAI-compat servers expect content present (possibly null) on assistant
        # tool-call messages and a string everywhere else.
        d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content"),
            tool_calls=[ToolCallRequest.from_dict(tc) for tc in d.get("tool_calls") or []],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


@dataclass
class SamplingParams:
    """Normalized sampling parameters.

    `extra` is the escape hatch for server-specific params (llama.cpp grammar,
    cache_prompt, dry_multiplier, ...). Adapters and later the logit pipeline
    populate it; the client merges it into the request body verbatim.
    """

    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int | None = None  # None = no cap; let the server decide (don't ration locally)
    seed: int | None = None
    logprobs: bool = False
    top_logprobs: int = 4
    extra: dict[str, Any] = field(default_factory=dict)

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:  # omit entirely unless a cap was asked for
            body["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.seed is not None:
            body["seed"] = self.seed
        if self.logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        body.update(self.extra)
        return body


@dataclass
class GenerationRequest:
    messages: list[Message]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    tools: list[dict[str, Any]] = field(default_factory=list)  # OpenAI tool schemas

    def to_body(self, model: str) -> dict[str, Any]:
        body = {"model": model, "messages": [m.to_dict() for m in self.messages]}
        if self.tools:
            body["tools"] = self.tools
        body.update(self.sampling.to_body())
        return body


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    top: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class GenerationResponse:
    message: Message
    finish_reason: str | None
    logprobs: list[TokenLogprob] | None
    usage: dict[str, Any]
    timing_ms: float
    raw: dict[str, Any]  # full server response, stored in the event log

    @property
    def text(self) -> str:
        return self.message.content or ""

    @classmethod
    def from_chat_response(cls, data: dict[str, Any], timing_ms: float) -> "GenerationResponse":
        choice = data["choices"][0]
        message = Message.from_dict(choice["message"])
        logprobs = None
        lp = choice.get("logprobs")
        if lp and lp.get("content"):
            # llama.cpp with `post_sampling_probs: true` reports linear `prob`
            # (and `top_probs`) instead of `logprob`/`top_logprobs`; normalize
            # both shapes into log-space so every consumer stays unchanged.
            def _lp(x: dict) -> float:
                if "logprob" in x:
                    return x["logprob"]
                return math.log(max(x.get("prob", 0.0), 1e-12))

            logprobs = [
                TokenLogprob(
                    token=t["token"],
                    logprob=_lp(t),
                    top=[
                        (x["token"], _lp(x))
                        for x in t.get("top_logprobs") or t.get("top_probs") or []
                    ],
                )
                for t in lp["content"]
            ]
        return cls(
            message=message,
            finish_reason=choice.get("finish_reason"),
            logprobs=logprobs,
            usage=data.get("usage") or {},
            timing_ms=timing_ms,
            raw=data,
        )

    @classmethod
    def from_responses_api(cls, data: dict[str, Any], timing_ms: float) -> "GenerationResponse":
        """Parse an Open Responses (/v1/responses) payload — LM Studio surfaces token
        logprobs here, nested in output[].content[].logprobs (output_text blocks),
        with reasoning in separate reasoning_text blocks. We reshape it into the chat
        response shape so every downstream caller (signals, demos, search) is unchanged."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        lp_content: list[dict] = []
        for item in data.get("output") or []:
            for block in item.get("content") or []:
                btype = block.get("type")
                if btype == "output_text":
                    content_parts.append(block.get("text") or "")
                    lp_content.extend(block.get("logprobs") or [])
                elif btype == "reasoning_text":
                    reasoning_parts.append(block.get("text") or "")
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
        if reasoning_parts:
            msg["reasoning_content"] = "".join(reasoning_parts)
        choice: dict[str, Any] = {"index": 0, "message": msg, "finish_reason": data.get("status")}
        if lp_content:
            choice["logprobs"] = {"content": lp_content}
        raw = {"choices": [choice], "usage": data.get("usage") or {}, "_responses": data}
        return cls.from_chat_response(raw, timing_ms)
