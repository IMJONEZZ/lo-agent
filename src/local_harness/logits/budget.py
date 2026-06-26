"""Reasoning-budget control (s1-style budget forcing) over plain HTTP.

Requires raw completion + a chat-template endpoint (llama.cpp /apply-template).
Strategy:
  1. Render the chat prompt, open a think block, generate with the budget as
     max_tokens and the think-close tag as a stop string.
  2. If the model closed its reasoning naturally, done. If the budget ran out,
     force-close with the think-end tag. To *extend* reasoning instead, append
     the forcing phrase ("Wait") and continue.
  3. Generate the visible answer after the closed think block.

This is generation-control via prefill — impossible on chat-only APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..inference.client import OpenAICompatClient
from ..inference.types import Message


@dataclass
class BudgetedResult:
    reasoning: str
    answer: str
    reasoning_tokens: int
    forced_close: bool
    extensions_used: int


async def apply_template(client: OpenAICompatClient, messages: list[Message]) -> str | None:
    """Render messages through the server's chat template (llama.cpp)."""
    resp = await client.post(
        "/apply-template", json={"messages": [m.to_dict() for m in messages]}
    )
    if resp.status_code != 200:
        return None
    return resp.json().get("prompt")


async def generate_with_think_budget(
    client: OpenAICompatClient,
    messages: list[Message],
    think_budget: int,
    answer_max_tokens: int = 512,
    think_open: str = "<think>",
    think_close: str = "</think>",
    extend: int = 0,
    extend_phrase: str = "Wait",
    seed: int | None = None,
    sampling_extra: dict[str, Any] | None = None,
) -> BudgetedResult:
    prompt = await apply_template(client, messages)
    if prompt is None:
        raise RuntimeError("server has no /apply-template endpoint; think budget needs llama.cpp")
    extra = {"temperature": 0.6, "cache_prompt": True, **(sampling_extra or {})}
    if seed is not None:
        extra["seed"] = seed

    # The template may already end with an opened think block; open one if not.
    if not prompt.rstrip().endswith(think_open):
        prompt = prompt + think_open
    reasoning = ""
    tokens_used = 0
    extensions_used = 0
    forced_close = False

    while True:
        out = await client.complete_raw(
            prompt + reasoning,
            {**extra, "max_tokens": max(1, think_budget - tokens_used), "stop": [think_close]},
        )
        choice = out["choices"][0]
        reasoning += choice.get("text", "")
        tokens_used += (out.get("usage") or {}).get("completion_tokens", 0)
        stopped_naturally = choice.get("finish_reason") == "stop"

        if stopped_naturally and extensions_used < extend and tokens_used < think_budget:
            reasoning += "\n" + extend_phrase  # s1 budget forcing: keep thinking
            extensions_used += 1
            continue
        if not stopped_naturally:
            forced_close = True  # budget exhausted: close the think block ourselves
        break

    answer_prompt = prompt + reasoning + think_close + "\n\n"
    out = await client.complete_raw(answer_prompt, {**extra, "max_tokens": answer_max_tokens})
    return BudgetedResult(
        reasoning=reasoning,
        answer=out["choices"][0].get("text", "").strip(),
        reasoning_tokens=tokens_used,
        forced_close=forced_close,
        extensions_used=extensions_used,
    )
