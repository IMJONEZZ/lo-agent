"""Free self-consistency: sample a step N times and take the consensus.

On a frontier API this is N× the cost; locally it's free (and on vLLM it's a
single parallel-n request). The agreement fraction doubles as a confidence
signal — low agreement means the step is genuinely uncertain.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from ...inference.capabilities import Capabilities
from ...inference.client import OpenAICompatClient
from ...inference.types import GenerationRequest, Message, SamplingParams


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _answer_of(msg: dict) -> str:
    return (msg.get("content") or msg.get("reasoning_content")
            or msg.get("reasoning") or "").strip()


async def collect_samples(
    client: OpenAICompatClient,
    caps: Capabilities,
    messages: list[Message],
    n: int = 5,
    sampling: SamplingParams | None = None,
    base_seed: int = 200,
) -> list[str]:
    """Draw N samples of the next step — the one sampling path shared by
    self-consistency (lexical voting) and semantic entropy (meaning clustering).

    On a vLLM `n>1` server this is a single parallel request; elsewhere it's N
    seeded calls. Either way it's free locally — the whole point."""
    sampling = sampling or SamplingParams(temperature=0.7, max_tokens=256)
    answers: list[str] = []
    if caps.parallel_n:
        body = GenerationRequest(messages=messages, sampling=sampling).to_body(client.model)
        body["n"] = n
        body["seed"] = base_seed
        resp = await client.chat_body(body)
        answers = [_answer_of(c.get("message", {})) for c in resp.raw["choices"]]
    else:
        for i in range(n):
            sampling.seed = base_seed + i
            r = await client.chat(GenerationRequest(messages=messages, sampling=sampling))
            answers.append(_answer_of(r.raw["choices"][0].get("message", {})))
    return [a for a in answers if a]


async def self_consistency(
    client: OpenAICompatClient,
    caps: Capabilities,
    messages: list[Message],
    n: int = 5,
    sampling: SamplingParams | None = None,
    base_seed: int = 200,
    key: Callable[[str], str] | None = None,
) -> tuple[str, float]:
    """Returns (consensus answer, agreement fraction in [0,1]).

    `key` extracts the comparable answer from each sample before voting (e.g. pull
    out the final number), so verbose models that wrap the same answer in different
    prose still register as agreeing. Defaults to whitespace/case normalization.

    This votes on *surface form*; for meaning-aware uncertainty see
    `signals.semantic_entropy` (Farquhar 2024), which clusters these same samples
    by entailment instead."""
    answers = await collect_samples(client, caps, messages, n, sampling, base_seed)
    if not answers:
        return "", 0.0
    norm = key or _normalize
    counts = Counter(norm(a) for a in answers)
    top_key, freq = counts.most_common(1)[0]
    representative = next(a for a in answers if norm(a) == top_key)
    return representative, freq / len(answers)
