"""Best-of-N with pluggable verifiers — free-tokens compute scaling.

On vLLM (parallel_n) all N candidates come from one request; on llama.cpp
they are sequential but each fork is a prefix-cache hit, so marginal cost is
only the completion tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...inference.capabilities import Capabilities
from ...inference.client import OpenAICompatClient
from ...inference.types import (
    GenerationRequest,
    GenerationResponse,
    Message,
    SamplingParams,
    TokenLogprob,
)
from ...signals.metrics import StepSignals


def _choice_logprobs(choice: dict) -> list[TokenLogprob] | None:
    """Per-choice token logprobs from an OpenAI-style choice dict (the n>1
    parallel path returns one such dict per candidate)."""
    lp = choice.get("logprobs")
    if not lp or not lp.get("content"):
        return None
    return [
        TokenLogprob(
            token=t["token"],
            logprob=t["logprob"],
            top=[(x["token"], x["logprob"]) for x in t.get("top_logprobs") or []],
        )
        for t in lp["content"]
    ]


@dataclass
class Candidate:
    text: str
    seed: int | None
    score: float = 0.0
    logprobs: list[TokenLogprob] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Verifier(Protocol):
    async def score(self, candidate: Candidate) -> float: ...


class MeanLogprobVerifier:
    """Model confidence as the score — zero extra inference."""

    async def score(self, candidate: Candidate) -> float:
        signals = StepSignals.from_logprobs(candidate.logprobs or [])
        return signals.mean_logprob if signals else float("-inf")


class SkillValidityVerifier:
    """Grammar validity gate with confidence tiebreak."""

    def __init__(self, skill):
        self.skill = skill

    async def score(self, candidate: Candidate) -> float:
        valid = self.skill.validate_output(candidate.text.strip())
        signals = StepSignals.from_logprobs(candidate.logprobs or [])
        conf = signals.mean_logprob if signals else -10.0
        return (100.0 + conf) if valid else (-100.0 + conf)


class JudgeVerifier:
    """LLM-as-judge: a 0-9 rating from the same endpoint, parsed from free
    generation. We deliberately do NOT grammar-constrain the digit: on reasoning
    models a single-char grammar never binds — the model reasons past any budget
    and the constrained content never lands (empty → score 0). Instead we let it
    reason and answer, then parse the rating from content-or-reasoning."""

    _RATING = re.compile(r"RATING:\s*([0-9])")
    _DIGIT = re.compile(r"\b([0-9])\b")

    def __init__(self, client: OpenAICompatClient, caps: Capabilities, rubric: str):
        self.client, self.caps, self.rubric = client, caps, rubric

    async def score(self, candidate: Candidate) -> float:
        prompt = (
            f"Rubric: {self.rubric}\n\nCandidate answer:\n{candidate.text[:1500]}\n\n"
            "Rate how well the candidate meets the rubric, 0 (worst) to 9 (best). "
            "End your reply with exactly:  RATING: <digit>"
        )
        # Generous budget + enable_thinking off: reasoning models reason in-channel
        # and still land a parseable rating; non-reasoning models answer immediately.
        resp = await self.client.chat(GenerationRequest(
            messages=[Message(role="user", content=prompt)],
            sampling=SamplingParams(
                temperature=0.0, max_tokens=2048,
                extra={"chat_template_kwargs": {"enable_thinking": False}})))
        msg = resp.raw.get("choices", [{}])[0].get("message", {})
        text = " ".join(s for s in (msg.get("content"), msg.get("reasoning_content"),
                                    msg.get("reasoning")) if s)
        hits = self._RATING.findall(text) or self._DIGIT.findall(text)
        return float(hits[-1]) if hits else 0.0


async def best_of_n(
    client: OpenAICompatClient,
    caps: Capabilities,
    messages: list[Message],
    verifier: Verifier,
    n: int = 4,
    sampling: SamplingParams | None = None,
    base_seed: int = 100,
) -> list[Candidate]:
    """Sample n candidates, score with the verifier, return sorted best-first."""
    sampling = sampling or SamplingParams(temperature=0.8, max_tokens=256)
    sampling.logprobs = caps.logprobs
    candidates: list[Candidate] = []

    if caps.parallel_n:
        body = GenerationRequest(messages=messages, sampling=sampling).to_body(client.model)
        body["n"] = n
        body["seed"] = base_seed
        response = await client.chat_body(body)
        for choice in response.raw["choices"]:
            msg = choice.get("message", {})
            candidates.append(Candidate(
                text=(msg.get("content") or msg.get("reasoning_content")
                      or msg.get("reasoning") or "").strip(),
                seed=base_seed,
                logprobs=_choice_logprobs(choice),  # per-candidate, so verifiers can rank
                raw=choice,
            ))
    else:
        for i in range(n):
            sampling.seed = base_seed + i
            response: GenerationResponse = await client.chat(
                GenerationRequest(messages=messages, sampling=sampling)
            )
            candidates.append(Candidate(
                text=response.text.strip(), seed=sampling.seed,
                logprobs=response.logprobs, raw=response.raw,
            ))

    for c in candidates:
        c.score = await verifier.score(c)
    return sorted(candidates, key=lambda c: c.score, reverse=True)
