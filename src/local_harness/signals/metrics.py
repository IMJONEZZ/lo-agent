"""Confidence metrics from per-token logprobs — the agent loop's nervous system."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..inference.types import TokenLogprob


def answer_logprobs(toks: list) -> list:
    """Drop chain-of-thought tokens (everything up to the last </think>) and any
    trailing special tokens, so a confidence overlay weights the ANSWER, not the
    reasoning. Reasoning models sometimes emit `<think>…</think>answer` inline in
    the content channel, which otherwise pollutes the overlay."""
    if not toks:
        return []
    joined = "".join(getattr(t, "token", "") for t in toks)
    marker = joined.rfind("</think>")
    start = marker + len("</think>") if marker >= 0 else 0
    out, off = [], 0
    for t in toks:
        tok = getattr(t, "token", "")
        if off >= start and tok.strip() and "<|" not in tok and "</think>" not in tok:
            out.append(t)
        off += len(tok)
    return out or toks


@dataclass
class StepSignals:
    n_tokens: int
    mean_logprob: float
    min_logprob: float
    mean_entropy: float      # estimated from top-k logprobs + tail mass
    mean_top2_margin: float  # avg logprob gap between best and runner-up
    # True when the values come from post-sampling probabilities (llama.cpp
    # `post_sampling_probs`) — i.e. confidence over the truncated distribution
    # actually sampled from, not the raw model distribution. Post-sampling
    # values run higher (survivors are renormalized); threshold accordingly.
    post_sampling: bool = False

    @classmethod
    def from_logprobs(
        cls, logprobs: list[TokenLogprob], *, post_sampling: bool = False
    ) -> "StepSignals | None":
        if not logprobs:
            return None
        lps = [t.logprob for t in logprobs]
        entropies, margins = [], []
        for t in logprobs:
            if len(t.top) >= 2:
                margins.append(t.top[0][1] - t.top[1][1])
            if t.top:
                probs = [math.exp(lp) for _, lp in t.top]
                tail = max(0.0, 1.0 - sum(probs))
                h = -sum(p * math.log(max(p, 1e-12)) for p in probs)
                if tail > 1e-6:
                    # treat the unseen tail as one pseudo-token
                    h += -tail * math.log(tail)
                entropies.append(h)
        return cls(
            n_tokens=len(lps),
            mean_logprob=sum(lps) / len(lps),
            min_logprob=min(lps),
            mean_entropy=sum(entropies) / len(entropies) if entropies else 0.0,
            mean_top2_margin=sum(margins) / len(margins) if margins else 0.0,
            post_sampling=post_sampling,
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()
