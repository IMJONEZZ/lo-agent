"""Adapter protocol: map the abstract capability API onto one server family.

Adapters contribute the *static* part of a Capabilities report (what we know
this server family supports, e.g. llama.cpp's sampler zoo). The prober then
verifies the *dynamic* part (seed determinism, logprobs shape, endpoint
existence) with live test requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Fingerprint:
    """Cheap, read-only observations about an endpoint."""

    owned_by: str | None = None          # from /v1/models data[0].owned_by
    has_props: bool = False              # llama.cpp GET /props
    has_slots: bool = False              # llama.cpp GET /slots (requires --slots)
    props: dict[str, Any] = field(default_factory=dict)
    model_card: dict[str, Any] = field(default_factory=dict)  # /v1/models data[0] (vLLM exposes max_model_len here)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class StaticCaps:
    server: str = "generic"
    grammar: str | None = None           # "gbnf" | "guided" | None
    logit_bias: bool = False
    sampler_zoo: set[str] = field(default_factory=set)
    cfg_scale: bool = False
    banned_strings: bool = False
    parallel_n: bool = False
    stream_logprobs: bool = False  # returns logprobs in a streamed tool-call request
    # Token probabilities computed AFTER the sampling chain (llama.cpp
    # `post_sampling_probs`) — confidence over what could actually be sampled,
    # not the raw distribution truncation samplers no longer draw from.
    post_sampling_probs: bool = False


class Adapter:
    @classmethod
    def matches(cls, fp: Fingerprint) -> bool:
        raise NotImplementedError

    @classmethod
    def static_caps(cls, fp: Fingerprint) -> StaticCaps:
        raise NotImplementedError

    @classmethod
    def prepare_body(cls, body: dict[str, Any]) -> dict[str, Any]:
        """Server-specific request tweaks. Default: pass through."""
        return body
