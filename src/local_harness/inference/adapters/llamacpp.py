from __future__ import annotations

from typing import Any

from .base import Adapter, Fingerprint, StaticCaps

# Samplers llama.cpp's server accepts as request params (normalized names).
LLAMACPP_SAMPLER_ZOO = {
    "min_p",
    "typical_p",
    "top_k",
    "mirostat",
    "dynatemp",
    "dry",
    "xtc",
    "top_n_sigma",
    "repeat_penalty",
    "sampler_order",
}


class LlamaCppAdapter(Adapter):
    @classmethod
    def matches(cls, fp: Fingerprint) -> bool:
        return fp.has_props or (fp.owned_by or "").lower() in ("llamacpp", "llama.cpp")

    @classmethod
    def static_caps(cls, fp: Fingerprint) -> StaticCaps:
        return StaticCaps(
            server="llama.cpp",
            grammar="gbnf",
            logit_bias=True,
            sampler_zoo=set(LLAMACPP_SAMPLER_ZOO),
            cfg_scale=False,
            banned_strings=False,
            parallel_n=False,
            post_sampling_probs=cls._supports_post_sampling_probs(fp),
        )

    @classmethod
    def _supports_post_sampling_probs(cls, fp: Fingerprint) -> bool:
        """Recent llama.cpp servers list `post_sampling_probs` among the default
        sampling params in /props — presence of the key means the server can
        return per-token probabilities computed after the sampler chain."""
        gen = fp.props.get("default_generation_settings") or {}
        if "post_sampling_probs" in gen:
            return True
        params = gen.get("params")
        return isinstance(params, dict) and "post_sampling_probs" in params

    @classmethod
    def prepare_body(cls, body: dict[str, Any]) -> dict[str, Any]:
        # Reuse the server-side prompt cache across agent steps; sibling forks
        # then hit the cached prefix (pillar D groundwork).
        body.setdefault("cache_prompt", True)
        return body
