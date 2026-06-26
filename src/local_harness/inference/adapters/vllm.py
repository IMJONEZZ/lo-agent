from __future__ import annotations

from .base import Adapter, Fingerprint, StaticCaps

VLLM_SAMPLER_ZOO = {"min_p", "top_k", "repetition_penalty"}


class VllmAdapter(Adapter):
    @classmethod
    def matches(cls, fp: Fingerprint) -> bool:
        return (fp.owned_by or "").lower() == "vllm"

    @classmethod
    def static_caps(cls, fp: Fingerprint) -> StaticCaps:
        return StaticCaps(
            server="vllm",
            grammar="guided",       # guided_grammar / guided_json via extra body
            logit_bias=True,
            sampler_zoo=set(VLLM_SAMPLER_ZOO),
            cfg_scale=False,
            banned_strings=False,
            parallel_n=True,        # n>1 parallel sampling + automatic prefix cache
            stream_logprobs=True,   # vLLM streams logprobs even with tools (llama.cpp won't)
        )
