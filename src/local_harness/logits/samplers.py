"""Normalized sampler zoo. llama.cpp's parameter names are the reference
dialect; other servers receive the subset they understand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..inference.capabilities import Capabilities
from .pipeline import StageResolution, StageStatus

# normalized name -> llama.cpp body params it expands to
_LLAMACPP_EXPANSIONS = {
    "min_p": lambda v: {"min_p": v},
    "top_k": lambda v: {"top_k": v},
    "typical_p": lambda v: {"typical_p": v},
    "top_n_sigma": lambda v: {"top_n_sigma": v},
    "repeat_penalty": lambda v: {"repeat_penalty": v},
    "mirostat": lambda v: {"mirostat": v.get("mode", 2), "mirostat_tau": v.get("tau", 5.0),
                            "mirostat_eta": v.get("eta", 0.1)},
    "dynatemp": lambda v: {"dynatemp_range": v.get("range", 0.0),
                            "dynatemp_exponent": v.get("exponent", 1.0)},
    "dry": lambda v: {"dry_multiplier": v.get("multiplier", 0.8), "dry_base": v.get("base", 1.75),
                       "dry_allowed_length": v.get("allowed_length", 2)},
    "xtc": lambda v: {"xtc_probability": v.get("probability", 0.5),
                       "xtc_threshold": v.get("threshold", 0.1)},
    "sampler_order": lambda v: {"samplers": v},
}

_VLLM_PASSTHROUGH = {"min_p", "top_k", "repeat_penalty"}


@dataclass
class SamplerChain:
    """settings: normalized sampler name -> value (scalar or dict, see above)."""

    settings: dict[str, Any] = field(default_factory=dict)
    name: str = "samplers"

    def compile_http(self, caps: Capabilities) -> StageResolution:
        if not self.settings:
            return StageResolution(self.name, StageStatus.NATIVE, {})
        supported = {k: v for k, v in self.settings.items() if k in caps.sampler_zoo}
        dropped = sorted(set(self.settings) - set(supported))

        params: dict[str, Any] = {}
        if caps.server == "llama.cpp":
            for k, v in supported.items():
                params.update(_LLAMACPP_EXPANSIONS[k](v))
        elif caps.server == "vllm":
            for k, v in supported.items():
                if k in _VLLM_PASSTHROUGH:
                    params[k if k != "repeat_penalty" else "repetition_penalty"] = v
        if not params:
            return StageResolution(
                self.name, StageStatus.UNAVAILABLE, note=f"no supported samplers; dropped {dropped}"
            )
        note = f"dropped unsupported: {dropped}" if dropped else ""
        return StageResolution(self.name, StageStatus.NATIVE, params, note)

    def process(self, input_ids, scores):  # Tier 4 handled by native sampler config
        raise NotImplementedError("sampler chain is applied by the backend, not as a processor")
