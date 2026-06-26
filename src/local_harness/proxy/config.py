"""Proxy configuration: server-level defaults, overridable per request via a
`harness` extension object in the request body (stripped before forwarding):

    {"model": ..., "messages": [...],
     "harness": {"skill": "sql_select",
                 "samplers": {"min_p": 0.05, "dry": {}},
                 "bias_profile": "concise",
                 "banned_phrases": ["delve", "tapestry"],
                 "think_budget": 300,
                 "rescue": true}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProxyConfig:
    upstream_url: str = "http://localhost:8080"
    model: str = ""                      # default model if client omits one
    db: str = "proxy.db"                 # event log (every proxied call is replayable)
    skills_dir: str = "skills"
    profiles_dir: str = "profiles"
    # pipeline defaults (per-request `harness` overrides win)
    skill: str | None = None
    samplers: dict[str, Any] = field(default_factory=dict)
    bias_profile: str | None = None
    banned_phrases: list[str] = field(default_factory=list)
    think_budget: int | None = None
    # guardrails
    rescue: bool = True
    max_internal_retries: int = 2

    def merged_ext(self, request_ext: dict[str, Any] | None) -> dict[str, Any]:
        ext = {
            "skill": self.skill,
            "samplers": dict(self.samplers),
            "bias_profile": self.bias_profile,
            "banned_phrases": list(self.banned_phrases),
            "think_budget": self.think_budget,
            "rescue": self.rescue,
        }
        for key, value in (request_ext or {}).items():
            if key in ext:
                ext[key] = value
        return ext
