from __future__ import annotations

from .base import Adapter, Fingerprint, StaticCaps


class GenericAdapter(Adapter):
    """Pure Tier 0: any OpenAI-compatible endpoint, no assumptions."""

    @classmethod
    def matches(cls, fp: Fingerprint) -> bool:
        return True

    @classmethod
    def static_caps(cls, fp: Fingerprint) -> StaticCaps:
        return StaticCaps(server="generic")
