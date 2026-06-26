"""Logit-bias profiles: static token-level persona/style control.

Profiles store *strings* with biases and resolve to token ids per server
(via llama.cpp's /tokenize) or per tokenizer (Tier 4), so one profile file
works across models with different vocabularies.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..inference.capabilities import Capabilities
from .pipeline import StageResolution, StageStatus


@dataclass
class BiasProfile:
    name: str
    description: str = ""
    string_biases: dict[str, float] = field(default_factory=dict)  # text piece -> bias

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BiasProfile":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def concise(cls) -> "BiasProfile":
        return cls(
            name="concise",
            description="Promote sentence endings and EOS-adjacent punctuation.",
            string_biases={".": 1.5, ".\n": 1.5, "!": 0.5, " however": -2.0,
                           " Additionally": -3.0, " Furthermore": -3.0},
        )

    @classmethod
    def verbose(cls) -> "BiasProfile":
        return cls(
            name="verbose",
            description="Suppress sentence endings to lengthen output.",
            string_biases={".": -1.5, ".\n": -2.0, " Additionally": 1.0, " also": 1.0},
        )

    @classmethod
    def from_corpus(
        cls,
        name: str,
        positive_texts: list[str],
        negative_texts: list[str],
        top_k: int = 32,
        scale: float = 2.0,
    ) -> "BiasProfile":
        """Contrastive word-frequency profile: words distinctive of the positive
        corpus get positive bias, distinctive negative words get suppressed."""

        def counts(texts: list[str]) -> Counter:
            c: Counter = Counter()
            for t in texts:
                c.update(re.findall(r"[A-Za-z][a-z']+", t))
            return c

        pos, neg = counts(positive_texts), counts(negative_texts)
        pos_total, neg_total = sum(pos.values()) or 1, sum(neg.values()) or 1
        scores: dict[str, float] = {}
        for word in set(pos) | set(neg):
            p = (pos[word] + 1) / (pos_total + 1)
            q = (neg[word] + 1) / (neg_total + 1)
            scores[word] = math.log(p / q)
        ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        biases = {f" {w}": max(-abs(scale), min(scale, s * scale)) for w, s in ranked}
        return cls(name=name, description="derived from contrastive corpora", string_biases=biases)


class BiasProfileStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, profile: BiasProfile) -> Path:
        path = self.dir / f"{profile.name}.json"
        profile.save(path)
        return path

    def get(self, name: str) -> BiasProfile:
        builtin = {"concise": BiasProfile.concise, "verbose": BiasProfile.verbose}
        path = self.dir / f"{name}.json"
        if path.exists():
            return BiasProfile.load(path)
        if name in builtin:
            return builtin[name]()
        raise KeyError(f"unknown bias profile {name!r}")


@dataclass
class BiasStage:
    profile: BiasProfile
    token_biases: dict[int, float] = field(default_factory=dict)  # resolved per server
    name: str = "bias"

    async def resolve_tokens(self, client) -> None:
        """Resolve string biases to token ids via the server's /tokenize
        (llama.cpp). Multi-token strings bias their first token."""
        self.token_biases = {}
        for text, bias in self.profile.string_biases.items():
            resp = await client.post("/tokenize", json={"content": text, "add_special": False})
            if resp.status_code != 200:
                return
            tokens = resp.json().get("tokens", [])
            if tokens:
                tok = tokens[0] if isinstance(tokens[0], int) else tokens[0].get("id")
                self.token_biases[tok] = self.token_biases.get(tok, 0.0) + bias

    def compile_http(self, caps: Capabilities) -> StageResolution:
        if not caps.logit_bias:
            return StageResolution(self.name, StageStatus.UNAVAILABLE, note="no logit_bias support")
        if not self.token_biases:
            return StageResolution(
                self.name, StageStatus.UNAVAILABLE,
                note="string biases not resolved to token ids (no /tokenize?)",
            )
        return StageResolution(
            self.name, StageStatus.NATIVE,
            {"logit_bias": {str(k): v for k, v in self.token_biases.items()}},
            note=f"profile={self.profile.name}",
        )

    def process(self, input_ids, scores):
        for tok, bias in self.token_biases.items():
            scores[..., tok] += bias
        return scores
