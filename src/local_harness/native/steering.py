"""Activation steering (write) and activation probes (read).

SteeringVector — Contrastive Activation Addition (CAA): run positive/negative
prompt pairs, take the mean difference of last-token hidden states per layer —
that direction is the steering vector. apply() registers forward hooks that
add it during inference.

ActivationProbe — the same direction used the other way (ITI/RepE-style
reading): project hidden states onto it and calibrate a threshold from the
contrastive pairs, giving a [0,1] score per layer for how "positive-class" a
text or an in-flight generation is. fit on honest/dishonest statements and the
score is a hallucination signal no logprob can provide.

Both serialize with torch.save and are model-architecture agnostic (keyed by
layer index).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _decoder_layers(model: Any):
    for path in ("transformer.h", "model.layers", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot locate decoder layers on this architecture")


@dataclass
class SteeringVector:
    name: str
    layers: dict[int, Any]  # layer index -> torch tensor [hidden]
    _handles: list = field(default_factory=list, repr=False)

    @classmethod
    def extract_contrastive(
        cls,
        backend,
        name: str,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layer_indices: list[int],
    ) -> "SteeringVector":
        torch = backend.torch
        model, tok = backend.model, backend.tokenizer
        sums: dict[int, Any] = {}
        count = 0

        def capture(prompts: list[str], sign: float) -> None:
            nonlocal count
            for prompt in prompts:
                ids = torch.tensor([tok.encode(prompt)], device=backend.device)
                with torch.no_grad():
                    hidden = model(input_ids=ids, output_hidden_states=True).hidden_states
                for li in layer_indices:
                    vec = hidden[li + 1][0, -1, :].float() * sign  # +1: skip embeddings
                    sums[li] = sums.get(li, 0) + vec
                if sign > 0:
                    count += 1

        capture(positive_prompts, +1.0)
        capture(negative_prompts, -1.0)
        return cls(name=name, layers={li: sums[li] / max(count, 1) for li in layer_indices})

    def apply(self, backend, strength: float = 1.0) -> None:
        layers = _decoder_layers(backend.model)

        def make_hook(vec):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                steered = hidden + strength * vec.to(hidden.dtype)
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered
            return hook

        for li, vec in self.layers.items():
            self._handles.append(layers[li].register_forward_hook(make_hook(vec)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def save(self, path: str | Path) -> None:
        import torch

        torch.save({"name": self.name, "layers": self.layers}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "SteeringVector":
        import torch

        data = torch.load(str(path), weights_only=False)
        return cls(name=data["name"], layers=data["layers"])


def _last_token_hidden(backend, text: str):
    """hidden_states tuple for one forward pass over `text` (no grad)."""
    torch = backend.torch
    ids = torch.tensor([backend.tokenizer.encode(text)], device=backend.device)
    with torch.no_grad():
        return backend.model(input_ids=ids, output_hidden_states=True).hidden_states


@dataclass
class ProbeReadout:
    """Aggregated probe scores: 0 = negative class, 1 = positive class."""

    name: str
    score: float                 # mean over layers (and tokens, when attached)
    per_layer: dict[int, float]
    n_observations: int          # forward passes that contributed

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "per_layer": {str(k): v for k, v in self.per_layer.items()},
            "n_observations": self.n_observations,
        }


@dataclass
class ActivationProbe:
    """Linear probe over hidden states, fit from contrastive prompt pairs.

    Per layer it stores a unit direction, a decision threshold (midpoint of the
    class means' projections), and a scale (inverse pooled std) so projections
    map through a sigmoid to a calibrated-ish [0,1] score. Use `score_text` for
    one-shot scoring, or `attach`/`readout`/`detach` to watch a generation
    token-by-token through forward hooks (the read-side twin of
    SteeringVector.apply)."""

    name: str
    directions: dict[int, Any]   # layer index -> unit tensor [hidden]
    thresholds: dict[int, float]
    scales: dict[int, float]
    _handles: list = field(default_factory=list, repr=False)
    _records: dict[int, list] = field(default_factory=dict, repr=False)

    @classmethod
    def fit_contrastive(
        cls,
        backend,
        name: str,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layer_indices: list[int],
    ) -> "ActivationProbe":
        torch = backend.torch

        def reps(prompts: list[str]) -> dict[int, Any]:
            out: dict[int, list] = {li: [] for li in layer_indices}
            for prompt in prompts:
                hidden = _last_token_hidden(backend, prompt)
                for li in layer_indices:
                    out[li].append(hidden[li + 1][0, -1, :].float())  # +1: skip embeddings
            return {li: torch.stack(v) for li, v in out.items()}

        pos, neg = reps(positive_prompts), reps(negative_prompts)
        directions, thresholds, scales = {}, {}, {}
        for li in layer_indices:
            direction = pos[li].mean(0) - neg[li].mean(0)
            direction = direction / direction.norm().clamp_min(1e-8)
            proj_pos = pos[li] @ direction
            proj_neg = neg[li] @ direction
            thresholds[li] = float((proj_pos.mean() + proj_neg.mean()) / 2)
            pooled = float(
                torch.cat([proj_pos, proj_neg]).std(unbiased=False).clamp_min(1e-6)
            )
            directions[li] = direction
            scales[li] = 1.0 / pooled
        return cls(name=name, directions=directions, thresholds=thresholds, scales=scales)

    def _score_hidden(self, li: int, hidden_vec) -> float:
        import math

        proj = float(hidden_vec.float() @ self.directions[li])
        margin = (proj - self.thresholds[li]) * self.scales[li]
        return 1.0 / (1.0 + math.exp(-margin))

    def score_text(self, backend, text: str) -> ProbeReadout:
        """One forward pass; score the last-token representation per layer."""
        hidden = _last_token_hidden(backend, text)
        per_layer = {
            li: self._score_hidden(li, hidden[li + 1][0, -1, :])
            for li in self.directions
        }
        return ProbeReadout(
            name=self.name,
            score=sum(per_layer.values()) / len(per_layer),
            per_layer=per_layer,
            n_observations=1,
        )

    def attach(self, backend) -> None:
        """Register read-only hooks that record a score per forward pass (i.e.
        per generated token once the prompt is prefixed). Call readout() then
        detach()."""
        layers = _decoder_layers(backend.model)
        self._records = {li: [] for li in self.directions}

        def make_hook(li):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                self._records[li].append(self._score_hidden(li, hidden[0, -1, :].detach()))
                return output

            return hook

        for li in self.directions:
            self._handles.append(layers[li].register_forward_hook(make_hook(li)))

    def readout(self) -> ProbeReadout:
        per_layer = {
            li: (sum(v) / len(v) if v else 0.5) for li, v in self._records.items()
        }
        n = max((len(v) for v in self._records.values()), default=0)
        return ProbeReadout(
            name=self.name,
            score=sum(per_layer.values()) / max(len(per_layer), 1),
            per_layer=per_layer,
            n_observations=n,
        )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def save(self, path: str | Path) -> None:
        import torch

        torch.save(
            {
                "name": self.name,
                "directions": self.directions,
                "thresholds": self.thresholds,
                "scales": self.scales,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ActivationProbe":
        import torch

        data = torch.load(str(path), weights_only=False)
        return cls(
            name=data["name"],
            directions=data["directions"],
            thresholds=data["thresholds"],
            scales=data["scales"],
        )
