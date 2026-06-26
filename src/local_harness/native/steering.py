"""Activation steering via Contrastive Activation Addition (CAA).

extract_contrastive: run positive/negative prompt pairs, take the mean
difference of last-token hidden states per layer — that direction is the
steering vector. apply() registers forward hooks that add it during
inference. Vectors serialize with torch.save and are model-architecture
agnostic (keyed by layer index).
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
