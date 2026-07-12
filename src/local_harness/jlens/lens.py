# Vendored from igorbarshteyn/jlens-gguf (Apache-2.0) at pinned commit,
# adapted for lo. See src/local_harness/jlens/NOTICE.
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""The Jacobian lens as a GGUF file.

A lens is a set of per-layer matrices ``J_l`` (optionally with a bias ``b_l``)
that transport a residual at layer ``l`` into the final-layer basis:
``h_hat = J_l @ h + b_l``. This module stores them in a GGUF container:

kv metadata:
    general.architecture   = "jlens"
    jlens.format_version   = 1
    jlens.d_model          (u32)
    jlens.n_prompts        (u32)   prompts averaged over during fitting
    jlens.target_layer     (i32)
    jlens.source_layers    (arr i32)
    jlens.fit_method       ("jacobian" | "regression" | "identity")
    jlens.base_model       (str, informational)
    jlens.h_rms            (arr f32, optional: typical residual RMS-norm per
                            source layer, same order as source_layers; used to
                            scale steering vectors)

tensors:
    jlens.J.{layer}   [d, d] F32/F16, row-major (h_hat = J @ h)
    jlens.b.{layer}   [d]    F32, optional
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

FORMAT_VERSION = 1


def _kv_scalar(reader, key: str, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        # older gguf-py: single scalar lives at field.data[0] in parts
        part = field.parts[field.data[0]]
        return part[0].item() if hasattr(part, "item") else part


class JacobianLensGGUF:
    """Per-layer transport matrices, stored in / loaded from GGUF.

    Attributes:
        jacobians: ``{layer: np.ndarray [d, d] float32}``.
        biases: ``{layer: np.ndarray [d] float32}`` (may be empty).
        source_layers: sorted fitted layer indices.
        d_model: residual width.
        n_prompts: prompts the fit averaged over.
        target_layer: layer the transport maps into (-1 = final).
        fit_method: "jacobian" | "regression" | "identity".
        h_rms: ``{layer: float}`` typical residual RMS norm (for steering scale).
    """

    def __init__(
        self,
        jacobians: dict[int, np.ndarray],
        *,
        d_model: int,
        n_prompts: int = 0,
        target_layer: int = -1,
        fit_method: str = "jacobian",
        base_model: str = "",
        biases: dict[int, np.ndarray] | None = None,
        h_rms: dict[int, float] | None = None,
    ) -> None:
        self.jacobians = {int(l): np.ascontiguousarray(J, dtype=np.float32) for l, J in jacobians.items()}
        self.biases = {int(l): np.ascontiguousarray(b, dtype=np.float32) for l, b in (biases or {}).items()}
        self.source_layers = sorted(self.jacobians)
        self.d_model = int(d_model)
        self.n_prompts = int(n_prompts)
        self.target_layer = int(target_layer)
        self.fit_method = fit_method
        self.base_model = base_model
        self.h_rms = {int(l): float(v) for l, v in (h_rms or {}).items()}
        for l, J in self.jacobians.items():
            if J.shape != (self.d_model, self.d_model):
                raise ValueError(f"J_{l} has shape {J.shape}, expected {(self.d_model,) * 2}")

    def __repr__(self) -> str:
        rng = f"[{self.source_layers[0]}..{self.source_layers[-1]}]" if self.source_layers else "[]"
        return (
            f"JacobianLensGGUF(d_model={self.d_model}, layers={rng} "
            f"({len(self.source_layers)}), method={self.fit_method!r}, n_prompts={self.n_prompts})"
        )

    # ------------------------------------------------------------------ #
    # core op
    # ------------------------------------------------------------------ #

    def transport(self, residual: np.ndarray, layer: int) -> np.ndarray:
        """``J_l @ h (+ b_l)`` for ``residual`` of shape ``[..., d_model]``."""
        J = self.jacobians[layer]
        out = residual.astype(np.float32, copy=False) @ J.T
        b = self.biases.get(layer)
        if b is not None:
            out = out + b
        return out

    # ------------------------------------------------------------------ #
    # GGUF I/O
    # ------------------------------------------------------------------ #

    def save(self, path: str, *, dtype=np.float16) -> None:
        """Write the lens as a GGUF file. ``dtype`` applies to the J matrices
        (fp16 halves the file; entries are O(1) so range is not a concern)."""
        import gguf

        writer = gguf.GGUFWriter(path, arch="jlens")
        writer.add_uint32("jlens.format_version", FORMAT_VERSION)
        writer.add_uint32("jlens.d_model", self.d_model)
        writer.add_uint32("jlens.n_prompts", self.n_prompts)
        writer.add_int32("jlens.target_layer", self.target_layer)
        writer.add_array("jlens.source_layers", [int(l) for l in self.source_layers])
        writer.add_string("jlens.fit_method", self.fit_method)
        if self.base_model:
            writer.add_string("jlens.base_model", self.base_model)
        if self.h_rms:
            writer.add_array(
                "jlens.h_rms", [float(self.h_rms.get(l, 0.0)) for l in self.source_layers]
            )
        for layer in self.source_layers:
            data = self.jacobians[layer].astype(dtype)
            writer.add_tensor(f"jlens.J.{layer}", data)
            if layer in self.biases:
                writer.add_tensor(f"jlens.b.{layer}", self.biases[layer].astype(np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def load(cls, path: str) -> "JacobianLensGGUF":
        """Load a lens previously written by :meth:`save`."""
        import gguf

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        reader = gguf.GGUFReader(path)
        arch = _kv_scalar(reader, "general.architecture")
        if arch != "jlens":
            raise ValueError(f"{path} is not a jlens GGUF (architecture={arch!r})")
        d_model = int(_kv_scalar(reader, "jlens.d_model"))
        n_prompts = int(_kv_scalar(reader, "jlens.n_prompts", 0))
        target_layer = int(_kv_scalar(reader, "jlens.target_layer", -1))
        fit_method = str(_kv_scalar(reader, "jlens.fit_method", "jacobian"))
        base_model = str(_kv_scalar(reader, "jlens.base_model", "") or "")
        source_layers = list(_kv_scalar(reader, "jlens.source_layers", []) or [])
        h_rms_arr = list(_kv_scalar(reader, "jlens.h_rms", []) or [])

        jacobians: dict[int, np.ndarray] = {}
        biases: dict[int, np.ndarray] = {}
        for tensor in reader.tensors:
            name = tensor.name
            if name.startswith("jlens.J."):
                layer = int(name.split(".")[-1])
                jacobians[layer] = np.asarray(tensor.data, dtype=np.float32).reshape(d_model, d_model)
            elif name.startswith("jlens.b."):
                layer = int(name.split(".")[-1])
                biases[layer] = np.asarray(tensor.data, dtype=np.float32).reshape(d_model)
        h_rms = {int(l): float(v) for l, v in zip(source_layers, h_rms_arr)}
        return cls(
            jacobians,
            d_model=d_model,
            n_prompts=n_prompts,
            target_layer=target_layer,
            fit_method=fit_method,
            base_model=base_model,
            biases=biases,
            h_rms=h_rms,
        )

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def identity(cls, *, d_model: int, layers: Sequence[int], base_model: str = "") -> "JacobianLensGGUF":
        """The logit-lens baseline: ``J_l = I`` at every requested layer."""
        eye = np.eye(d_model, dtype=np.float32)
        return cls(
            {int(l): eye.copy() for l in layers},
            d_model=d_model,
            n_prompts=0,
            fit_method="identity",
            base_model=base_model,
        )

    @classmethod
    def merge(cls, lenses: Sequence["JacobianLensGGUF"]) -> "JacobianLensGGUF":
        """n_prompts-weighted mean of lenses fitted on disjoint prompt sets."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.source_layers != first.source_layers or other.d_model != first.d_model:
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(max(l.n_prompts, 1) for l in lenses)
        merged = {
            layer: sum(l.jacobians[layer] * max(l.n_prompts, 1) for l in lenses) / n_total
            for layer in first.source_layers
        }
        return cls(
            merged,
            d_model=first.d_model,
            n_prompts=n_total,
            target_layer=first.target_layer,
            fit_method=first.fit_method,
            base_model=first.base_model,
        )
