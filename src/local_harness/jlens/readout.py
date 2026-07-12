# Vendored from igorbarshteyn/jlens-gguf (Apache-2.0) at pinned commit,
# adapted for lo. See src/local_harness/jlens/NOTICE.
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Numpy lens math: readout, ranks, lens vectors, interventions, decomposition.

A :class:`LensReadout` ties together a model's readout weights
(:class:`~jlens_gguf.model_reader.ReadoutWeights`) and a fitted lens
(:class:`~jlens_gguf.lens.JacobianLensGGUF`) and provides everything the
bridge/UI needs:

- ``lens_logits``: ``unembed(J_l @ h)`` per layer (``J = I`` on the final
  layer, which reproduces the model's own output distribution).
- ``topk`` / ``ranks_of``: vocabulary rankings, chunked to bound memory.
- ``lens_vector``: the J-lens vector ``v_t = J_l^T (gamma ⊙ W_U[t])`` — the
  direction in layer-l residual space whose amplification raises token t's
  lens logit (first-order, through the RMS norm's diagonal scale).
- steering / ablation / swap factor builders that translate paper operations
  into the native server's generic "add" / "lowrank" residual edits:
    steer:  h += alpha * v̂_t
    ablate: h -= (v̂_t · h) v̂_t                    (rank-1 lowrank)
    swap:   h += V (sigma(c) - c),  c = V⁺ h       (rank-2 lowrank)
- ``decompose``: greedy matching pursuit of h onto J-lens vectors (the paper's
  sparse J-space decomposition, k ≤ 25).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from local_harness.jlens.lens import JacobianLensGGUF
from local_harness.jlens.model_reader import ReadoutWeights


class LensReadout:
    def __init__(self, weights: ReadoutWeights, lens: JacobianLensGGUF) -> None:
        if weights.d_model != lens.d_model:
            raise ValueError(
                f"model d_model={weights.d_model} but lens d_model={lens.d_model}"
            )
        self.weights = weights
        self.lens = lens
        # gamma ⊙ W_U rows are the readout directions in final-layer space
        gamma = weights.norm_weight
        self._wu_gamma = (
            weights.w_unembed * gamma if gamma is not None else weights.w_unembed
        )  # [vocab, d]
        self._lens_vec_norms: dict[int, np.ndarray] = {}  # layer -> [vocab] cache (lazy, partial)

    # ------------------------------------------------------------------ #
    # readout
    # ------------------------------------------------------------------ #

    def readout_layers(self, n_layers: int) -> list[int]:
        """Fitted layers plus the model's final layer (rendered with J = I)."""
        final = n_layers - 1
        return sorted(set(self.lens.source_layers) | {final})

    def lens_logits(self, residual: np.ndarray, layer: int, *, use_lens: bool = True) -> np.ndarray:
        """Lens logits for residuals ``[..., d]`` at ``layer``.

        Layers without a fitted ``J`` (the final layer, or any layer when
        ``use_lens=False``) are read out with ``J = I`` (logit lens); on the
        final layer that equals the model's own head.
        """
        h = residual
        if use_lens and layer in self.lens.jacobians:
            h = self.lens.transport(h, layer)
        return self.weights.unembed(h)

    @staticmethod
    def topk(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k ids and values per row for ``logits [T, vocab]`` (descending)."""
        k = min(k, logits.shape[-1])
        part = np.argpartition(-logits, k - 1, axis=-1)[..., :k]
        vals = np.take_along_axis(logits, part, axis=-1)
        order = np.argsort(-vals, axis=-1)
        return np.take_along_axis(part, order, axis=-1), np.take_along_axis(vals, order, axis=-1)

    @staticmethod
    def ranks_of(logits: np.ndarray, token_ids: np.ndarray, *, chunk: int = 128) -> np.ndarray:
        """Full-vocab rank (0 = top) of each ``token_ids`` at every row.

        ``logits``: [T, vocab]; ``token_ids``: [n]; returns [T, n] int32.
        Chunked over rows: peak memory is one ``[chunk, vocab]`` sort buffer.
        """
        T = logits.shape[0]
        ids = np.asarray(token_ids, dtype=np.int64)
        out = np.empty((T, ids.shape[0]), dtype=np.int32)
        vocab = logits.shape[1]
        for s in range(0, T, chunk):
            rows = logits[s : s + chunk]
            srt = np.sort(rows, axis=-1)  # ascending
            targets = np.take_along_axis(rows, np.broadcast_to(ids, (rows.shape[0], ids.shape[0])), axis=-1)
            # rank = number of strictly greater entries
            gt = vocab - np.stack(
                [np.searchsorted(srt[i], targets[i], side="right") for i in range(rows.shape[0])]
            )
            out[s : s + chunk] = gt.astype(np.int32)
        return out

    # ------------------------------------------------------------------ #
    # lens vectors + intervention factors
    # ------------------------------------------------------------------ #

    def lens_vector(self, layer: int, token_id: int, *, unit: bool = True) -> np.ndarray:
        """J-lens vector ``v_t`` for ``token_id`` at ``layer`` (layer-l space)."""
        u = self._wu_gamma[int(token_id)]  # [d], final-layer space
        J = self.lens.jacobians.get(layer)
        v = u if J is None else J.T @ u
        if unit:
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
        return v.astype(np.float32)

    def steer_vector(self, layer: int, token_id: int, alpha: float, *, h_rms: float | None = None) -> np.ndarray:
        """Additive steering vector ``alpha * scale * v̂_t``.

        ``alpha`` is in units of the layer's typical residual RMS-norm
        (``h_rms``), so alpha ~ 1 injects a component comparable to the whole
        residual. Falls back to the lens's stored per-layer norm, then 1.0.
        """
        if h_rms is None:
            h_rms = self.lens.h_rms.get(layer, 1.0)
        v = self.lens_vector(layer, token_id, unit=True)
        return (float(alpha) * float(h_rms) * v).astype(np.float32)

    def ablate_factors(self, layer: int, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Rank-k factors that remove the projection onto the tokens' J-lens
        directions: ``h += A(Bh)`` with ``A = -Q``, ``B = Qᵀ`` where Q is an
        orthonormal basis of span{v_t}."""
        V = np.stack([self.lens_vector(layer, t, unit=True) for t in token_ids], axis=1)  # [d,k]
        Q, _ = np.linalg.qr(V)
        return (-Q).astype(np.float32), Q.T.astype(np.float32)

    def swap_factors(self, layer: int, token_a: int, token_b: int) -> tuple[np.ndarray, np.ndarray]:
        """Rank-2 factors implementing the paper's concept patch:
        ``h' = h + V (sigma(c) - c)`` with ``c = V⁺ h`` and sigma the swap of
        the two lens coordinates. Returns (A, B) with A = V(S - I), B = V⁺."""
        v_a = self.lens_vector(layer, token_a, unit=False)
        v_b = self.lens_vector(layer, token_b, unit=False)
        V = np.stack([v_a, v_b], axis=1)  # [d, 2]
        Vp = np.linalg.pinv(V)            # [2, d]
        S = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        A = V @ (S - np.eye(2, dtype=np.float32))  # [d, 2]
        return A.astype(np.float32), Vp.astype(np.float32)

    # ------------------------------------------------------------------ #
    # sparse J-space decomposition (matching pursuit)
    # ------------------------------------------------------------------ #

    def decompose(
        self,
        residual: np.ndarray,
        layer: int,
        *,
        k: int = 12,
        candidates: int = 2048,
        nonneg: bool = True,
    ) -> list[dict]:
        """Greedy sparse decomposition of ``residual`` onto J-lens vectors.

        Matching pursuit: at each step, score every vocabulary token by the
        normalized inner product of its J-lens vector with the current
        residual remainder (computed cheaply as ``(gamma ⊙ W_U) (J r)`` over
        the whole vocab, with exact atom norms evaluated only for the top
        ``candidates``), select the best, and re-fit coefficients of the
        active set by least squares.

        Returns ``[{"token": id, "coeff": float, "score": float}, ...]`` in
        selection order, plus the share of ``||h||²`` explained under key
        ``"explained"`` on each entry (cumulative).
        """
        h = residual.astype(np.float32)
        candidates = min(candidates, self._wu_gamma.shape[0] - 1)
        J = self.lens.jacobians.get(layer)
        r = h.copy()
        active: list[int] = []
        atoms: list[np.ndarray] = []
        results: list[dict] = []
        h_ss = float(h @ h) or 1.0

        for _ in range(k):
            jr = r if J is None else J @ r
            scores = self._wu_gamma @ jr  # [vocab] — unnormalized <v_t, r>
            if nonneg:
                cand = np.argpartition(-scores, candidates)[:candidates]
            else:
                cand = np.argpartition(-np.abs(scores), candidates)[:candidates]
            # exact normalized scores for candidates only
            U = self._wu_gamma[cand]                       # [c, d]
            Vc = U if J is None else U @ J                 # [c, d] rows = v_t
            norms = np.linalg.norm(Vc, axis=1) + 1e-12
            nscores = scores[cand] / norms
            best = int(cand[int(np.argmax(nscores))])
            if best in active:
                break
            active.append(best)
            atoms.append(self.lens_vector(layer, best, unit=False))

            V = np.stack(atoms, axis=1)  # [d, m]
            coef, *_ = np.linalg.lstsq(V, h, rcond=None)
            if nonneg:
                coef = np.maximum(coef, 0.0)
            r = h - V @ coef
            explained = 1.0 - float(r @ r) / h_ss
            results.append({"token": best, "coeff": float(coef[-1]), "score": float(np.max(nscores)), "explained": explained})

        # final coefficients (post re-fit) back-fill
        if atoms:
            V = np.stack(atoms, axis=1)
            coef, *_ = np.linalg.lstsq(V, h, rcond=None)
            if nonneg:
                coef = np.maximum(coef, 0.0)
            for i, res in enumerate(results):
                res["coeff"] = float(coef[i])
        return results


# ---------------------------------------------------------------------- #
# grid computation (position x layer slice)
# ---------------------------------------------------------------------- #


@dataclass
class SliceGrid:
    """Top-k grid over (position, layer) plus rank data for tracked tokens."""

    layers: list[int]                 # readout layers, ascending
    top_ids: np.ndarray               # [T, L, K] int32
    top_logits: np.ndarray            # [T, L, K] float32
    norms: dict[int, float]           # layer -> median residual norm (steering scale)


def compute_grid(
    readout: LensReadout,
    activations: dict[int, np.ndarray],
    layers: list[int],
    *,
    top_n: int = 10,
    use_lens: bool = True,
    positions: slice | None = None,
    logits_fn=None,
) -> SliceGrid:
    """Lens top-k for every (position, layer) from captured activations.

    ``logits_fn(layer) -> [T, vocab]`` overrides the default per-layer logits
    computation; the bridge passes its LRU-cached accessor so later rank
    queries reuse the same arrays instead of re-running the unembedding GEMM.
    """
    sel = positions if positions is not None else slice(None)
    T = activations[layers[0]][sel].shape[0]
    L = len(layers)
    top_ids = np.zeros((T, L, top_n), dtype=np.int32)
    top_logits = np.zeros((T, L, top_n), dtype=np.float32)
    norms: dict[int, float] = {}
    for li, layer in enumerate(layers):
        H = activations[layer][sel]
        norms[layer] = float(np.median(np.linalg.norm(H, axis=-1)))
        if logits_fn is not None:
            logits = logits_fn(layer)
        else:
            logits = readout.lens_logits(H, layer, use_lens=use_lens)
        ids, vals = LensReadout.topk(logits, top_n)
        top_ids[:, li] = ids.astype(np.int32)
        top_logits[:, li] = vals.astype(np.float32)
        del logits
    return SliceGrid(layers=list(layers), top_ids=top_ids, top_logits=top_logits, norms=norms)
