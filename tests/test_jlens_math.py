"""Hermetic tests for the vendored Jacobian-lens math + JLNS client.

No llama.cpp, no real model — everything runs against tests/jlens_support.py's
in-process mock sidecar and synthetic weights. Skips cleanly if the `lens`
extra (numpy/gguf) is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gguf")

import numpy as np  # noqa: E402

from local_harness.jlens import JacobianLensGGUF, LensReadout, NativeClient  # noqa: E402
from local_harness.jlens.model_reader import ReadoutWeights  # noqa: E402
from local_harness.jlens.readout import compute_grid  # noqa: E402
from local_harness.jlens.topology import (  # noqa: E402
    emitting_layers,
    final_readable_layer,
)
from jlens_support import D, MockModel, MockSidecar  # noqa: E402


# ---------------------------------------------------------------- topology --


def test_mtp_readable_layers():
    # qwen3.6-27b-mtp: 65 blocks, 1 NextN → 64 emit, final index 63
    assert emitting_layers(65, 1) == 64
    assert final_readable_layer(65, 1) == 63
    # dense model: no NextN
    assert emitting_layers(32, 0) == 32
    assert final_readable_layer(32, 0) == 31
    with pytest.raises(ValueError):
        emitting_layers(1, 1)


def test_readout_weights_final_layer_property():
    w = ReadoutWeights(arch="x", n_layers=65, d_model=8, n_vocab=10,
                       w_unembed=np.zeros((10, 8), np.float32),
                       norm_weight=None, norm_bias=None, norm_type="rms",
                       eps=1e-5, n_nextn=1)
    assert w.n_readable_layers == 64
    assert w.final_layer == 63


# ----------------------------------------------------------------- lens io --


def test_lens_gguf_roundtrip(tmp_path):
    d = 8
    jacs = {l: np.random.default_rng(l).standard_normal((d, d)).astype(np.float32)
            for l in (1, 2, 3)}
    biases = {1: np.arange(d, dtype=np.float32)}
    hrms = {1: 12.5, 2: 20.0, 3: 30.0}
    lens = JacobianLensGGUF(jacs, d_model=d, n_prompts=42, target_layer=4,
                            fit_method="regression", biases=biases, h_rms=hrms)
    p = tmp_path / "lens.gguf"
    lens.save(str(p))
    back = JacobianLensGGUF.load(str(p))
    assert back.d_model == d
    assert back.source_layers == [1, 2, 3]
    assert back.n_prompts == 42
    assert back.fit_method == "regression"
    for l in jacs:
        np.testing.assert_allclose(back.jacobians[l], jacs[l], atol=1e-2)  # fp16
    np.testing.assert_allclose(back.biases[1], biases[1], atol=1e-3)
    assert back.h_rms[1] == pytest.approx(12.5, abs=0.1)


def test_identity_lens_is_logit_lens():
    d = 8
    lens = JacobianLensGGUF.identity(d_model=d, layers=[0, 1, 2])
    h = np.random.default_rng(0).standard_normal((5, d)).astype(np.float32)
    np.testing.assert_allclose(lens.transport(h, 1), h, atol=1e-6)


def test_merge_weighted_mean():
    d = 4
    a = JacobianLensGGUF({0: np.ones((d, d), np.float32)}, d_model=d, n_prompts=1)
    b = JacobianLensGGUF({0: np.full((d, d), 3.0, np.float32)}, d_model=d, n_prompts=3)
    m = JacobianLensGGUF.merge([a, b])
    # (1*1 + 3*3)/4 = 2.5
    np.testing.assert_allclose(m.jacobians[0], np.full((d, d), 2.5), atol=1e-3)
    assert m.n_prompts == 4


# ---------------------------------------------- intervention factor algebra --


def _readout():
    m = MockModel()
    w = ReadoutWeights(arch="mock", n_layers=m.n_layers, d_model=m.d,
                       n_vocab=m.vocab, w_unembed=m.w_unembed,
                       norm_weight=m.norm_weight, norm_bias=None,
                       norm_type="rms", eps=1e-5)
    lens = JacobianLensGGUF.identity(d_model=m.d, layers=list(range(m.n_layers - 1)))
    return LensReadout(w, lens), m


def test_ablate_removes_projection():
    ro, m = _readout()
    tid = 5
    v = ro.lens_vector(1, tid, unit=True)
    A, B = ro.ablate_factors(1, [tid])
    h = np.random.default_rng(1).standard_normal(m.d).astype(np.float32)
    h2 = h + A @ (B @ h)
    # component along v must be gone
    assert abs(float(v @ h2)) < 1e-4
    # orthogonal complement preserved
    perp = h - (v @ h) * v
    np.testing.assert_allclose(h2, perp, atol=1e-4)


def test_swap_exchanges_lens_coords():
    ro, m = _readout()
    ta, tb = 5, 9
    A, B = ro.swap_factors(1, ta, tb)
    va = ro.lens_vector(1, ta, unit=False)
    vb = ro.lens_vector(1, tb, unit=False)
    V = np.stack([va, vb], axis=1)
    Vp = np.linalg.pinv(V)
    h = 2.0 * va + 5.0 * vb + np.random.default_rng(2).standard_normal(m.d).astype(np.float32) * 0.01
    h2 = h + A @ (B @ h)
    c2 = Vp @ h2
    c1 = Vp @ h
    # coordinates swapped
    assert c2[0] == pytest.approx(c1[1], abs=1e-2)
    assert c2[1] == pytest.approx(c1[0], abs=1e-2)


def test_steer_vector_scales_with_hrms():
    ro, _ = _readout()
    v1 = ro.steer_vector(1, 5, alpha=1.0, h_rms=10.0)
    v2 = ro.steer_vector(1, 5, alpha=2.0, h_rms=10.0)
    np.testing.assert_allclose(v2, 2 * v1, atol=1e-5)
    assert float(np.linalg.norm(v1)) == pytest.approx(10.0, rel=1e-3)


# ------------------------------------------------------- ranks / topk math --


def test_topk_and_ranks_agree():
    logits = np.random.default_rng(3).standard_normal((4, 50)).astype(np.float32)
    ids, vals = LensReadout.topk(logits, 5)
    assert ids.shape == (4, 5)
    # top-1 rank of the argmax token is 0
    top1 = logits.argmax(-1)
    r = LensReadout.ranks_of(logits, top1)
    assert (np.diagonal(r) == 0).all()


# --------------------------------------------------- JLNS client end-to-end --


def test_client_forward_against_mock():
    with MockSidecar() as sc:
        c = NativeClient(sc.url)
        assert c.health()
        props = c.props()
        assert props["l_out_ok"] is True
        toks = c.tokenize("the quick brown fox")
        fr = c.forward(toks, capture_layers=[0, 2, 3], dtype="f16")
        assert set(fr.activations) == {0, 2, 3}
        assert fr.activations[0].shape == (len(toks), D)
        # deterministic: same tokens → same residuals
        fr2 = c.forward(toks, capture_layers=[2], dtype="f32")
        np.testing.assert_allclose(fr.activations[2], fr2.activations[2], atol=1e-2)


def test_client_intervention_changes_activation():
    with MockSidecar() as sc:
        c = NativeClient(sc.url)
        toks = c.tokenize("hello world")
        base = c.forward(toks, capture_layers=[1], dtype="f32").activations[1]
        vec = np.ones(D, np.float32) * 5.0
        iv = {"layer": 1, "pos_start": 0, "pos_end": -1, "mode": "add", "vector": vec}
        steered = c.forward(toks, capture_layers=[1], dtype="f32",
                            interventions=[iv]).activations[1]
        np.testing.assert_allclose(steered - base, np.broadcast_to(vec, base.shape), atol=1e-2)


def test_compute_grid_shapes():
    with MockSidecar() as sc:
        c = NativeClient(sc.url)
        m = MockModel()
        w = ReadoutWeights(arch="mock", n_layers=m.n_layers, d_model=m.d,
                           n_vocab=m.vocab, w_unembed=m.w_unembed,
                           norm_weight=m.norm_weight, norm_bias=None,
                           norm_type="rms", eps=1e-5)
        lens = JacobianLensGGUF.identity(d_model=m.d, layers=list(range(m.n_layers - 1)))
        ro = LensReadout(w, lens)
        toks = c.tokenize("a b c d")
        layers = [0, 1, 2, 3]
        fr = c.forward(toks, capture_layers=layers, dtype="f32")
        grid = compute_grid(ro, fr.activations, layers, top_n=5, use_lens=True)
        assert grid.top_ids.shape == (len(toks), 4, 5)
        assert set(grid.norms) == set(layers)
