"""Phase-4 tests: control-vector + abliteration exports (math + GGUF I/O)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")
pytest.importorskip("gguf")

from local_harness.jlens import exports  # noqa: E402
from local_harness.jlens.lens import JacobianLensGGUF  # noqa: E402
from local_harness.jlens.model_reader import ReadoutWeights  # noqa: E402
from local_harness.jlens.readout import LensReadout  # noqa: E402
from jlens_support import MockModel  # noqa: E402


def _readout():
    m = MockModel()
    w = ReadoutWeights(arch="mock", n_layers=m.n_layers, d_model=m.d, n_vocab=m.vocab,
                       w_unembed=m.w_unembed, norm_weight=m.norm_weight, norm_bias=None,
                       norm_type="rms", eps=1e-5)
    lens = JacobianLensGGUF.identity(d_model=m.d, layers=list(range(m.n_layers - 1)))
    # give it h_rms so steer_vector has a scale
    lens.h_rms = {l: 10.0 for l in lens.source_layers}
    return LensReadout(w, lens), m


def test_abliterate_matrix_removes_direction():
    d, k, r = 16, 8, 2
    rng = np.random.default_rng(0)
    W = rng.standard_normal((d, k)).astype(np.float32)          # [d_model, k]
    dirs = rng.standard_normal((d, r)).astype(np.float32)
    Wp = exports.abliterate_matrix(W, dirs)
    Q, _ = np.linalg.qr(dirs)
    # every output column must be orthogonal to the direction span
    proj = Q.T @ Wp
    assert np.abs(proj).max() < 1e-4
    # the orthogonal-complement part is preserved
    W_perp = W - Q @ (Q.T @ W)
    np.testing.assert_allclose(Wp, W_perp, atol=1e-4)


def test_build_control_vector_accumulates():
    ro, _ = _readout()
    specs = [{"type": "steer", "token_id": 5, "alpha": 2.0, "layers": [0, 2]}]
    dirs = exports.build_control_vector(ro, specs)
    assert set(dirs) <= set(ro.lens.source_layers)
    # magnitude ~ alpha * h_rms
    for v in dirs.values():
        assert 15 < float(np.linalg.norm(v)) < 25  # 2 * 10, with lens-vector scaling


def test_build_control_vector_rejects_ablate():
    ro, _ = _readout()
    with pytest.raises(ValueError):
        exports.build_control_vector(ro, [{"type": "ablate", "token_id": 5}])


def test_export_control_vector_gguf_roundtrip(tmp_path):
    import gguf

    ro, _ = _readout()
    out = tmp_path / "cv.gguf"
    specs = [{"type": "steer", "token_id": 5, "alpha": 3.0, "layers": [0, 2]}]
    exports.export_control_vector(ro, specs, str(out), model_hint="mock")
    reader = gguf.GGUFReader(str(out))
    assert str(reader.fields["general.architecture"].contents()) == "controlvector"
    dirs = [t for t in reader.tensors if t.name.startswith("direction.")]
    assert dirs
    # 1-based layer indexing: lens layer l -> direction.{l+1}
    idxs = sorted(int(t.name.split(".")[1]) for t in dirs)
    assert min(idxs) >= 1
    # each direction is 1-D of width d_model, F32
    for t in dirs:
        assert list(t.shape) == [ro.weights.d_model]
        assert t.tensor_type == gguf.GGMLQuantizationType.F32


def test_concept_directions_unit():
    ro, _ = _readout()
    D = exports.concept_directions(ro, [5, 9])
    assert D.shape == (ro.weights.d_model, 2)
    for j in range(2):
        assert float(np.linalg.norm(D[:, j])) == pytest.approx(1.0, abs=1e-4)
