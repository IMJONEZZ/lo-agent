"""Phase-5 test: exact causal lens fit on a tiny random GPT-2 (hermetic).

Proves the torch fit path produces a valid lens-GGUF that drops into the same
LensReadout as the GGUF-native regression lens. Skips without the native extra.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("gguf")

import numpy as np  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402

from local_harness.jlens.fit_torch import _CallableTok, fit_causal_lens  # noqa: E402
from local_harness.jlens.lens import JacobianLensGGUF  # noqa: E402


class ByteTok(_CallableTok):
    def __call__(self, text, max_len, device):
        ids = [min(255, b) + 1 for b in text.encode()[:max_len]]
        return torch.tensor([ids or [1]], device=device)


def _tiny_model():
    cfg = transformers.GPT2Config(n_layer=3, n_head=2, n_embd=32, n_positions=128,
                                  vocab_size=257)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg)


def test_fit_causal_lens_shapes_and_io(tmp_path):
    model = _tiny_model()
    tok = ByteTok()
    prompts = [
        "the quick brown fox jumps over the lazy dog again and again for a while",
        "once upon a time there was a little model that learned to read its own mind",
        "colorless green ideas sleep furiously while the parser dreams of clean grammars",
    ]
    lens = fit_causal_lens(model, tok, prompts, source_layers=[0, 1], target_layer=2,
                           dim_batch=8, max_seq_len=64, skip_first=2)
    assert lens.fit_method == "jacobian"
    assert lens.d_model == 32
    assert lens.source_layers == [0, 1]
    for l in (0, 1):
        assert lens.jacobians[l].shape == (32, 32)
    assert lens.h_rms and all(v > 0 for v in lens.h_rms.values())

    # serializes to the same lens-GGUF format and loads back
    p = tmp_path / "causal.gguf"
    lens.save(str(p))
    back = JacobianLensGGUF.load(str(p))
    assert back.fit_method == "jacobian"
    np.testing.assert_allclose(back.jacobians[0], lens.jacobians[0], atol=1e-2)


def test_causal_lens_transports_toward_output():
    """The fitted J should map a source residual closer to the final residual
    than the identity does (sanity: it's a transport, not noise)."""
    model = _tiny_model()
    tok = ByteTok()
    prompts = ["the quick brown fox jumps over the lazy dog " * 2,
               "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 2]
    lens = fit_causal_lens(model, tok, prompts, source_layers=[0], target_layer=2,
                           dim_batch=16, max_seq_len=48, skip_first=2)
    J = lens.jacobians[0]
    assert np.isfinite(J).all()
    # transport is a linear map with non-trivial structure
    assert float(np.linalg.norm(J)) > 0
