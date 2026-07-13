"""Phase-1 tests: lens service API + capability probe, against the mock sidecar."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gguf")
pytest.importorskip("starlette")

import numpy as np  # noqa: E402

from starlette.testclient import TestClient  # noqa: E402

from local_harness.inference.capabilities import Capabilities, probe_lens  # noqa: E402
from local_harness.jlens import manager  # noqa: E402
from local_harness.jlens.model_reader import ReadoutWeights  # noqa: E402
from local_harness.jlens.service import LensService, create_app  # noqa: E402
from jlens_support import MockModel, MockSidecar  # noqa: E402


def _fake_weights(monkeypatch, model: MockModel):
    """Patch ReadoutWeights.from_gguf so the service doesn't need a real GGUF."""
    w = ReadoutWeights(arch="mock", n_layers=model.n_layers, d_model=model.d,
                       n_vocab=model.vocab, w_unembed=model.w_unembed,
                       norm_weight=model.norm_weight, norm_bias=None,
                       norm_type="rms", eps=1e-5, n_nextn=0, model_name="mock")
    monkeypatch.setattr(ReadoutWeights, "from_gguf", staticmethod(lambda p: w))
    return w


def _service(monkeypatch, sc):
    _fake_weights(monkeypatch, sc.model)
    return LensService(model_path="/mock/toy.gguf", native_url=sc.url, lens_path=None)


def test_service_health_and_slice(monkeypatch):
    with MockSidecar() as sc:
        app = create_app(_service(monkeypatch, sc))
        client = TestClient(app)
        h = client.get("/health").json()
        assert h["status"] == "ok"
        assert h["n_layers"] == sc.model.n_layers
        sl = client.post("/lens/slice", json={"prompt": "a b c", "top_n": 3}).json()
        assert "ctx_id" in sl
        assert len(sl["pieces"]) == len(sl["tokens"])
        # ranks on the returned context
        r = client.post("/lens/ranks", json={"ctx_id": sl["ctx_id"], "token_ids": [5, 9]}).json()
        assert r["shape"][2] == 2
        ro = client.post("/lens/readout", json={"ctx_id": sl["ctx_id"], "pos": 0,
                                                "layer": sl["layers"][-1], "top_n": 4}).json()
        assert len(ro["tokens"]) == 4


def test_steer_defaults_to_prompt_only(monkeypatch):
    """The doctrine: steer with no explicit pos scopes to prompt positions."""
    with MockSidecar() as sc:
        svc = _service(monkeypatch, sc)
        toks = svc.client.tokenize("one two three")
        native = svc.translate_interventions(
            [{"type": "steer", "token_id": 5, "alpha": 1.0, "layers": [0, 3]}], toks)
        assert native, "expected at least one native edit"
        # every steer edit must be bounded to the prompt, not open-ended (-1)
        for iv in native:
            assert iv["pos_end"] == len(toks)
        # ablate, by contrast, defaults to all positions
        nat_ab = svc.translate_interventions(
            [{"type": "ablate", "token_id": 5, "layers": [0, 3]}], toks)
        assert all(iv["pos_end"] == -1 for iv in nat_ab)


def test_service_intervention_changes_capture(monkeypatch):
    """A steer edit at a fitted layer changes that layer's lens readout grid."""
    import base64

    with MockSidecar() as sc:
        app = create_app(_service(monkeypatch, sc))
        client = TestClient(app)
        base = client.post("/lens/slice", json={"prompt": "hello world", "top_n": 3}).json()
        steered = client.post("/lens/slice", json={
            "prompt": "hello world", "top_n": 3,
            "interventions": [{"type": "steer", "token_id": 5, "alpha": 8.0,
                               "layers": [0, 2], "pos": [0, -1]}]}).json()
        b = np.frombuffer(base64.b64decode(base["top_ids"]), "<i4")
        s = np.frombuffer(base64.b64decode(steered["top_ids"]), "<i4")
        # a strong add edit at fitted layers must move at least one top-1 token
        assert not np.array_equal(b, s)


@pytest.mark.asyncio
async def test_probe_lens_sets_tier4(monkeypatch):
    with MockSidecar() as sc:
        app = create_app(_service(monkeypatch, sc))
        # run the service on a real socket for the async probe
        import threading
        import uvicorn
        cfg = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        server = uvicorn.Server(cfg)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        import time
        for _ in range(100):
            if server.started and server.servers:
                break
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        caps = Capabilities(server="llama.cpp", seed=True, logprobs=True,
                            grammar="gbnf", logit_bias=True, parallel_n=True)
        assert caps.tier() == 3
        await probe_lens(caps, f"http://127.0.0.1:{port}")
        assert caps.activations and caps.interventions
        assert caps.tier() == 4
        server.should_exit = True


def test_check_l_out_ok_mtp():
    """MTP false-negative handling: 64 emitting layers match readable count."""
    w = ReadoutWeights(arch="qwen35", n_layers=65, d_model=8, n_vocab=10,
                       w_unembed=np.zeros((10, 8), np.float32), norm_weight=None,
                       norm_bias=None, norm_type="rms", eps=1e-5, n_nextn=1)
    ok, why = manager.check_l_out_ok({"l_out_ok": False, "n_layer": 64}, w)
    assert ok, why
    ok2, _ = manager.check_l_out_ok({"l_out_ok": False, "n_layer": 30}, w)
    assert not ok2


def test_resolve_model_gguf_from_path(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF")
    assert manager.resolve_model_gguf(model=str(f)) == str(f)
    assert manager.resolve_model_gguf(model=str(tmp_path / "nope.gguf")) is None
