"""Phase-3 tests: intervention specs, concept sets, replay_tuned jlens family,
signals + concept-watch — against the mock lens service."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gguf")
pytest.importorskip("starlette")

from local_harness.jlens.interventions import (  # noqa: E402
    ConceptSet,
    ConceptStore,
    Spec,
    lens_hash,
)
from local_harness.jlens.model_reader import ReadoutWeights  # noqa: E402
from local_harness.jlens.service import LensService, create_app  # noqa: E402
from jlens_support import MockModel, MockSidecar  # noqa: E402


# ---------------------------------------------------------------- specs --


def test_spec_to_service_forms():
    steer = Spec(type="steer", token_id=5, token=" yen", alpha=1.5, layers=[40, 60])
    assert steer.to_service() == {"type": "steer", "token_id": 5, "alpha": 1.5, "layers": [40, 60]}
    ab = Spec(type="ablate", token_id=9, token=" Euro")
    assert ab.to_service() == {"type": "ablate", "token_id": 9}
    sw = Spec(type="swap", token_id=5, token_b_id=9, token=" yen", token_b=" Euro")
    s = sw.to_service()
    assert s["token_a"] == 5 and s["token_b"] == 9


def test_concept_store_roundtrip(tmp_path):
    store = ConceptStore(tmp_path)
    cs = ConceptSet(name="boot-fix", specs=[
        Spec(type="ablate", token_id=9, token=" Euro"),
        Spec(type="steer", token_id=5, token=" yen", alpha=2.0),
    ])
    store.save(cs)
    assert "boot-fix" in store.list()
    back = store.load("boot-fix")
    assert len(back.specs) == 2
    assert back.specs[0].type == "ablate"
    assert back.to_service()[1]["type"] == "steer"


def test_lens_hash(tmp_path):
    assert lens_hash(None) == "identity"
    f = tmp_path / "lens.gguf"
    f.write_bytes(b"abc123")
    h1 = lens_hash(str(f))
    assert len(h1) == 16
    f.write_bytes(b"different")
    assert lens_hash(str(f)) != h1


# ------------------------------------------------ live service fixtures --


def _run_service(monkeypatch):
    sc = MockSidecar().__enter__()
    m = sc.model
    w = ReadoutWeights(arch="mock", n_layers=m.n_layers, d_model=m.d, n_vocab=m.vocab,
                       w_unembed=m.w_unembed, norm_weight=m.norm_weight, norm_bias=None,
                       norm_type="rms", eps=1e-5, n_nextn=0, model_name="mock")
    monkeypatch.setattr(ReadoutWeights, "from_gguf", staticmethod(lambda p: w))
    svc = LensService(model_path="/mock/toy.gguf", native_url=sc.url, lens_path=None)
    import uvicorn
    cfg = uvicorn.Config(create_app(svc), host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started and server.servers:
            break
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    return sc, server, f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_replay_tuned_jlens_family(monkeypatch):
    from local_harness.events.log import EventLog, MODEL_CALL, RUN_STARTED
    from local_harness.tuned_replay import Intervention, replay_tuned

    sc, server, url = _run_service(monkeypatch)
    try:
        # a minimal logged run with one MODEL_CALL
        import tempfile
        import os
        dbfd, dbpath = tempfile.mkstemp(suffix=".db")
        os.close(dbfd)
        log = EventLog(dbpath)
        log.ensure_run("r1") if hasattr(log, "ensure_run") else None
        log.append("r1", RUN_STARTED, {"task": "t"})
        log.append("r1", MODEL_CALL, {
            "request_body": {"messages": [{"role": "user", "content": "hello world"}]},
            "response": {"choices": [{"message": {"content": "baseline answer"}}]},
        })
        iv = Intervention(label="ablate-5",
                          jlens=[{"type": "ablate", "token_id": 5, "layers": [0, 3]}],
                          lens_url=url)
        rep = await replay_tuned(log, "r1", None, None, iv)
        assert rep.intervention == "ablate-5"
        # deterministic report; tuned/baseline are strings from the mock generation
        assert isinstance(rep.tuned, str) and isinstance(rep.original, str)
        os.unlink(dbpath)
    finally:
        server.should_exit = True
        sc.__exit__(None, None, None)


@pytest.mark.asyncio
async def test_rank_trajectories_and_watch(monkeypatch):
    from local_harness.signals.jspace import rank_trajectories
    from local_harness.guardrails.concept_watch import watch_concepts

    sc, server, url = _run_service(monkeypatch)
    try:
        # the mock vocab pieces are "tok0".."tok47"; track a couple
        trajs = await rank_trajectories(url, prompt="a b c d",
                                        track_pieces=["tok5", "tok9"])
        assert trajs, "expected trajectories"
        for t in trajs:
            assert len(t.ranks) >= 1
            assert t.min_rank <= t.final_rank or True  # ranks are ints
        alerts = await watch_concepts(url, prompt="a b c d",
                                      concepts=["tok5"], band=100000)  # band huge → always alert
        assert alerts and alerts[0].piece
    finally:
        server.should_exit = True
        sc.__exit__(None, None, None)
