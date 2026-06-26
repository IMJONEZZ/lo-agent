"""TUI --server (thin client) plumbing: log-mirror primitives + caps reconstruction.

The full server-mode flow (probe /health → POST /session → mirror SSE → render) is
covered by a live smoke against `harness serve`; here we lock the pure pieces that
make it work without standing up Textual.
"""

from local_harness.events.log import EventLog, MODEL_CALL, RUN_COMPLETED
from local_harness.inference.capabilities import Capabilities
from local_harness.tui.app import _caps_from_health


def test_event_log_mirror_primitives(tmp_path):
    log = EventLog(tmp_path / "m.db")
    log.ensure_run("srv123", "do a thing")
    assert log.run("srv123").status == "running"
    log.ensure_run("srv123", "do a thing")  # idempotent — no duplicate row / no reset

    log.import_event("srv123", MODEL_CALL, {"call_index": 0})
    log.import_event("srv123", RUN_COMPLETED, {"answer": "hi"})
    assert log.run("srv123").status == "completed"   # terminal flips status, like append
    assert [e.type for e in log.events("srv123")] == [MODEL_CALL, RUN_COMPLETED]
    assert [e.seq for e in log.events("srv123")] == [0, 1]  # ordered, no run_started row


def test_caps_from_health_reconstructs_tier():
    real = Capabilities(server="llama.cpp", seed=True, logprobs=True,
                        grammar="gbnf", logit_bias=True, kv_snapshot=True,
                        sampler_zoo={"min_p", "dry"})
    caps = _caps_from_health({"model": "m", "capabilities": real.to_dict()})
    assert caps is not None
    assert caps.server == "llama.cpp"
    assert caps.tier() == real.tier()             # full round-trip incl. derived tier
    assert caps.sampler_zoo == {"min_p", "dry"}   # list re-coerced to set


def test_caps_from_health_handles_empty():
    assert _caps_from_health({}) is None
    assert _caps_from_health({"capabilities": None}) is None
