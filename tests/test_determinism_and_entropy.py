"""Thread-3 deep pass: batch-invariant determinism (3a) + semantic entropy (3b).

These lock the contracts:
  - batch-invariance compares the SAME seeded request alone vs under concurrent
    load; a batch-variant server (output drifts when other requests are in flight)
    must be caught, and it must never run at connect-time.
  - semantic entropy clusters samples by MEANING via an entailment judge, so
    "Paris" and "It's Paris." collapse to one cluster while distinct facts don't.
"""

import json

import httpx
import pytest

from local_harness.inference.capabilities import (
    Capabilities,
    probe_batch_invariance,
    probe,
)
from local_harness.inference.client import OpenAICompatClient
from local_harness.inference.types import Message
from local_harness.signals.semantic_entropy import semantic_entropy, _verdict_yes


# --- 3a: batch invariance -------------------------------------------------


class _BatchVariantServer(httpx.AsyncBaseTransport):
    """A server whose output for a fixed seed DRIFTS once other requests have
    been seen — the batch-size-varying-reduction failure mode. The target's
    answer depends on how many calls happened before it, so firing it amid
    decoys changes its output."""

    def __init__(self):
        self.calls = 0

    async def handle_async_request(self, request):
        p = request.url.path
        if p == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "owned_by": "acme"}]})
        body = json.loads(request.content)
        self.calls += 1
        # 'mountains' is the target prompt; its content depends on total load seen
        is_target = "mountains" in json.dumps(body.get("messages", []))
        content = f"target-load-{self.calls}" if is_target else "decoy"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )


class _BatchInvariantServer(httpx.AsyncBaseTransport):
    """Output for the target is a pure function of its seed — unaffected by any
    concurrent decoys (a single-slot llama.cpp, or vLLM in batch-invariant mode)."""

    async def handle_async_request(self, request):
        p = request.url.path
        if p == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "owned_by": "acme"}]})
        body = json.loads(request.content)
        is_target = "mountains" in json.dumps(body.get("messages", []))
        content = f"seed-{body.get('seed')}" if is_target else "decoy"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )


async def test_batch_invariance_pass_on_invariant_server():
    client = OpenAICompatClient("http://x", "m", transport=_BatchInvariantServer())
    caps = Capabilities(server="llama.cpp", seed=True)
    ok = await probe_batch_invariance(client, caps, concurrency=7)
    assert ok is True and caps.batch_invariant is True


async def test_batch_invariance_catches_variant_server():
    client = OpenAICompatClient("http://x", "m", transport=_BatchVariantServer())
    caps = Capabilities(server="vllm", seed=True)
    ok = await probe_batch_invariance(client, caps, concurrency=7)
    assert ok is False and caps.batch_invariant is False


async def test_batch_invariance_skips_without_seed():
    client = OpenAICompatClient("http://x", "m", transport=_BatchInvariantServer())
    caps = Capabilities(server="generic", seed=False)
    assert await probe_batch_invariance(client, caps) is None
    assert caps.batch_invariant is None  # left unprobed, not False


async def test_connect_probe_never_sets_batch_invariant():
    """The connect-time probe must NOT run the load-generating batch check."""
    from mocks import MockLlamaCpp

    client = OpenAICompatClient(
        "http://x", "test-model", transport=MockLlamaCpp().transport()
    )
    caps = await probe(client)
    assert caps.batch_invariant is None  # only `lo bench` probes it


# --- 3b: semantic entropy -------------------------------------------------


class _MeaningServer(httpx.AsyncBaseTransport):
    """Serves K samples for the question, then acts as an entailment judge.
    `samples` are returned in order (seeded calls); the judge says yes iff the
    two answers are in the same `meaning_groups` bucket."""

    def __init__(self, samples, meaning_groups):
        self._samples = samples
        self._i = 0
        self._groups = meaning_groups  # list[set[str]]

    def _group_of(self, text):
        for gi, g in enumerate(self._groups):
            if any(tok in text.lower() for tok in g):
                return gi
        return -1

    async def handle_async_request(self, request):
        p = request.url.path
        if p == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "owned_by": "acme"}]})
        body = json.loads(request.content)
        msgs = body.get("messages", [])
        sys = msgs[0].get("content", "") if msgs else ""
        if "entailment" in sys or "same information" in sys:
            user = msgs[-1]["content"]
            # parse "Answer A: ...\nAnswer B: ..." and judge same-group
            a = user.split("Answer A:")[1].split("Answer B:")[0].strip()
            b = user.split("Answer B:")[1].split("Does")[0].strip()
            verdict = "yes" if self._group_of(a) == self._group_of(b) >= 0 else "no"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": verdict},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        # a sampling call — return the next scripted sample
        s = self._samples[self._i % len(self._samples)]
        self._i += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": s},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )


async def test_semantic_entropy_collapses_same_meaning():
    # three surface forms, ALL meaning 'Paris' → one cluster → entropy 0
    server = _MeaningServer(
        ["Paris", "It's Paris.", "The answer is Paris"], meaning_groups=[{"paris"}]
    )
    client = OpenAICompatClient("http://x", "m", transport=server)
    caps = Capabilities(server="llama.cpp", seed=True)
    res = await semantic_entropy(
        client, caps, [Message(role="user", content="Capital of France?")], n=3
    )
    assert res.n_clusters == 1
    assert res.normalized == 0.0
    assert res.judged is True


async def test_semantic_entropy_separates_distinct_meanings():
    # three genuinely different answers → three clusters → max entropy
    server = _MeaningServer(
        ["Paris", "London", "Berlin"],
        meaning_groups=[{"paris"}, {"london"}, {"berlin"}],
    )
    client = OpenAICompatClient("http://x", "m", transport=server)
    caps = Capabilities(server="llama.cpp", seed=True)
    res = await semantic_entropy(
        client, caps, [Message(role="user", content="A city?")], n=3
    )
    assert res.n_clusters == 3
    assert res.normalized == pytest.approx(1.0)  # log3 / log3


async def test_semantic_entropy_open_exceeds_determinate():
    determinate = _MeaningServer(
        ["Paris", "Paris", "It is Paris"], meaning_groups=[{"paris"}]
    )
    c1 = OpenAICompatClient("http://x", "m", transport=determinate)
    caps = Capabilities(server="llama.cpp", seed=True)
    det = await semantic_entropy(
        c1, caps, [Message(role="user", content="Capital of France?")], n=3
    )

    openq = _MeaningServer(
        ["Cleopatra", "Newton", "Genghis Khan"],
        meaning_groups=[{"cleopatra"}, {"newton"}, {"genghis"}],
    )
    c2 = OpenAICompatClient("http://x", "m", transport=openq)
    opn = await semantic_entropy(
        c2, caps, [Message(role="user", content="A historical figure?")], n=3
    )
    assert opn.normalized > det.normalized


def test_verdict_is_the_last_yes_no():
    assert _verdict_yes("Hmm, at first no, but they match, so yes.") is True
    assert _verdict_yes("They seem similar (yes?) but ultimately no.") is False
    assert _verdict_yes("nothing here") is False
