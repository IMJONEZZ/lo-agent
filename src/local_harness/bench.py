"""`lo bench` — run the harness's claimed advantages as actual measurements.

Every advantage we sell has to be a test a frontier API *fails*. This module runs
those tests against a live server and prints the local result next to the frontier
contrast. Each bench declares the capability it needs and SKIPs (honestly) when the
server can't do it — no green checkmark is ever printed for something we didn't run.

    lo bench --url http://HOST:PORT [--n 8]

Benches:
  determinism   two identical seeded calls must be bit-identical   (needs seed)
  batch-invar.  output unchanged under concurrent batch load        (needs seed;
                the audit-grade claim — generates real load, opt-in here only)
  grammar       N grammar-constrained gens, % valid-by-construction (needs grammar)
  uncertainty   sample-agreement separates a clear vs open prompt   (lexical signal)
  semantic-ent. meaning-clustered entropy separates determinate vs   (Farquhar 2024 —
                open — the SOTA signal; needs free sampling + a judge)
  cost          $0 local vs a frontier $/task estimate from a capture file (if present)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .inference.capabilities import Capabilities
from .inference.client import OpenAICompatClient
from .inference.types import GenerationRequest, Message, SamplingParams, canonical_text
from .tree.search.self_consistency import self_consistency


@dataclass
class BenchResult:
    name: str
    status: str  # PASS | FAIL | SKIP | INFO
    detail: str
    frontier: str = ""  # the contrast: what a frontier API does here


def _text(resp) -> str:
    return canonical_text(resp.raw["choices"][0].get("message", {}))


async def bench_determinism(
    client: OpenAICompatClient, caps: Capabilities
) -> BenchResult:
    if not caps.seed:
        return BenchResult(
            "determinism",
            "SKIP",
            "server doesn't honor a seed",
            "API output varies even at temp 0",
        )
    req = GenerationRequest(
        messages=[Message(role="user", content="Write one sentence about rivers.")],
        sampling=SamplingParams(temperature=1.0, seed=12345, max_tokens=64),
    )
    a = _text(await client.chat(req))
    b = _text(await client.chat(req))
    if a.strip() and a == b:
        return BenchResult(
            "determinism",
            "PASS",
            "2 seeded runs bit-identical",
            "API: nondeterministic (batch-variant kernels, no real seed)",
        )
    return BenchResult(
        "determinism", "FAIL", "seeded runs diverged", "API: also nondeterministic"
    )


async def bench_batch_invariance(
    client: OpenAICompatClient, caps: Capabilities, *, concurrency: int = 7
) -> BenchResult:
    """Determinism under concurrent load — the claim a frontier API can't even let
    you test. NOTE: this issues `concurrency`+1 requests at once; it's only run
    from `lo bench` (opt-in), never at connect-time."""
    if not caps.seed:
        return BenchResult(
            "batch-invariance",
            "SKIP",
            "server doesn't honor a seed",
            "API: nondeterministic even single-stream",
        )
    from .inference.capabilities import probe_batch_invariance

    ok = await probe_batch_invariance(client, caps, concurrency=concurrency)
    if ok is None:
        return BenchResult(
            "batch-invariance", "SKIP", "probe inconclusive (request error)", ""
        )
    if ok:
        return BenchResult(
            "batch-invariance",
            "PASS",
            f"output unchanged under {concurrency}-way concurrent load",
            "API: batch-variant kernels — output drifts with server load",
        )
    return BenchResult(
        "batch-invariance",
        "FAIL",
        f"output changed under {concurrency}-way load (batch-variant)",
        "API: same flaw, but you can't detect it (no seed, no replay)",
    )


async def bench_grammar(
    client: OpenAICompatClient, caps: Capabilities, skills_dir: str, n: int
) -> BenchResult:
    if caps.grammar is None:
        return BenchResult(
            "grammar",
            "SKIP",
            "no grammar/guided decoding on this server",
            "API exposes JSON-schema only",
        )
    try:
        from .skills.skill import SkillRegistry
        from .skills.exec import generate_with_skill

        skill = SkillRegistry(skills_dir).get("yes_no")
    except Exception as e:
        return BenchResult("grammar", "SKIP", f"no usable skill: {e}", "")
    valid = 0
    for i in range(n):
        r = await generate_with_skill(client, caps, skill, "Is water wet?", seed=i + 1)
        valid += int(r.valid)
    status = "PASS" if valid == n else "FAIL"
    return BenchResult(
        "grammar",
        status,
        f"{valid}/{n} valid by construction",
        "API: structured output can still violate hard grammars",
    )


async def bench_uncertainty(
    client: OpenAICompatClient, caps: Capabilities, n: int
) -> BenchResult:
    """The *correct* uncertainty signal: a clear question should produce high
    sample-agreement, an open one low — distinguishable only because local sampling
    is free. (This is the signal that replaced single-logprob 'confidence'.)"""
    clear = [Message(role="user", content="What is 2+2? Reply with just the number.")]
    openq = [Message(role="user", content="Write a random, unusual six-word sentence.")]
    _, agree_clear = await self_consistency(
        client,
        caps,
        clear,
        n=n,
        sampling=SamplingParams(temperature=1.0, max_tokens=16),
    )
    _, agree_open = await self_consistency(
        client,
        caps,
        openq,
        n=n,
        sampling=SamplingParams(temperature=1.0, max_tokens=32),
    )
    ok = agree_clear >= agree_open
    return BenchResult(
        "uncertainty",
        "PASS" if ok else "INFO",
        f"agreement clear={agree_clear:.2f} open={agree_open:.2f} (K={n}, free locally)",
        "API: logprobs metered/limited; this needs cheap unlimited sampling",
    )


async def bench_semantic_entropy(
    client: OpenAICompatClient, caps: Capabilities, n: int
) -> BenchResult:
    """Semantic entropy (Farquhar, Nature 2024): cluster K samples by *meaning*
    (entailment, judged by the local model) and take entropy over clusters. A
    determinate question should land in one meaning-cluster (≈0); an open one
    should spread across many (→1). Strictly more than lexical agreement, and
    needs the free sampling + free judge a frontier API can't offer."""
    from .signals.semantic_entropy import semantic_entropy

    determinate = [
        Message(
            role="user", content="What is the capital of France? Answer in one word."
        )
    ]
    openq = [
        Message(
            role="user",
            content="Name an interesting historical figure. Reply with just the name.",
        )
    ]
    det = await semantic_entropy(client, caps, determinate, n=n)
    opn = await semantic_entropy(client, caps, openq, n=n)
    ok = opn.normalized > det.normalized
    judge = (
        "" if det.judged and opn.judged else " (lexical fallback — judge unavailable)"
    )
    return BenchResult(
        "semantic-entropy",
        "PASS" if ok else "INFO",
        f"H_norm determinate={det.normalized:.2f}({det.n_clusters}c) "
        f"open={opn.normalized:.2f}({opn.n_clusters}c){judge}",
        "API: no free K-sampling and no free entailment judge to compute it",
    )


def bench_cost(capture: str | Path = "demos/frontier_cost_capture.json") -> BenchResult:
    p = Path(capture)
    frontier = "no capture file — run a frontier_*_capture to populate"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            frontier = json.dumps(data)[:120]
        except Exception:
            pass
    return BenchResult(
        "cost", "INFO", "$0.00 local marginal cost", f"frontier: {frontier}"
    )


async def run_bench(
    client: OpenAICompatClient,
    caps: Capabilities,
    skills_dir: str = "skills",
    n: int = 8,
    batch_invariance: bool = True,
) -> list[BenchResult]:
    results = [await bench_determinism(client, caps)]
    if batch_invariance:
        results.append(await bench_batch_invariance(client, caps))
    results += [
        await bench_grammar(client, caps, skills_dir, n),
        await bench_uncertainty(client, caps, n),
        await bench_semantic_entropy(client, caps, n),
        bench_cost(),
    ]
    return results


_MARK = {"PASS": "✓", "FAIL": "✗", "SKIP": "–", "INFO": "·"}


def format_report(
    results: list[BenchResult], *, model: str, server: str, tier: int
) -> str:
    lines = [
        "local_harness bench — advantages as measurements",
        f"server: {server} · tier {tier} · {model}",
        "",
    ]
    for r in results:
        lines.append(
            f"  {_MARK.get(r.status, '?')} {r.name:<13} {r.status:<5} {r.detail}"
        )
        if r.frontier:
            lines.append(f"      └ frontier: {r.frontier}")
    return "\n".join(lines)
