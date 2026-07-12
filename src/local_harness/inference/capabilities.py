"""Capability probing: fingerprint the server, verify dynamic features, report a tier.

Static capabilities come from the matched adapter (server families don't lie
about their sampler zoo); dynamic ones are verified with live test requests
because they depend on launch flags and model support:

- seed:        two identical seeded requests at temperature 1.0 must match
- logprobs:    a request with logprobs=true must return token logprobs
- raw completion: POST /v1/completions must accept a prompt
- kv_snapshot: GET /slots must exist (llama.cpp --slots)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import httpx

from .adapters import ADAPTERS
from .adapters.base import Fingerprint
from .client import OpenAICompatClient
from .types import GenerationRequest, Message, SamplingParams, canonical_text

PROBE_SEED = 424242


@dataclass
class Capabilities:
    server: str = "generic"
    model: str = ""
    seed: bool = False
    logprobs: bool = False
    grammar: str | None = None
    logit_bias: bool = False
    sampler_zoo: set[str] = field(default_factory=set)
    raw_completion: bool = False
    cfg_scale: bool = False
    banned_strings: bool = False
    kv_snapshot: bool = False
    parallel_n: bool = False
    stream_logprobs: bool = (
        False  # logprobs in a streamed tool-call request (vLLM yes, llama.cpp no)
    )
    # Per-token probabilities computed AFTER the sampler chain (llama.cpp
    # `post_sampling_probs`): confidence over what could actually be sampled
    # under min_p/XTC/etc., not the raw distribution they truncated away.
    post_sampling_probs: bool = False
    responses_api: bool = False  # /v1/responses with logprobs (e.g. LM Studio)
    logprobs_via_responses: bool = False  # logprobs come from /v1/responses, not chat
    in_process: bool = False  # Tier 4, native backend only (Phase 5)
    # Rung 6: activation access via a paired jlens lens service. `activations`
    # = read the residual stream (lens tab); `interventions` = steer/ablate/
    # swap. Either grants Tier 4 (like in_process) but over HTTP to a GGUF, not
    # a torch model on this box. lens_url/lens_method describe the pairing.
    activations: bool = False
    interventions: bool = False
    lens_url: str | None = None
    lens_method: str | None = None  # 'regression' | 'identity' | 'jacobian'
    lora_mode: str | None = None  # hot-swappable LoRA adapters: 'vllm'|'llamacpp'|None
    lora_adapters: list = field(default_factory=list)  # preloaded adapters (llama.cpp)
    # Loaded context-window size in tokens, read from the server (llama.cpp /props
    # n_ctx, vLLM /v1/models max_model_len, generic context_length). The agent
    # locks its auto-compaction trigger at a fraction of this. None = unknown.
    context_window: int | None = None
    # Determinism UNDER CONCURRENT LOAD (Thinking Machines 2025). None = not
    # probed (the probe generates real load, so it's opt-in via `lo bench`,
    # never run at connect-time). True/False once probed.
    batch_invariant: bool | None = None

    def tier(self) -> int:
        # Rung 6 activation access (in-process torch OR a paired lens service)
        # is Tier 4 — it's above the HTTP token-distribution rungs.
        if self.in_process or (self.activations and self.interventions):
            return 4
        tier = 0
        if self.seed and self.logprobs:
            tier = 1
            if self.grammar is not None and self.logit_bias:
                tier = 2
                if self.kv_snapshot or self.parallel_n:
                    tier = 3
        return tier

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["sampler_zoo"] = sorted(self.sampler_zoo)
        d["tier"] = self.tier()
        return d

    def summary(self) -> str:
        lines = [
            f"server:          {self.server}",
            f"model:           {self.model}",
            f"tier:            {self.tier()}",
            f"seed (verified): {self.seed}",
            f"logprobs:        {self.logprobs}",
            f"grammar:         {self.grammar or 'none'}",
            f"logit_bias:      {self.logit_bias}",
            f"sampler zoo:     {', '.join(sorted(self.sampler_zoo)) or 'none'}",
            f"raw completion:  {self.raw_completion}",
            f"cfg_scale:       {self.cfg_scale}",
            f"banned_strings:  {self.banned_strings}",
            f"kv_snapshot:     {self.kv_snapshot}",
            f"parallel_n:      {self.parallel_n}",
            f"responses_api:   {self.responses_api}",
            f"logprobs_via:    {'responses' if self.logprobs_via_responses else 'chat' if self.logprobs else 'none'}",
            f"post-sampling p: {self.post_sampling_probs}",
            f"lora (hot-swap): {self.lora_mode or 'none'}"
            + (f" · {len(self.lora_adapters)} preloaded" if self.lora_adapters else ""),
            f"context window:  {self.context_window or 'unknown'}"
            + (
                f" (auto-compact at {int(self.context_window * 0.85):,})"
                if self.context_window
                else ""
            ),
            f"batch-invariant: {'unprobed' if self.batch_invariant is None else self.batch_invariant}",
        ]
        if self.activations or self.interventions:
            mech = "in-process" if self.in_process else f"lens ({self.lens_method or 'identity'})"
            lines.append(
                f"activations:     read={self.activations} write={self.interventions}"
                f" · {mech}" + (f" @ {self.lens_url}" if self.lens_url else "")
            )
        return "\n".join(lines)


async def fingerprint(client: OpenAICompatClient) -> Fingerprint:
    fp = Fingerprint()
    try:
        resp = await client.get("/v1/models")
        if resp.status_code == 200:
            data = resp.json().get("data") or []
            if data:
                fp.owned_by = data[0].get("owned_by")
                fp.model_card = data[0]
    except (httpx.HTTPError, json.JSONDecodeError):
        pass
    for path, attr in (("/props", "has_props"), ("/slots", "has_slots")):
        try:
            resp = await client.get(path)
            if resp.status_code == 200:
                setattr(fp, attr, True)
                if path == "/props":
                    fp.props = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            pass
    return fp


def detect_context_window(fp: Fingerprint) -> int | None:
    """Read the loaded model's context-window size, in tokens, from whatever the
    server exposes — no metadata table, just the live endpoint:

    - llama.cpp / LM Studio: GET /props → default_generation_settings.n_ctx
      (the context the server was actually launched with), or a top-level n_ctx.
    - vLLM: GET /v1/models → data[0].max_model_len.
    - generic OpenAI-compat: data[0].{context_length,max_context_length,context_window}.

    Returns None when the server reports nothing usable (Tier-0 fallback: the
    agent then needs an explicit --context-budget to auto-compact)."""
    props = fp.props or {}
    gen = props.get("default_generation_settings") or {}
    for v in (gen.get("n_ctx"), props.get("n_ctx")):
        if isinstance(v, int) and v > 0:
            return v
    card = fp.model_card or {}
    for key in (
        "max_model_len",
        "context_length",
        "max_context_length",
        "context_window",
    ):
        v = card.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return None


def _probe_request(seed: int) -> GenerationRequest:
    return GenerationRequest(
        messages=[
            Message(role="user", content="Reply with one short sentence about rivers.")
        ],
        sampling=SamplingParams(
            temperature=1.0, max_tokens=16, seed=seed, logprobs=True, top_logprobs=2
        ),
    )


async def probe(client: OpenAICompatClient) -> Capabilities:
    fp = await fingerprint(client)
    adapter = next(a for a in ADAPTERS if a.matches(fp))
    static = adapter.static_caps(fp)

    caps = Capabilities(
        server=static.server,
        model=client.model,
        grammar=static.grammar,
        logit_bias=static.logit_bias,
        sampler_zoo=static.sampler_zoo,
        cfg_scale=static.cfg_scale,
        banned_strings=static.banned_strings,
        parallel_n=static.parallel_n,
        stream_logprobs=static.stream_logprobs,
        post_sampling_probs=static.post_sampling_probs,
        kv_snapshot=fp.has_slots,
        context_window=detect_context_window(fp),
    )

    # Dynamic: seed determinism + logprobs in one pair of test requests.
    # Compare the canonical message (content + reasoning_content + tool calls):
    # on reasoning models a short probe lands entirely in reasoning_content.
    try:
        a = await client.chat(_probe_request(PROBE_SEED))
        b = await client.chat(_probe_request(PROBE_SEED))
        text_a = canonical_text(a.raw["choices"][0]["message"])
        text_b = canonical_text(b.raw["choices"][0]["message"])
        caps.seed = text_a == text_b and bool(text_a.strip("|"))
        caps.logprobs = a.logprobs is not None and len(a.logprobs) > 0
    except httpx.HTTPError:
        pass

    # Dynamic: raw completion endpoint.
    try:
        await client.complete_raw("Once", {"max_tokens": 2, "temperature": 0.0})
        caps.raw_completion = True
    except httpx.HTTPError:
        pass

    # Dynamic: Open Responses endpoint with logprobs (LM Studio exposes token
    # logprobs ONLY here, not on chat-completions). We ONLY probe it when chat
    # logprobs were absent — i.e. a server that might be LM Studio. This keeps the
    # probe from ever issuing a generation on /v1/responses to a server that
    # already gave us chat logprobs (vLLM does), which previously could spin up a
    # large generation on vLLM's less-hardened Responses endpoint. The probe must
    # be gentle: it inspects capabilities, it does not stress the server.
    chat_logprobs = caps.logprobs
    if not chat_logprobs:
        try:
            r = await client.post(
                "/v1/responses",
                json={
                    "model": client.model,
                    "input": "Reply with one short word.",
                    "max_output_tokens": 512,
                    "include": ["message.output_text.logprobs"],
                    "top_logprobs": 2,
                },
            )
            if r.status_code == 200:
                caps.responses_api = True
                # LM Studio serves /v1/responses; pure llama.cpp does not. LM Studio
                # is built on llama.cpp (also serves /props) so it fingerprints as
                # llama.cpp — correct the label so it isn't taken for runtime-LoRA.
                if caps.server == "llama.cpp":
                    caps.server = "lmstudio"
                if _responses_has_logprobs(r.json()):
                    caps.logprobs = True
                    caps.logprobs_via_responses = True
                    client.logprobs_via_responses = True  # transparent routing
        except (httpx.HTTPError, json.JSONDecodeError):
            pass

    # Hot-swappable LoRA adapters (skills-as-adapters): vLLM serves them as model
    # names + runtime-loadable; llama.cpp exposes its preloaded set at /lora-adapters.
    from .lora import probe_lora

    await probe_lora(client, caps)

    # Rung 6: a paired lens service (activations over HTTP). Cheap health check
    # only — NEVER spawns a sidecar or runs a capture pass. Opt-in via a
    # configured lens_url (LO_LENS_URL / config), so connect-time stays light.
    lens_url = getattr(client, "lens_url", None)
    if lens_url:
        await probe_lens(caps, lens_url)

    return caps


async def probe_lens(caps: Capabilities, lens_url: str) -> None:
    """Set activation/intervention flags from a lens service's /health.

    Does not build or start anything — if the service is up, its /health
    reports the lens method and readable-layer count. Absent/unreachable → the
    flags stay False and the tier is unaffected.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(lens_url.rstrip("/") + "/health")
        if r.status_code != 200:
            return
        data = r.json()
        if data.get("status") != "ok":
            return
        caps.activations = True
        caps.interventions = True  # any lens service can translate steer/ablate/swap
        caps.lens_url = lens_url
        caps.lens_method = (data.get("lens") or {}).get("method")
    except (httpx.HTTPError, json.JSONDecodeError):
        pass


def _responses_has_logprobs(data: dict) -> bool:
    """Defensive scan for token logprobs anywhere in an Open Responses payload."""
    for item in data.get("output") or []:
        for block in item.get("content") or []:
            if block.get("logprobs"):
                return True
    return bool(data.get("logprobs"))


# Decoy load for the batch-invariance probe: deliberately varied prompts and
# lengths so the server's batch composition genuinely differs from the baseline.
_BATCH_DECOYS: list[tuple[str, int]] = [
    ("List three fruits.", 24),
    ("Explain why the sky is blue, in detail.", 96),
    ("Say hello.", 8),
    ("Write a short poem about the sea.", 64),
    ("What is 17 times 23? Show your work.", 48),
    ("Describe a sunset in one sentence.", 32),
    ("Count from one to ten.", 20),
    ("Name a country in Europe.", 12),
]


def _batch_target(seed: int) -> GenerationRequest:
    return GenerationRequest(
        messages=[Message(role="user", content="Write two sentences about mountains.")],
        sampling=SamplingParams(temperature=1.0, max_tokens=64, seed=seed),
    )


async def probe_batch_invariance(
    client: OpenAICompatClient,
    caps: Capabilities,
    *,
    concurrency: int = 7,
    seed: int = PROBE_SEED,
) -> bool | None:
    """Determinism UNDER CONCURRENT LOAD — the audit-grade reproducibility claim.

    Two seeded calls matching back-to-back (`caps.seed`) only proves the easy
    case. The real nondeterminism source is batch-size-varying reductions: a
    request's output can change with whatever else is batched alongside it on the
    server (Thinking Machines, *Defeating Nondeterminism in LLM Inference*, 2025).
    A frontier API can't even let you test this; here we can.

    Fires the SAME seeded request alone, then again amid `concurrency` decoy
    requests of varying length, and checks the target output is unchanged. Sets
    and returns `caps.batch_invariant`; returns None (leaves it unprobed) if the
    server has no seed or a request errored.

    GENERATES REAL CONCURRENT LOAD — opt-in only (`lo bench`), never part of
    the connect-time probe, so it can't slam a shared server unasked.
    """
    if not caps.seed:
        return None
    target = _batch_target(seed)
    try:
        baseline = canonical_text(
            (await client.chat(target)).raw["choices"][0]["message"]
        )
    except httpx.HTTPError:
        return None
    decoys = [
        GenerationRequest(
            messages=[Message(role="user", content=prompt)],
            sampling=SamplingParams(temperature=1.0, max_tokens=mt),
        )
        for prompt, mt in _BATCH_DECOYS[: max(0, concurrency)]
    ]
    results = await asyncio.gather(
        client.chat(_batch_target(seed)),
        *(client.chat(d) for d in decoys),
        return_exceptions=True,
    )
    target_result = results[0]
    if isinstance(target_result, BaseException):
        return None
    under_load = canonical_text(target_result.raw["choices"][0]["message"])
    caps.batch_invariant = baseline == under_load and bool(baseline.strip("|"))
    return caps.batch_invariant
