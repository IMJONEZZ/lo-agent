"""Live, in-TUI demonstrations of local-LLM advantages.

Each coroutine runs ONE advantage against the *connected* endpoint and returns a
Rich renderable for the transcript. These back the TUI slash-commands
(/samplers /antislop /overlay /consistency /escalate /bestof /thinkbudget) so the
advantages are things you can actually *do* in the harness.

Every advantage takes an optional `arg`: whatever the user types after the slash
command becomes the prompt, e.g. `/overlay Is a 30y Treasury suitable here?`.
With no arg each falls back to its built-in prompt. `/grammar N` sets the count;
`/antislop word1, word2 | prompt` sets the banned words.

Reasoning models (Step, GLM) ignore enable_thinking and reason in-channel, so we
give every call a generous token budget: reasoning is free locally, and a model
that reasons to the ceiling without a budget never emits its answer.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from ..inference.types import GenerationRequest, Message, SamplingParams
from . import render

# enable_thinking=False lets non-reasoning models answer directly; reasoning models
# ignore it (we give them budget to think instead).
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
_BUDGET = 2048  # generous, so reasoning models finish thinking and emit real content


def _text(resp) -> str:
    msg = resp.raw.get("choices", [{}])[0].get("message", {})
    return (msg.get("content") or msg.get("reasoning_content")
            or msg.get("reasoning") or "").strip()


async def _stream(client, messages, sampling, on_token):
    """Stream a generation, feeding each token to on_token(kind, text) so the TUI
    can type it out live — BOTH reasoning ('thinking…') and content, so a slow
    reasoning model visibly works instead of looking hung. Returns (text, assembled
    response) — identical shape to a non-streamed call, so downstream is unchanged."""
    body = GenerationRequest(messages=messages, sampling=sampling).to_body(client.model)

    def on_delta(kind, text):
        if text and on_token and kind in ("content", "reasoning"):
            on_token(kind, text)
    resp = await client.chat_body_stream(body, on_delta)
    return _text(resp), resp


async def _gen(client, prompt, *, extra=None, max_tokens=_BUDGET, temperature=0.8,
               seed=3, logprobs=False, on_token=None):
    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed,
                              logprobs=logprobs, top_logprobs=2 if logprobs else None,
                              extra={**_NO_THINK, **(extra or {})})
    msgs = [Message(role="user", content=prompt)]
    if on_token is not None:
        return await _stream(client, msgs, sampling, on_token)
    r = await client.chat(GenerationRequest(messages=msgs, sampling=sampling))
    return _text(r), r


async def _stream_samples(client, messages, sampling, n, base_seed, on_sample):
    """Fan out N rollouts as CONCURRENT streaming requests, each with its own seed,
    feeding on_sample(i, kind, text) per token so each gets its own live row. On a
    parallel-slot server all N stream at once (the free-tokens fan-out made visible);
    on a single-slot server they fill as slots free. Returns [(text, response), …]."""
    async def one(i):
        sp = replace(sampling, seed=base_seed + i, extra=dict(sampling.extra or {}))
        body = GenerationRequest(messages=messages, sampling=sp).to_body(client.model)

        def on_delta(kind, text):
            if text and kind in ("content", "reasoning"):
                on_sample(i, kind, text)
        try:
            resp = await client.chat_body_stream(body, on_delta)
            return _text(resp), resp
        except Exception:  # noqa: BLE001 — one rollout failing must not sink the fan-out
            return "", None
    return await asyncio.gather(*[one(i) for i in range(n)])


async def _fan_consensus(client, caps, msgs, n, key, sampling, live, base_seed, title,
                         agent_factory=None, semantic=None):
    """Sample N rollouts (streamed into fan rows when live), vote via `key`, and
    return (representative answer, agreement fraction, groups) — where groups is
    [(vote key, count, representative text), …] most-agreed first. The caller needs
    the spread, not just the winner: reading *which* answers disagree is the signal."""
    from ..tree.search.self_consistency import collect_samples
    question = next((m.content for m in reversed(msgs) if m.role == "user"), "")
    if agent_factory is not None:
        # Each rollout is a full agent run: it may retrieve and call tools before
        # answering, so the agreement we measure is over *grounded* answers.
        if live is not None:
            live.fan([str(i + 1) for i in range(n)], title)

        async def one(i):
            cb = None
            if live is not None:
                def cb(kind, text, _i=i):   # noqa: E306
                    if kind in ("content", "reasoning"):
                        live.fan_token(_i, kind, text)
            try:
                ans, _lp = await _tool_answer(agent_factory, caps, question, base_seed + i, cb)
            except Exception:  # noqa: BLE001 — one rollout must not sink the fan-out
                ans = ""
            if live is not None:
                live.fan_done(i, (key(ans) if ans else "—")[:24])
            return ans
        texts = [t for t in await asyncio.gather(*[one(i) for i in range(n)]) if t]
    elif live is None:
        texts = [t for t in await collect_samples(
            client, caps, msgs, n=n, sampling=sampling, base_seed=base_seed) if t]
    else:
        live.fan([str(i + 1) for i in range(n)], title)
        results = await _stream_samples(client, msgs, sampling, n, base_seed, live.fan_token)
        for i, (t, _) in enumerate(results):
            live.fan_done(i, key(t) if t else "—")
        texts = [t for t, _ in results if t]
    if not texts:
        return "", 0.0, [], False
    groups, sem = await _group(client, question, texts, key, semantic)
    return groups[0][2], groups[0][1] / len(texts), groups, sem


# Answers at or under this length are treated as "short form" (a number, T+1,
# yes/no) where exact-match voting is correct. Longer prose is meaning-clustered
# instead — surface-form voting reports false uncertainty on paraphrases.
_SHORT_FORM = 48


def _parse_arg(arg: str) -> tuple[set[str], str]:
    """Split leading `--flags` off an advantage argument.

    `--tools What is the settlement cycle?` → ({"tools"}, "What is the …?")
    Recognized: --tools (agent loop w/ retrieval), --semantic / --lexical
    (force the /consistency grouping mode instead of auto-detecting)."""
    flags: set[str] = set()
    rest = (arg or "").strip()
    while rest.startswith("--"):
        tok, _, remainder = rest.partition(" ")
        flags.add(tok[2:].strip().lower())
        rest = remainder.strip()
    return flags, rest


def _final_logprobs(log, run_id):
    """Per-token logprobs of the agent's LAST model call — i.e. the confidence of
    the answer it gave *after* its tools came back."""
    from ..events.log import MODEL_CALL
    from ..inference.types import GenerationResponse
    evs = log.events(run_id, type=MODEL_CALL)
    if not evs:
        return None
    try:
        return GenerationResponse.from_chat_response(
            evs[-1].payload.get("response") or {}, 0.0).logprobs
    except Exception:  # noqa: BLE001 — confidence is best-effort here
        return None


async def _tool_answer(agent_factory, caps, prompt, seed, on_token=None):
    """One tool-enabled agent run → (answer, final-call logprobs).

    The agent retrieves and calls tools; we then measure confidence on the answer
    it produced *with* that evidence, which is the number worth reading."""
    agent = agent_factory(on_token)
    agent.base_seed = seed
    if getattr(caps, "logprobs", False):
        agent.sampling = replace(agent.sampling, logprobs=True, top_logprobs=2)
    res = await agent.run(prompt)
    return (res.answer or "").strip(), _final_logprobs(agent.log, res.run_id)


async def _group(client, question, texts, key, semantic):
    """Group samples for voting → ([(label, count, representative)], semantic?).

    Lexical exact-match for short canonical answers; meaning-clustering (semantic
    entropy, Farquhar 2024) for prose, where paraphrases of one answer would
    otherwise read as disagreement. `semantic=None` auto-detects on answer length."""
    from collections import Counter
    if semantic is None:
        semantic = not all(len(_flat(t)) <= _SHORT_FORM for t in texts)
    if not semantic:
        counts = Counter(key(t) for t in texts)
        return ([(k, c, next(t for t in texts if key(t) == k))
                 for k, c in counts.most_common()], False)
    from ..signals.semantic_entropy import _cluster
    clusters, _judged = await _cluster(client, question, texts)
    clusters = sorted(clusters, key=len, reverse=True)
    return ([(c[0], len(c), c[0]) for c in clusters], True)


def _flat(t: str, limit: int | None = None) -> str:
    """Collapse whitespace so a multi-line answer reads as one line in a panel."""
    out = " ".join((t or "").split())
    return out if limit is None or len(out) <= limit else out[:limit] + "…"


def _panel(title, body) -> RenderableType:
    return Panel(body, title=title, title_align="left",
                 border_style=render.B_ACCENT, padding=(1, 2))


def _overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / len(wa | wb) if (wa | wb) else 1.0


# ── /samplers ──────────────────────────────────────────────────────────────
async def samplers(client, caps, live=None, arg: str = "") -> RenderableType:
    extra: dict = {}
    zoo = getattr(caps, "sampler_zoo", None) or []
    if "dry" in zoo:
        extra.update(dry_multiplier=0.9, dry_base=1.75, dry_allowed_length=2)
    if "min_p" in zoo:
        extra["min_p"] = 0.1
    if "xtc" in zoo:
        extra.update(xtc_probability=0.5, xtc_threshold=0.1)
    body = Text()
    if not extra:
        body.append("no advanced samplers exposed on this endpoint", render.AMBER)
        return _panel("⚡ sampler zoo", body)
    on_token = live.token if live is not None else None
    prompt = arg.strip() or (
        "List 25 single words you might say to describe a stormy sea. Comma-separated.")
    base, _ = await _gen(client, prompt, seed=3, on_token=on_token)
    tuned, _ = await _gen(client, prompt, seed=3, extra=extra, on_token=on_token)
    ov = _overlap(base, tuned)
    body.append("applied: ", render.C_DIM)
    body.append(", ".join(sorted(extra)) + "\n", render.JADE)
    body.append("default:  ", render.C_DIM)
    body.append(base[:80] + "\n", render.CREAM)
    body.append("tuned:    ", render.C_DIM)
    body.append(tuned[:80] + "\n", render.C_ANSWER)
    body.append(f"\nword-set overlap {ov:.0%} ", render.GOLD)
    body.append("— same seed; lower overlap = the samplers steered the trajectory", render.C_DIM)
    return _panel("⚡ sampler zoo (DRY · min_p · XTC)", body)


# ── /grammar ───────────────────────────────────────────────────────────────
async def grammar(client, caps, live=None, arg: str = "") -> RenderableType:
    from ..skills.exec import build_pipeline, generate_with_skill
    from ..skills.ir import Grammar
    from ..skills.skill import Skill
    # 39 binds reliably across GBNF (llama.cpp) and guided (vLLM/LM Studio); a
    # 137-long rule does NOT enforce on the guided-grammar boxes (Step emits 196,
    # GLM 0). The point — exact-N is unrepresentable-if-wrong — holds at any N.
    n = 39
    if arg.strip().isdigit():   # /grammar 127 — override the count
        n = max(1, int(arg.strip()))
    body = Text()
    if getattr(caps, "grammar", None) is None:
        body.append("no grammar support here — falls back to validate-and-retry", render.AMBER)
        return _panel("⚡ grammar: exactly-N by construction", body)
    skill = Skill(name="exact_sevens",
                  grammar=Grammar.from_rules({"root": " ".join(['"7"'] * n)}, root="root"),
                  system_prompt="Output only the characters requested.",
                  sampling_overrides={"temperature": 0.7})
    mt = max(1024, n + 128)   # generous budget: reasoning models emit nothing until done thinking
    on_token = live.token if live is not None else None
    if on_token is not None:
        # stream the grammar-constrained generation so the 7s type out live, the
        # grammar forcing each token. Same grammar body the skill would use — but
        # retry on an invalid result (some servers enforce GBNF only flakily in
        # streaming mode; generate_with_skill retries, so we must too).
        plan = (await build_pipeline(skill, client)).resolve(caps)
        msgs = [Message(role="system", content=skill.system_prompt),
                Message(role="user", content=f"Output exactly {n} sevens.")]
        cnt, valid = 0, False
        for attempt in range(4):
            if attempt and live is not None:
                live.reset()   # clear the prior (invalid) attempt from the live area
            sampling = SamplingParams(temperature=0.7, max_tokens=mt, seed=1 + attempt,
                                      extra=dict(plan.body_params))
            text, _ = await _stream(client, msgs, sampling, on_token)
            cnt, valid = text.count("7"), skill.validate_output(text.strip())
            if valid:
                break
    else:
        res = await generate_with_skill(client, caps, skill, f"Output exactly {n} sevens.",
                                        max_tokens=mt, seed=1)
        cnt, valid = res.text.count("7"), res.valid
    ok = cnt == n and valid
    body.append(f"asked for exactly {n} sevens\n", render.GOLD)
    body.append("got: ", render.C_DIM)
    body.append(f"{cnt} sevens", render.C_ANSWER if ok else render.ROSE)
    body.append(f"   valid={valid}   grammar={caps.grammar}\n", render.C_DIM)
    body.append("\nexactly N by construction — the grammar makes a miscount unrepresentable",
                render.C_DIM)
    return _panel("⚡ grammar: exactly-N by construction", body)


# ── /antislop ──────────────────────────────────────────────────────────────
async def antislop(client, caps, live=None, arg: str = "") -> RenderableType:
    from ..logits.antislop import generate_antislop
    from ..logits.budget import apply_template
    body = Text()
    if not getattr(caps, "raw_completion", False) or \
            await apply_template(client, [Message(role="user", content="x")]) is None:
        body.append("needs raw-completion + a chat-template endpoint (llama.cpp).\n", render.AMBER)
        body.append("elsewhere it degrades to tree-backtracking (a fork per banned hit).", render.C_DIM)
        return _panel("⚡ anti-slop (banned phrase + KV-rewind)", body)
    banned = ["dog"]
    prompt = ("Complete this sentence with the single most obvious word: "
              "'The quick brown fox jumps over the lazy ___'. "
              "Reply with the full completed sentence and nothing else.")
    if arg.strip():   # `/antislop word1, word2 | prompt` — either side may be omitted
        left, sep, right = arg.partition("|")
        if sep:
            words = [w.strip() for w in left.split(",") if w.strip()]
            if words:
                banned = words
            if right.strip():
                prompt = right.strip()
        else:
            prompt = arg.strip()
    msgs = [Message(role="user", content=prompt)]
    base = await generate_antislop(client, msgs, [], max_tokens=64, seed=5,
                                   prefill="<think>\n\n</think>\n\n")
    clean = await generate_antislop(client, msgs, banned, max_tokens=64, seed=5,
                                    prefill="<think>\n\n</think>\n\n")
    hits = sum(base.text.lower().count(b) for b in banned)
    clean_hits = sum(clean.text.lower().count(b) for b in banned)
    ok = clean_hits == 0 and hits > 0
    body.append(f"banned: {banned}\n", render.GOLD)
    body.append("unconstrained: ", render.C_DIM)
    body.append(f"{base.text.strip()[:70]!r}  ({hits}× {banned[0]!r})\n",
                render.ROSE if hits else render.CREAM)
    body.append("anti-slop:     ", render.C_DIM)
    body.append(f"{clean.text.strip()[:70]!r}  ({clean_hits}× {banned[0]!r}, "
                f"{clean.rewinds} KV rewind(s))\n",
                render.C_ANSWER if ok else render.AMBER)
    body.append("\ndetect → rewind KV → ban first token → resample", render.C_DIM)
    return _panel("⚡ anti-slop (banned phrase + KV-rewind)", body)


# ── /overlay ───────────────────────────────────────────────────────────────
async def overlay(client, caps, live=None, arg: str = "", agent_factory=None) -> RenderableType:
    body = Text()
    if not getattr(caps, "logprobs", False):
        body.append("no logprobs on this endpoint — the overlay can't render "
                    "(honest degrade, not faked).", render.AMBER)
        return _panel("⚡ confidence overlay", body)
    flags, prompt_arg = _parse_arg(arg)
    use_tools = "tools" in flags and agent_factory is not None
    prompt = prompt_arg or "In one sentence, why is the sky blue?"
    on_token = live.token if live is not None else None
    if use_tools:
        # Agent runs with retrieval/tools; we overlay the confidence of the answer
        # it gave AFTER its tools returned. Only hand the agent a token callback when
        # there's a live view: a callback puts it in streaming mode, and streamed
        # responses don't carry per-token logprobs.
        cb = None
        if on_token is not None:
            def cb(kind, text):   # noqa: E306
                if kind in ("content", "reasoning"):
                    on_token(kind, text)
        _ans, lps = await _tool_answer(agent_factory, caps, prompt, 3, cb)
        toks = _clean_logprobs(lps)
    else:
        _, r = await _gen(client, prompt, seed=3, temperature=0.7, logprobs=True,
                          on_token=on_token)
        toks = _answer_logprobs(r)
    if not toks:
        body.append("endpoint returned no per-token logprobs for this generation.", render.AMBER)
        return _panel("⚡ confidence overlay", body)
    body.append(render.confidence_text(toks))
    weak = min(toks, key=lambda t: t.logprob)
    body.append("\n\nleast-confident token: ", render.C_DIM)
    body.append(f"{weak.token!r} ({weak.logprob:+.2f})", render.ROSE)
    body.append(" → flag for resample / review", render.C_DIM)
    if use_tools:
        body.append("\n\nanswered with tools — confidence is measured on the grounded answer",
                    render.C_DIM)
    return _panel("⚡ confidence-as-weight overlay", body)


# EOS / chat-control markers a server may include in the token stream — they carry
# no meaning for a reader, so they don't belong in the confidence overlay.
_SPECIAL_TOK = re.compile(
    r"^(<[|｜][^<>]*[|｜]>|</?s>|<\|?(?:endoftext|eos|end_of_turn|im_end)\|?>)$")


def _clean_logprobs(lps):
    """Answer-only per-token logprobs (chain-of-thought and special tokens stripped)."""
    from ..signals.metrics import answer_logprobs
    toks = answer_logprobs(lps or [])
    return [t for t in toks if not _SPECIAL_TOK.match(t.token.strip())]


def _answer_logprobs(resp):
    return _clean_logprobs(resp.logprobs)


# ── /consistency ───────────────────────────────────────────────────────────
def _money(a: str) -> str:
    m = re.findall(r"\$?\s*0?\.\d{1,2}\b|\$?\s*\d+(?:\.\d{1,2})?", a.replace(",", ""))
    if not m:
        return a.strip().lower()[:24]
    raw = m[-1].replace(" ", "").lstrip("$")
    try:
        return f"{float(raw):.2f}"
    except ValueError:
        return raw


def _generic(a: str) -> str:
    """Vote key for a user-supplied prompt: the final number if there is one,
    else the normalized answer text."""
    m = re.findall(r"\d[\d,]*(?:\.\d+)?", a.replace(",", ""))
    return m[-1] if m else " ".join(a.lower().split())[:60]


async def consistency(client, caps, live=None, arg: str = "",
                      agent_factory=None) -> RenderableType:
    flags, custom = _parse_arg(arg)
    use_tools = "tools" in flags and agent_factory is not None
    semantic = True if "semantic" in flags else (False if "lexical" in flags else None)
    msgs = [Message(role="user", content=custom or (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost? Answer with just the dollar amount, no explanation."))]
    key = _generic if custom else _money
    n = 5
    sampling = SamplingParams(temperature=0.8, max_tokens=_BUDGET, extra=_NO_THINK)
    t0 = time.monotonic()
    answer, agree, groups, sem = await _fan_consensus(
        client, caps, msgs, n, key, sampling, live, base_seed=200,
        title="⚡ self-consistency · 5 parallel rollouts",
        agent_factory=agent_factory if use_tools else None, semantic=semantic)
    dt = time.monotonic() - t0
    mode = f"{n} tool-enabled agent runs" if use_tools else (
        f"{n} concurrent rollouts" if live is not None else (
            "one parallel-n request" if getattr(caps, "parallel_n", False)
            else "5 prefix-cache forks"))
    body = Text()
    body.append(f"sampled {n} answers via {mode} in {dt:.1f}s\n", render.C_DIM)
    body.append("grouped by " + ("meaning (semantic entropy)" if sem else "exact answer") + "\n\n",
                render.C_DIM)
    body.append("consensus answer\n", render.C_DIM)
    body.append(f"{_flat(answer)}\n", render.C_ANSWER)
    body.append("\nagreement: ", render.C_DIM)
    style = render.JADE if agree >= 0.8 else render.AMBER
    body.append(f"{agree:.0%}", style)
    body.append(f"   ({len(groups)} distinct of {n})\n", render.C_DIM)
    if len(groups) > 1:   # the spread IS the uncertainty — show every variant
        body.append("\nspread\n", render.C_DIM)
        for _k, c, rep in groups:
            body.append(f"  {c}× ", render.GOLD)
            body.append(f"{_flat(rep, 90)}\n", render.CREAM)
    body.append("\nagreement is the uncertainty signal — low agreement routes the loop "
                "(resample / escalate / ask)", render.C_DIM)
    return _panel("⚡ self-consistency (consensus = confidence)", body)


# ── /escalate ──────────────────────────────────────────────────────────────
def _num(a: str) -> str:
    # the raw final number — a Fermi estimate genuinely diverges run-to-run, so
    # voting on the exact value (not its magnitude bucket) yields low agreement →
    # the loop escalates, which is the whole point of the demo. Also clearer in the
    # fan rows (shows '200', not '1e2').
    m = re.findall(r"\d[\d,]*", a.replace(",", ""))
    return m[-1] if m else a.strip().lower()[:16]


async def escalate(client, caps, live=None, arg: str = "",
                   agent_factory=None) -> RenderableType:
    from ..agent.permissions import Permissions
    flags, prompt_arg = _parse_arg(arg)
    use_tools = "tools" in flags and agent_factory is not None
    perms = Permissions(allow=["read_file"], ask=["write_file"], deny=["bash"])
    body = Text()
    body.append("permissions (deterministic, by tool name):  ", render.C_DIM)
    for tool in ("read_file", "write_file", "bash"):
        body.append(f"{tool}=", render.C_DIM)
        body.append(f"{perms.decide(tool)}  ", render.JADE)
    body.append("\n", render.C_DIM)
    msgs = [Message(role="user", content=prompt_arg or (
        "How many piano tuners work in Chicago? Reply with just a single number, no explanation."))]
    answer, agree, groups, _sem = await _fan_consensus(
        client, caps, msgs, 5, _num,
        SamplingParams(temperature=0.9, max_tokens=_BUDGET, extra=_NO_THINK),
        live, base_seed=300,
        title="⚡ agreement-routed escalation · 5 parallel rollouts",
        agent_factory=agent_factory if use_tools else None)
    nums = re.findall(r"\d[\d,]*", answer.replace(",", ""))
    shown = nums[-1] if nums else answer.strip()[:20]
    escalate_now = agree < 0.8
    body.append(f"\nestimate '{shown}' agreed ", render.C_DIM)
    body.append(f"{agree:.0%}", render.AMBER if escalate_now else render.JADE)
    if escalate_now:
        body.append(" < 80% → ", render.C_DIM)
        body.append("ESCALATE", render.ROSE)
        body.append(" (resample / ask / bigger model)\n", render.C_DIM)
    else:
        body.append(" ≥ 80% → proceed\n", render.C_DIM)
    if len(groups) > 1:
        body.append("\nspread\n", render.C_DIM)
        for _k, c, rep in groups:
            body.append(f"  {c}× ", render.GOLD)
            body.append(f"{_flat(rep, 90)}\n", render.CREAM)
    body.append("\nconfidence lives in verification (agreement), not the permission check",
                render.C_DIM)
    return _panel("⚡ agreement-routed escalation", body)


# ── /bestof ────────────────────────────────────────────────────────────────
async def bestof(client, caps, live=None, arg: str = "", agent_factory=None) -> RenderableType:
    from ..signals.metrics import StepSignals
    from ..tree.search.best_of_n import MeanLogprobVerifier, best_of_n
    n = 4
    flags, prompt_arg = _parse_arg(arg)
    use_tools = "tools" in flags and agent_factory is not None
    question = prompt_arg or "In one short sentence, why is the sky blue?"
    msgs = [Message(role="user", content=question)]
    # Only request logprobs in the stream if the server actually streams them. GLM
    # serves logprobs via the /responses API, not chat streaming — asking for them
    # mid-stream errors there, so we degrade to unranked candidates (still a fan-out).
    lp = bool(getattr(caps, "logprobs", False)) and bool(getattr(caps, "stream_logprobs", True))
    sampling = SamplingParams(temperature=0.9, max_tokens=_BUDGET, logprobs=lp,
                              top_logprobs=2 if lp else 4, extra=_NO_THINK)
    t0 = time.monotonic()
    if use_tools:
        # N independent tool-enabled agent runs, each ranked on the confidence of
        # its own final (post-tool) answer.
        if live is not None:
            live.fan([str(i + 1) for i in range(n)],
                     "⚡ best-of-N · 4 tool-enabled agent runs, verifier-ranked")

        async def one(i):
            cb = None
            if live is not None:
                def cb(kind, text, _i=i):   # noqa: E306
                    if kind in ("content", "reasoning"):
                        live.fan_token(_i, kind, text)
            try:
                ans, lp = await _tool_answer(agent_factory, caps, question, 100 + i, cb)
            except Exception:  # noqa: BLE001
                ans, lp = "", None
            sig = StepSignals.from_logprobs(lp or [])
            score = sig.mean_logprob if sig else float("-inf")
            if live is not None:
                live.fan_done(i, f"score={score:+.3f}" if score != float("-inf") else "n/a")
            return (ans, score)
        cands = [c for c in await asyncio.gather(*[one(i) for i in range(n)]) if c[0]]
        cands.sort(key=lambda c: c[1], reverse=True)
    elif live is not None:
        live.fan([str(i + 1) for i in range(n)],
                 "⚡ best-of-N · 4 candidates streaming, verifier-ranked")
        results = await _stream_samples(client, msgs, sampling, n, 100, live.fan_token)
        cands = []
        for i, (text, resp) in enumerate(results):
            sig = StepSignals.from_logprobs(resp.logprobs or []) if resp is not None else None
            score = sig.mean_logprob if sig else float("-inf")
            cands.append((text.strip(), score))
            live.fan_done(i, f"score={score:+.3f}" if score != float("-inf") else "n/a")
        cands.sort(key=lambda c: c[1], reverse=True)
    else:
        bn = await best_of_n(client, caps, msgs, MeanLogprobVerifier(), n=n, sampling=sampling)
        cands = [((c.text or "").strip(), c.score) for c in bn]
    dt = time.monotonic() - t0
    mode = f"{n} tool-enabled agent runs" if use_tools else (
        f"{n} concurrent rollouts" if live is not None else (
            "one parallel-n request" if getattr(caps, "parallel_n", False)
            else "sequential prefix-cache forks"))
    ranked = any(s != float("-inf") for _, s in cands)
    body = Text()
    body.append(f"sampled {n} candidates via {mode} in {dt:.1f}s\n", render.C_DIM)
    for text, score in cands:
        s = f"{score:+.3f}" if ranked else "  n/a"
        body.append(f"  score={s}  ", render.GOLD)
        body.append(f"{text[:58]!r}\n", render.CREAM)
    body.append("\nbest selected by the verifier — "
                + ("each run retrieved independently" if use_tools
                   else "forks reuse the shared prefix"), render.C_DIM)
    return _panel("⚡ best-of-N (verifier-ranked)", body)


# ── /thinkbudget ───────────────────────────────────────────────────────────
async def thinkbudget(client, caps, live=None, arg: str = "") -> RenderableType:
    from ..logits.budget import apply_template, generate_with_think_budget
    body = Text()
    if not getattr(caps, "raw_completion", False) or \
            await apply_template(client, [Message(role="user", content="x")]) is None:
        body.append("needs raw-completion + a chat-template endpoint (llama.cpp).", render.AMBER)
        return _panel("⚡ think-budget forcing (s1-style)", body)
    msgs = [Message(role="user", content=arg.strip() or (
        "How many r's are in 'strawberry'? Think, then answer."))]
    r = await generate_with_think_budget(client, msgs, think_budget=64, seed=2)
    body.append("budget = 64 reasoning tokens\n", render.GOLD)
    body.append("reasoning used: ", render.C_DIM)
    body.append(f"{r.reasoning_tokens} tokens", render.JADE)
    body.append(f"   forced_close={r.forced_close}\n", render.C_DIM)
    body.append("answer: ", render.C_DIM)
    body.append(f"{r.answer[:80]!r}\n", render.C_ANSWER)
    body.append("\nreasoning length capped at the token level (bias </think>, s1 'Wait' continuation)",
                render.C_DIM)
    return _panel("⚡ think-budget forcing (s1-style)", body)


# Advantages whose samples can run through the full agent loop (tools + retrieval)
# when invoked with `--tools`. The others are decoding-mechanics demos — grammar,
# samplers, token bans, reasoning budget — where a mid-generation tool call would
# change the very thing being measured.
TOOL_CAPABLE = frozenset({"overlay", "consistency", "escalate", "bestof"})

ADVANTAGES = {
    "grammar": grammar,
    "samplers": samplers,
    "antislop": antislop,
    "overlay": overlay,
    "consistency": consistency,
    "escalate": escalate,
    "bestof": bestof,
    "thinkbudget": thinkbudget,
}
