"""Anti-slop: phrase-level bans with backtracking — the canonical example of
a control impossible with static logit_bias.

Emulation over HTTP (llama.cpp: raw completion + /apply-template + /tokenize):
generate, scan for banned phrases; on a hit, rewind the text to just before
the phrase, ban the phrase's first token via logit_bias, and regenerate from
the rewind point (a prefix-cache hit, so rewinds are cheap).

Approximation note: the first-token ban is global for the remainder of the
generation, not positional. Exact positional banning arrives with the native
backend (Phase 5). Servers with native banned_strings (KoboldCpp/TabbyAPI)
should use that instead — check `caps.banned_strings`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..inference.client import OpenAICompatClient
from ..inference.types import Message
from .budget import apply_template


@dataclass
class AntislopResult:
    text: str
    rewinds: int
    banned_token_ids: dict[int, str] = field(default_factory=dict)  # id -> phrase


async def _first_token_ids(client: OpenAICompatClient, phrase: str) -> list[int]:
    """Token ids that can begin this phrase (bare and space-prefixed forms)."""
    ids = []
    for variant in (phrase, " " + phrase):
        resp = await client.post("/tokenize", json={"content": variant, "add_special": False})
        if resp.status_code != 200:
            continue
        tokens = resp.json().get("tokens", [])
        if tokens:
            tok = tokens[0] if isinstance(tokens[0], int) else tokens[0].get("id")
            ids.append(tok)
    return ids


def _find_banned(text: str, phrases: list[str], from_pos: int) -> tuple[int, str] | None:
    """Earliest banned-phrase occurrence at or after from_pos (case-insensitive)."""
    lower = text.lower()
    best: tuple[int, str] | None = None
    for phrase in phrases:
        i = lower.find(phrase.lower(), max(0, from_pos))
        if i != -1 and (best is None or i < best[0]):
            best = (i, phrase)
    return best


async def generate_antislop(
    client: OpenAICompatClient,
    messages: list[Message],
    banned_phrases: list[str],
    max_tokens: int = 256,
    seed: int | None = None,
    max_rewinds: int = 10,
    sampling_extra: dict[str, Any] | None = None,
    prefill: str = "",
    slop_detector=None,
) -> AntislopResult:
    """`prefill` is appended to the rendered prompt before generation — e.g.
    "<think>\\n\\n</think>\\n\\n" skips a reasoning model's think block so the
    scan applies to the visible answer (raw completion bypasses the server's
    reasoning parser).

    `slop_detector` (a logits.slop.SlopDetector) additionally rewinds on overused
    rhetorical *structures*. A fixed phrase is removed by banning its first token;
    a structure is a pattern, not a token, so we instead rewind to it and bump the
    seed to force a different continuation."""
    prompt = await apply_template(client, messages)
    if prompt is None:
        raise RuntimeError("anti-slop emulation needs /apply-template (llama.cpp)")
    prompt += prefill

    extra: dict[str, Any] = {"temperature": 0.7, "cache_prompt": True, **(sampling_extra or {})}

    text = ""
    rewinds = 0
    cur_seed = seed
    banned_ids: dict[int, str] = {}
    max_phrase = max((len(p) for p in banned_phrases), default=0)

    while True:
        body = {**extra, "max_tokens": max_tokens}
        if cur_seed is not None:
            body["seed"] = cur_seed
        if banned_ids:
            body["logit_bias"] = {str(t): -100 for t in banned_ids}
        out = await client.complete_raw(prompt + text, body)
        new = out["choices"][0].get("text", "")
        combined = text + new

        # earliest offender: a banned phrase or a slop structure, whichever comes first
        phrase_hit = _find_banned(combined, banned_phrases, len(text) - max_phrase)
        struct = slop_detector.first_match(combined) if slop_detector is not None else None
        offenders: list[tuple[int, str, str]] = []  # (pos, kind, what)
        if phrase_hit is not None:
            offenders.append((phrase_hit[0], "phrase", phrase_hit[1]))
        if struct is not None:
            offenders.append((struct.start, "structure", struct.name))
        if not offenders:
            return AntislopResult(text=combined, rewinds=rewinds, banned_token_ids=banned_ids)

        pos, kind, what = min(offenders, key=lambda o: o[0])
        rewinds += 1
        if kind == "phrase":
            for tok in await _first_token_ids(client, what):
                banned_ids[tok] = what
        else:  # structure: can't ban a pattern token — diverge via a seed bump
            cur_seed = (cur_seed or 0) + 1
        text = combined[:pos].rstrip() if pos else ""
        if rewinds >= max_rewinds:
            return AntislopResult(text=text, rewinds=rewinds, banned_token_ids=banned_ids)
