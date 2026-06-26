"""Tier 4: in-process Transformers backend with a custom sampling loop.

What this unlocks that no HTTP server can offer:
- arbitrary stateful logit processors (the pipeline's `process()` path)
- exact anti-slop: rewind the KV cache itself (DynamicCache.crop)
- second-pass guidance (CFG / contrastive — see logits/guidance.py)
- activation hooks for steering (native/steering.py)

The custom loop is intentionally simple (temperature sampling + processors);
it is a research substrate, not a serving engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from ..inference.capabilities import Capabilities


@dataclass
class NativeResult:
    text: str
    token_ids: list[int]
    logprobs: list[float]
    rewinds: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class NativeBackend:
    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu"):
        import torch  # deferred: native extra

        self.torch = torch
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = "cpu") -> "NativeBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        return cls(
            AutoModelForCausalLM.from_pretrained(model_id),
            AutoTokenizer.from_pretrained(model_id),
            device=device,
        )

    def capabilities(self) -> Capabilities:
        return Capabilities(
            server="native", model=getattr(self.model.config, "name_or_path", "in-process"),
            seed=True, logprobs=True, grammar=None, logit_bias=True,
            sampler_zoo={"min_p", "top_k"}, raw_completion=True,
            cfg_scale=True, banned_strings=True, in_process=True,
        )

    # --- generation ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 1.0,
        seed: int = 0,
        processors: list[Any] | None = None,        # objects with .process(ids, scores)
        banned_phrases: list[str] | None = None,    # exact anti-slop with KV rewind
        max_rewinds: int = 20,
        stop: list[str] | None = None,
        logits_hook: Callable[[Any, Any], Any] | None = None,  # (ids, scores) -> scores
    ) -> NativeResult:
        torch = self.torch
        from transformers import DynamicCache

        generator = torch.Generator(device="cpu").manual_seed(seed)
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], device=self.device)
        cache = DynamicCache()
        eos = getattr(self.tokenizer, "eos_token_id", None)

        out_ids: list[int] = []
        logprobs: list[float] = []
        rewinds = 0
        positional_bans: dict[int, set[int]] = {}  # position -> banned token ids
        next_input = input_ids

        with torch.no_grad():
            while len(out_ids) < max_tokens:
                output = self.model(input_ids=next_input, past_key_values=cache, use_cache=True)
                scores = output.logits[:, -1, :].float()
                full_ids = torch.tensor([ids + out_ids], device=self.device)

                for proc in processors or []:
                    scores = proc.process(full_ids, scores)
                if logits_hook is not None:
                    scores = logits_hook(full_ids, scores)
                for tok in positional_bans.get(len(out_ids), ()):  # anti-slop rewind bans
                    scores[..., tok] = -float("inf")

                if temperature <= 0:
                    token = int(scores.argmax(dim=-1))
                else:
                    probs = torch.softmax(scores / temperature, dim=-1)
                    token = int(torch.multinomial(probs, 1, generator=generator))
                logprob = float(torch.log_softmax(scores, dim=-1)[0, token])

                out_ids.append(token)
                logprobs.append(logprob)
                next_input = torch.tensor([[token]], device=self.device)

                text = self.tokenizer.decode(out_ids)
                if banned_phrases:
                    hit = self._find_phrase(text, banned_phrases)
                    if hit is not None and rewinds < max_rewinds:
                        # Rewind the KV cache to just before the token where the
                        # phrase began and ban *that token at that position* —
                        # exact positional banning, impossible over HTTP.
                        start_tok = self._token_index_for_char(out_ids, hit)
                        positional_bans.setdefault(start_tok, set()).add(out_ids[start_tok])
                        out_ids = out_ids[:start_tok]
                        logprobs = logprobs[:start_tok]
                        # To sample position start_tok we need logits conditioned
                        # on prefix = ids + out_ids: crop the cache to everything
                        # before the prefix's last token, then re-feed that token.
                        prefix = ids + out_ids
                        cache.crop(len(prefix) - 1)
                        next_input = torch.tensor([[prefix[-1]]], device=self.device)
                        rewinds += 1
                        continue

                if eos is not None and token == eos:
                    out_ids.pop()
                    logprobs.pop()
                    break
                if stop and any(s in text for s in stop):
                    break

        return NativeResult(
            text=self.tokenizer.decode(out_ids), token_ids=out_ids,
            logprobs=logprobs, rewinds=rewinds,
        )

    def _find_phrase(self, text: str, phrases: list[str]) -> int | None:
        lower = text.lower()
        hits = [lower.find(p.lower()) for p in phrases]
        hits = [h for h in hits if h != -1]
        return min(hits) if hits else None

    def _token_index_for_char(self, out_ids: list[int], char_pos: int) -> int:
        """First generated-token index whose decoded span reaches char_pos."""
        for i in range(len(out_ids)):
            if len(self.tokenizer.decode(out_ids[: i + 1])) > char_pos:
                return i
        return max(0, len(out_ids) - 1)
