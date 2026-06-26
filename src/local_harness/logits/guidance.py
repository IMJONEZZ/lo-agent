"""Second-pass guidance: logit corrections that need extra forward passes.

- Classifier-free guidance (Sanchez et al. 2023): logits_cond and
  logits_uncond from two contexts; sample from
  uncond + scale * (cond - uncond). scale > 1 strengthens instruction
  adherence; the unconditioned context can be a *negative prompt*.
- Contrastive decoding (Li et al. 2022): expert - amateur logit difference,
  with an expert-probability plausibility floor.

Both run only in-process (Tier 4): they need two synchronized forward passes
per token.
"""

from __future__ import annotations

from ..native.backend import NativeBackend, NativeResult


def cfg_generate(
    backend: NativeBackend,
    prompt: str,
    negative_prompt: str = "",
    scale: float = 1.5,
    max_tokens: int = 64,
    temperature: float = 0.7,
    seed: int = 0,
) -> NativeResult:
    torch = backend.torch
    cond = backend.tokenizer.encode(prompt)
    uncond = backend.tokenizer.encode(negative_prompt) if negative_prompt else cond[:1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    eos = getattr(backend.tokenizer, "eos_token_id", None)
    out: list[int] = []
    logprobs: list[float] = []

    with torch.no_grad():
        while len(out) < max_tokens:
            ids_c = torch.tensor([cond + out], device=backend.device)
            ids_u = torch.tensor([uncond + out], device=backend.device)
            logits_c = backend.model(input_ids=ids_c).logits[:, -1, :].float()
            logits_u = backend.model(input_ids=ids_u).logits[:, -1, :].float()
            scores = logits_u + scale * (logits_c - logits_u)

            if temperature <= 0:
                token = int(scores.argmax(dim=-1))
            else:
                probs = torch.softmax(scores / temperature, dim=-1)
                token = int(torch.multinomial(probs, 1, generator=generator))
            if eos is not None and token == eos:
                break
            out.append(token)
            logprobs.append(float(torch.log_softmax(scores, dim=-1)[0, token]))

    return NativeResult(text=backend.tokenizer.decode(out), token_ids=out, logprobs=logprobs,
                        meta={"cfg_scale": scale})


def contrastive_generate(
    expert: NativeBackend,
    amateur: NativeBackend,
    prompt: str,
    alpha: float = 0.1,
    max_tokens: int = 64,
    seed: int = 0,
) -> NativeResult:
    """Greedy contrastive decoding: argmax(log p_expert - log p_amateur) over
    tokens whose expert probability >= alpha * max expert probability."""
    torch = expert.torch
    ids = expert.tokenizer.encode(prompt)
    eos = getattr(expert.tokenizer, "eos_token_id", None)
    out: list[int] = []
    logprobs: list[float] = []

    with torch.no_grad():
        while len(out) < max_tokens:
            full = torch.tensor([ids + out], device=expert.device)
            lp_e = torch.log_softmax(expert.model(input_ids=full).logits[:, -1, :].float(), dim=-1)
            lp_a = torch.log_softmax(amateur.model(input_ids=full).logits[:, -1, :].float(), dim=-1)
            # plausibility constraint
            floor = lp_e.max() + torch.log(torch.tensor(alpha))
            diff = (lp_e - lp_a).masked_fill(lp_e < floor, -float("inf"))
            token = int(diff.argmax(dim=-1))
            if eos is not None and token == eos:
                break
            out.append(token)
            logprobs.append(float(lp_e[0, token]))

    return NativeResult(text=expert.tokenizer.decode(out), token_ids=out, logprobs=logprobs,
                        meta={"alpha": alpha})
