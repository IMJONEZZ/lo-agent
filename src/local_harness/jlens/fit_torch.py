"""Exact causal Jacobian lens on an HF/safetensors checkpoint (Rung 6, vLLM path).

llama.cpp has no autograd, so the GGUF path fits a regression surrogate. On a
real torch checkpoint we can fit the paper's exact estimator
``J_l = E[∂h_final/∂h_l]`` via backprop, then serialize to the SAME lens-GGUF
format so it drops straight into the lens service. Needs the ``native`` extra
(torch + transformers). Safetensors is the better-supported fitting format.

This is the vLLM/safetensors reach: fit here, then either steer the vLLM box
with a LoRA/abliterated export, or run a CPU mirror-forward for inspection.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SKIP_FIRST = 16


def _decoder_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError("could not locate decoder layers on this model")


def fit_causal_lens(model, tokenizer, prompts, *, source_layers=None, target_layer=None,
                    dim_batch: int = 8, max_seq_len: int = 128, skip_first: int = SKIP_FIRST,
                    device: str = "cpu"):
    """Fit the exact causal lens. Returns a JacobianLensGGUF (fit_method='jacobian').

    Estimator (Anthropic reference): for each output dim, inject a one-hot
    cotangent at every valid target position and backprop; the gradient at
    source position p sums ∂h_final[p']/∂h_l[p] over p'≥p; average over p and
    prompts. One forward + ceil(d/dim_batch) backward passes per prompt.
    """
    import torch

    from local_harness.jlens.lens import JacobianLensGGUF

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layers = _decoder_layers(model)
    n_layers = len(layers)
    d = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    target = (n_layers - 1) if target_layer is None else target_layer % n_layers
    src = list(source_layers) if source_layers is not None else list(range(target))
    src = sorted({l % n_layers for l in src if l < target})

    # capture residual outputs via hooks; start_graph so backprop reaches sources
    captured: dict[int, "torch.Tensor"] = {}

    def _hook(idx):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if idx in src:
                t.requires_grad_(True)
                t.retain_grad()
            captured[idx] = t
            return out
        return fn

    handles = [layers[i].register_forward_hook(_hook(i)) for i in range(n_layers)]
    jac_sum = {l: np.zeros((d, d), np.float64) for l in src}
    hrms = {l: 0.0 for l in src}
    n_done = 0
    try:
        for text in prompts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_seq_len)["input_ids"].to(device) \
                if not isinstance(tokenizer, _CallableTok) else tokenizer(text, max_seq_len, device)
            T = ids.shape[1]
            if T <= skip_first + 1:
                continue
            captured.clear()
            model(ids)  # populate hooks, build graph
            h_final = captured[target]                      # [1, T, d]
            valid = slice(skip_first, T - 1)
            for l in src:
                hrms[l] += float(captured[l][0, valid].pow(2).mean().sqrt().item() * (d ** 0.5))
            # backprop dim by dim
            for l in src:
                Jl = np.zeros((d, d), np.float64)
                for start in range(0, d, dim_batch):
                    end = min(start + dim_batch, d)
                    for dim in range(start, end):
                        model.zero_grad(set_to_none=True)
                        if captured[l].grad is not None:
                            captured[l].grad = None
                        cot = torch.zeros_like(h_final)
                        cot[0, valid, dim] = 1.0
                        h_final.backward(cot, retain_graph=True)
                        g = captured[l].grad[0, valid].double().mean(0).cpu().numpy()
                        Jl[dim] = g
                jac_sum[l] += Jl
            n_done += 1
            logger.info("fit_causal prompt %d/%d (T=%d)", n_done, len(prompts), T)
    finally:
        for h in handles:
            h.remove()

    if n_done == 0:
        raise ValueError("no prompts long enough to fit")
    jacobians = {l: (jac_sum[l] / n_done).astype(np.float32) for l in src}
    h_rms = {l: hrms[l] / n_done for l in src}
    return JacobianLensGGUF(jacobians, d_model=d, n_prompts=n_done, target_layer=target,
                            fit_method="jacobian", h_rms=h_rms)


class _CallableTok:
    """Marker for a plain callable tokenizer (text, max_len, device) -> ids."""
