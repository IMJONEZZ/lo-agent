"""Weight-level optimization: fine-tune a LoRA adapter for a skill.

Completes the optimization stack: prompts (bootstrap.py) -> weights (here).
Adapters save to a directory the AdapterManager can hot-swap from.
"""

from __future__ import annotations

from pathlib import Path

from .bootstrap import Example


def finetune_lora(
    backend,
    examples: list[Example],
    output_dir: str | Path,
    adapter_name: str = "skill",
    epochs: int = 3,
    lr: float = 1e-3,
    rank: int = 8,
    target_modules: list[str] | None = None,
) -> list[float]:
    """Train a LoRA on (input -> expected) pairs with plain LM loss.
    Returns per-epoch mean losses (callers assert they decrease)."""
    import torch
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank, lora_alpha=2 * rank, target_modules=target_modules, task_type="CAUSAL_LM"
    )
    # Name the adapter so it can be hot-swapped by name later (AdapterManager /
    # skills-as-adapters). Without this, PEFT names it "default".
    model = get_peft_model(backend.model, config, adapter_name=adapter_name)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    encoded = []
    for ex in examples:
        ids = backend.tokenizer.encode(ex.input + ex.expected)
        encoded.append(torch.tensor([ids], device=backend.device))

    losses: list[float] = []
    for _ in range(epochs):
        total = 0.0
        for ids in encoded:
            optimizer.zero_grad()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            optimizer.step()
            total += float(out.loss)
        losses.append(total / len(encoded))

    model.eval()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    # leave the backend holding the adapted model (hot-swappable via AdapterManager)
    backend.model = model
    return losses
