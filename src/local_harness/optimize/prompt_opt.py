"""Prompt optimization that produces a tuned-replay intervention.

The goal: take a base instruction + a metric + a few examples, search for a better
instruction against the LOCAL endpoint (unlimited free trials), and hand the winner
back as an `Intervention(system_prompt=...)` so `replay_tuned` can re-synthesize a
logged conversation under the optimized prompt.

Two native methods (no extra deps, run on the local server):
  - "mipro": model-proposes instruction variants, scored on a valset (reuses
    optimize.bootstrap.instruction_search — a mini-MIPRO).
  - "gepa":  GEPA-style *reflective* evolution — propose variants by reflecting on
    the current best's failures (its low-scoring outputs + the metric feedback),
    keep the Pareto-best. Reflective mutation is GEPA's distinguishing idea.

And a DSPy bridge ("dspy-gepa" / "dspy-mipro") for the full optimizers, gated on the
`dspy` extra. Local inference is what makes any of this free — a frontier API meters
every trial.
"""

from __future__ import annotations

from dataclasses import replace

from ..inference.capabilities import Capabilities
from ..inference.client import OpenAICompatClient
from ..inference.types import GenerationRequest, Message, SamplingParams
from .bootstrap import Example, FewShotProgram, Metric, evaluate, instruction_search


async def _failures(client, program: FewShotProgram, valset: list[Example],
                    metric: Metric, seed: int, limit: int = 3) -> str:
    """Run the program and describe its lowest-scoring cases — the reflection fuel."""
    scored = []
    for i, ex in enumerate(valset):
        out = await program.run(client, ex.input, seed=seed + i)
        scored.append((metric(out, ex), ex, out))
    scored.sort(key=lambda s: s[0])
    lines = []
    for score, ex, out in scored[:limit]:
        lines.append(f"- input: {ex.input[:120]}\n  output (score {score:.2f}): {out[:160]}"
                     + (f"\n  expected: {ex.expected[:120]}" if ex.expected else ""))
    return "\n".join(lines)


async def _reflective_propose(client, instruction: str, feedback: str, seed: int) -> str:
    r = await client.chat(GenerationRequest(
        messages=[Message(role="user", content=(
            "You are improving a task instruction for a language model. Here is the "
            "current instruction and how it performed on its hardest cases. Reflect on "
            "what went wrong and write a single improved instruction that would fix it. "
            "Reply with ONLY the new instruction.\n\n"
            f"Current instruction:\n{instruction}\n\nHardest cases:\n{feedback}"))],
        sampling=SamplingParams(temperature=1.0, max_tokens=256, seed=seed)))
    return r.text.strip()


async def reflective_search(
    client: OpenAICompatClient, caps: Capabilities, program: FewShotProgram,
    valset: list[Example], metric: Metric, rounds: int = 2,
    candidates_per_round: int = 3, seed: int = 1,
) -> tuple[FewShotProgram, float]:
    """GEPA-style reflective prompt evolution."""
    best = program
    best_score = await evaluate(client, program, valset, metric, seed=seed)
    for r in range(rounds):
        feedback = await _failures(client, best, valset, metric, seed)
        for i in range(candidates_per_round):
            inst = await _reflective_propose(client, best.instruction, feedback,
                                             seed + 6151 * (r * candidates_per_round + i + 1))
            if not inst:
                continue
            variant = replace(best, instruction=inst)
            score = await evaluate(client, variant, valset, metric, seed=seed)
            if score > best_score:
                best, best_score = variant, score
    return best, best_score


async def _dspy_optimize(method, client, base_instruction, valset, metric):  # pragma: no cover
    """Full DSPy GEPA / MIPROv2 against the local endpoint (needs `--extra dspy`)."""
    from .dspy_opt import configure_dspy
    import dspy

    configure_dspy(client.base_url, client.model)

    class _Sig(dspy.Signature):
        __doc__ = base_instruction
        input = dspy.InputField()
        output = dspy.OutputField()

    program = dspy.Predict(_Sig)
    trainset = [dspy.Example(input=e.input, output=e.expected).with_inputs("input")
                for e in valset]

    def _m(example, pred, trace=None):
        return metric(getattr(pred, "output", ""), Example(input=example.input,
                                                           expected=example.expected))
    Optimizer = dspy.GEPA if method == "dspy-gepa" else dspy.MIPROv2
    compiled = Optimizer(metric=_m).compile(program, trainset=trainset)
    inst = getattr(compiled.signature, "instructions", base_instruction) or base_instruction
    return inst, 1.0


async def optimize_instruction(
    client: OpenAICompatClient, caps: Capabilities, base_instruction: str,
    valset: list[Example], metric: Metric, method: str = "gepa",
    seed: int = 1, **kw,
) -> tuple[str, float]:
    """Search for a better instruction; return (best_instruction, val_score)."""
    program = FewShotProgram(instruction=base_instruction)
    if method == "gepa":
        best, score = await reflective_search(client, caps, program, valset, metric, seed=seed, **kw)
        return best.instruction, score
    if method == "mipro":
        best, score = await instruction_search(client, caps, program, valset, metric, seed=seed, **kw)
        return best.instruction, score
    if method in ("dspy-gepa", "dspy-mipro"):
        return await _dspy_optimize(method, client, base_instruction, valset, metric)
    raise ValueError(f"unknown method {method!r} (use gepa | mipro | dspy-gepa | dspy-mipro)")


def instruction_intervention(instruction: str, method: str = "gepa"):
    """Wrap an optimized instruction as a tuned-replay Intervention."""
    from ..tuned_replay import Intervention
    return Intervention(label=f"prompt-opt:{method}", system_prompt=instruction)
