"""Prompt optimization against the local endpoint — unlimited trials, no rate
limits. Native implementations of the two workhorse algorithms:

- bootstrap_few_shot (DSPy BootstrapFewShot): run the program over a trainset,
  keep traces the metric accepts, use the best as in-context demos.
- instruction_search (mini-MIPRO): have the model itself propose instruction
  variants, evaluate each on a valset, keep the winner.

Optimized programs serialize to JSON beside the skill (<skill>.optimized.json)
and are applied automatically when the skill runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from ..inference.capabilities import Capabilities
from ..inference.client import OpenAICompatClient
from ..inference.types import GenerationRequest, Message, SamplingParams

Metric = Callable[[str, "Example"], float]  # (output, example) -> score in [0,1]


@dataclass
class Example:
    input: str
    expected: str = ""


@dataclass
class FewShotProgram:
    instruction: str
    demos: list[Example] = field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 256
    extra: dict = field(default_factory=dict)  # e.g. chat_template_kwargs

    def messages(self, input: str) -> list[Message]:
        msgs = [Message(role="system", content=self.instruction)]
        for d in self.demos:
            msgs.append(Message(role="user", content=d.input))
            msgs.append(Message(role="assistant", content=d.expected))
        msgs.append(Message(role="user", content=input))
        return msgs

    async def run(self, client: OpenAICompatClient, input: str, seed: int = 1) -> str:
        response = await client.chat(GenerationRequest(
            messages=self.messages(input),
            sampling=SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens,
                                     seed=seed, extra=dict(self.extra)),
        ))
        return response.text.strip()

    def to_dict(self) -> dict:
        return {"instruction": self.instruction,
                "demos": [{"input": d.input, "expected": d.expected} for d in self.demos]}

    @classmethod
    def from_dict(cls, d: dict) -> "FewShotProgram":
        return cls(instruction=d["instruction"],
                   demos=[Example(**e) for e in d.get("demos", [])])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FewShotProgram":
        return cls.from_dict(json.loads(Path(path).read_text()))


async def evaluate(
    client: OpenAICompatClient,
    program: FewShotProgram,
    dataset: list[Example],
    metric: Metric,
    seed: int = 1,
) -> float:
    if not dataset:
        return 0.0
    total = 0.0
    for i, ex in enumerate(dataset):
        output = await program.run(client, ex.input, seed=seed + i)
        total += metric(output, ex)
    return total / len(dataset)


async def bootstrap_few_shot(
    client: OpenAICompatClient,
    program: FewShotProgram,
    trainset: list[Example],
    metric: Metric,
    max_demos: int = 4,
    threshold: float = 1.0,
    seed: int = 1,
) -> FewShotProgram:
    """Self-generate demos: keep (input, model_output) pairs the metric accepts."""
    demos: list[Example] = []
    for i, ex in enumerate(trainset):
        if len(demos) >= max_demos:
            break
        output = await program.run(client, ex.input, seed=seed + i)
        if metric(output, ex) >= threshold:
            demos.append(Example(input=ex.input, expected=output))
    return FewShotProgram(instruction=program.instruction, demos=demos,
                          temperature=program.temperature, max_tokens=program.max_tokens,
                          extra=dict(program.extra))


async def instruction_search(
    client: OpenAICompatClient,
    caps: Capabilities,
    program: FewShotProgram,
    valset: list[Example],
    metric: Metric,
    num_candidates: int = 4,
    seed: int = 1,
) -> tuple[FewShotProgram, float]:
    """Model-proposed instruction variants, picked by validation score."""
    candidates = [program.instruction]
    propose = GenerationRequest(
        messages=[Message(role="user", content=(
            "Rewrite this task instruction to be clearer and more effective for a "
            "language model. Reply with only the rewritten instruction.\n\n"
            f"Instruction: {program.instruction}"
        ))],
        sampling=SamplingParams(temperature=1.0, max_tokens=program.max_tokens,
                                 extra=dict(program.extra)),
    )
    for i in range(num_candidates - 1):
        propose.sampling.seed = seed + 7919 * (i + 1)
        response = await client.chat(propose)
        text = response.text.strip()
        if text:
            candidates.append(text)

    best, best_score = program, -1.0
    for inst in candidates:
        variant = FewShotProgram(instruction=inst, demos=program.demos,
                                 temperature=program.temperature,
                                 max_tokens=program.max_tokens, extra=dict(program.extra))
        score = await evaluate(client, variant, valset, metric, seed=seed)
        if score > best_score:
            best, best_score = variant, score
    return best, best_score
