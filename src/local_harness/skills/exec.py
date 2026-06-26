"""Skill execution: build the logit pipeline, resolve it against capabilities,
generate, validate, and retry on Tier-0 (or on server-constraint failure).

Output is always validated client-side — even when the server enforced the
grammar — because validity is the success criterion, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..inference.capabilities import Capabilities
from ..inference.client import OpenAICompatClient
from ..inference.types import GenerationRequest, Message, SamplingParams, TokenLogprob
from ..logits.bias import BiasProfileStore, BiasStage
from ..logits.grammar_stage import GrammarStage
from ..logits.pipeline import LogitPipeline, ResolvedPlan
from ..logits.samplers import SamplerChain
from ..signals.metrics import StepSignals
from .skill import Skill


@dataclass
class SkillResult:
    text: str
    valid: bool
    attempts: int
    plan: ResolvedPlan
    signals: StepSignals | None = None
    logprobs: list[TokenLogprob] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


async def build_pipeline(
    skill: Skill,
    client: OpenAICompatClient | None = None,
    profiles: BiasProfileStore | None = None,
) -> LogitPipeline:
    pipeline = LogitPipeline([GrammarStage(skill)])
    if skill.samplers:
        pipeline.add(SamplerChain(skill.samplers))
    if skill.bias_profile and profiles is not None:
        stage = BiasStage(profiles.get(skill.bias_profile))
        if client is not None:
            await stage.resolve_tokens(client)
        pipeline.add(stage)
    return pipeline


async def generate_with_skill(
    client: OpenAICompatClient,
    caps: Capabilities,
    skill: Skill,
    user_prompt: str,
    system_prompt: str | None = None,
    max_attempts: int = 4,
    seed: int = 1,
    max_tokens: int | None = None,
    profiles: BiasProfileStore | None = None,
) -> SkillResult:
    pipeline = await build_pipeline(skill, client, profiles)
    plan = pipeline.resolve(caps)

    extra = dict(plan.body_params)
    # Hot-swap a LoRA adapter in for this skill (skills-as-adapters). The base model
    # gains a specialized skill with no extra model to ship — and a frontier API
    # can't do this at all (closed weights).
    if skill.adapter and getattr(caps, "lora_mode", None):
        from ..inference.lora import ensure_adapter, request_overrides
        await ensure_adapter(client, caps, skill.adapter)
        extra.update(request_overrides(caps, skill.adapter))

    sampling = SamplingParams(
        temperature=skill.sampling_overrides.get("temperature", 0.7),
        top_p=skill.sampling_overrides.get("top_p"),
        max_tokens=max_tokens or skill.sampling_overrides.get("max_tokens", 512),
        logprobs=caps.logprobs,
        extra=extra,
    )
    system = system_prompt or skill.system_prompt
    messages = [Message(role="system", content=system)] if system else []
    # Optimizer-bootstrapped demos (skills/<name>.optimized.json) become
    # in-context examples; the optimized instruction overrides the default.
    optimized = skill.metadata.get("optimized")
    if optimized:
        if optimized.get("instruction") and not system_prompt:
            messages = [Message(role="system", content=optimized["instruction"])]
        for demo in optimized.get("demos", []):
            messages.append(Message(role="user", content=demo["input"]))
            messages.append(Message(role="assistant", content=demo["expected"]))
    messages.append(Message(role="user", content=user_prompt))

    last: SkillResult | None = None
    for attempt in range(1, max_attempts + 1):
        sampling.seed = seed + attempt - 1
        response = await client.chat(GenerationRequest(messages=messages, sampling=sampling))
        text = response.text.strip()
        valid = skill.validate_output(text)
        last = SkillResult(
            text=text,
            valid=valid,
            attempts=attempt,
            plan=plan,
            signals=StepSignals.from_logprobs(response.logprobs or []),
            logprobs=response.logprobs,
            raw=response.raw,
        )
        if valid:
            return last
    assert last is not None
    return last
