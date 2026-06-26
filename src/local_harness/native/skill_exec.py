"""Run a grammar skill on the in-process native backend, hot-swapping the skill's
LoRA adapter in for the call — the on-device / edge skills path.

This bridges the *same* Skill object used over HTTP (grammar + prompt + adapter)
onto NativeBackend + AdapterManager, so "skills-as-adapters" works with no server:
a small base model in-process, a library of MB-sized adapters, swapped per skill.
Grammar/samplers run as native logit processors where supported; otherwise the
skill's grammar is enforced by validate-and-retry.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

from ..skills.exec import build_pipeline
from ..skills.skill import Skill


@dataclass
class NativeSkillResult:
    text: str
    valid: bool
    attempts: int
    adapter: str | None = None
    rewinds: int = 0


class _SafeProcessor:
    """Wrap a logit stage so HTTP-only stages that can't run native are skipped
    (their .process raises) rather than breaking generation."""

    def __init__(self, stage):
        self.stage = stage

    def process(self, input_ids, scores):
        try:
            return self.stage.process(input_ids, scores)
        except Exception:
            return scores


async def generate_with_skill_native(
    backend,
    skill: Skill,
    prompt: str,
    *,
    adapters=None,
    max_attempts: int = 4,
    seed: int = 1,
    max_tokens: int = 256,
    temperature: float | None = None,
) -> NativeSkillResult:
    """Generate with `skill` on `backend` (a NativeBackend). When `adapters`
    (native.lora.AdapterManager) and `skill.adapter` are given, that adapter is
    hot-swapped in for the duration of the call and restored afterward."""
    pipeline = await build_pipeline(skill)
    processors = [_SafeProcessor(s) for s in pipeline.stages]
    temp = (temperature if temperature is not None
            else skill.sampling_overrides.get("temperature", 0.7))
    text_in = f"{skill.system_prompt}\n\n{prompt}" if skill.system_prompt else prompt
    ctx = (adapters.with_adapter(skill.adapter)
           if (adapters is not None and skill.adapter) else nullcontext())

    last: NativeSkillResult | None = None
    with ctx:
        for attempt in range(max_attempts):
            r = backend.generate(text_in, max_tokens=max_tokens, temperature=temp,
                                 seed=seed + attempt, processors=processors)
            text = r.text.strip()
            valid = skill.validate_output(text)
            last = NativeSkillResult(text=text, valid=valid, attempts=attempt + 1,
                                     adapter=skill.adapter, rewinds=getattr(r, "rewinds", 0))
            if valid:
                break
    assert last is not None
    return last
