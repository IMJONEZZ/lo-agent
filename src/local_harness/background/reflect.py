"""Trajectory reflection: mine failed runs for lessons.

A lesson only enters memory if the model *agrees with itself* on it across K
resamples (sample-consistency, the defensible uncertainty signal — see
docs/uncertainty-done-right.md). This replaces the old gate on mean token-logprob,
which measured surface-form probability, not whether a lesson was trustworthy.
Locally the K extra samples are free, which is exactly what makes this affordable.
"""

from __future__ import annotations

from ..agent.memory import Memory
from ..events.log import MODEL_CALL, RUN_FAILED, EventLog
from ..inference.capabilities import Capabilities
from ..inference.client import OpenAICompatClient
from ..inference.types import Message, SamplingParams
from ..tree.search.self_consistency import self_consistency

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


async def reflect(
    log: EventLog,
    memory: Memory,
    client: OpenAICompatClient,
    limit: int = 5,
    max_tokens: int = 96,
    caps: Capabilities | None = None,
    consistency_n: int = 3,
    min_agreement: float = 0.5,
) -> int:
    """Mine failed runs for lessons. A lesson is kept only if at least
    `min_agreement` of `consistency_n` samples converge on it — i.e. the model is
    self-consistent about the lesson, not merely fluent."""
    caps = caps or Capabilities()
    done = 0
    for run in log.runs():
        if done >= limit or run.status != "failed" or memory.has_run(run.run_id):
            continue
        failures = log.events(run.run_id, type=RUN_FAILED)
        error = failures[-1].payload.get("error", "unknown") if failures else "unknown"
        n_calls = len(log.events(run.run_id, type=MODEL_CALL))
        messages = [Message(role="user", content=(
            "An agent run failed. State one concrete lesson for next time, one sentence.\n"
            f"Task: {run.task}\nError: {error}\nModel calls before failure: {n_calls}"
        ))]
        lesson, agreement = await self_consistency(
            client, caps, messages, n=consistency_n,
            sampling=SamplingParams(temperature=0.5, max_tokens=max_tokens,
                                    extra=dict(NO_THINK)))
        if not lesson or agreement < min_agreement:
            continue  # the model didn't agree with itself — don't trust it as durable
        memory.store("lesson", lesson, run_id=run.run_id, agreement=agreement)
        done += 1
    return done
