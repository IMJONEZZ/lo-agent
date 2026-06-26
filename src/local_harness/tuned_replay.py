"""Tunable replay: re-run a logged conversation under an *intervention* so it
produces something different — deterministically.

Exact replay (`events.replay.replay_run`) re-issues every logged model call with
its recorded seed and verifies the output is bit-identical. Tuned replay instead
takes the conversation up to a fork point (default: the final answer step), keeps
all the gathered evidence (logged tool results) fixed, and re-generates that step
under one of two interventions:

  - prompt optimization (GEPA / MIPRO / hand-written): swap the system instruction
    → the model synthesizes the same evidence differently.
  - logit / guidance constraint: attach a grammar/skill → the answer is forced
    into a different shape (structured summary, constrained vocabulary, …).

The seed is held fixed, so the *only* changed variable is the intervention — the
difference is attributable, and re-running the same intervention is reproducible.
This is the counterfactual the event log makes possible and a frontier API can't:
same trajectory, one knob turned, exact attribution.

First cut re-generates the single fork step (evidence held constant). Re-running
the whole trajectory under a new prompt (which could call different tools) is a
deliberate non-goal here — tool results would no longer match the log.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .events.log import MODEL_CALL, EventLog
from .inference.capabilities import Capabilities
from .inference.client import OpenAICompatClient
from .inference.types import canonical_text
from .skills.skill import Skill


@dataclass
class Intervention:
    """One knob to turn on a tuned replay. Combine system_prompt (prompt-opt) and
    skill (guidance/grammar) freely — the goal explicitly wants both."""
    label: str
    system_prompt: str | None = None   # prompt optimization: override the instruction
    skill: Skill | None = None         # guidance/grammar: constrain the answer
    extra_body: dict | None = None     # raw body params (samplers, bias, …)
    seed: int | None = None            # override the seed (default: keep the logged one)


@dataclass
class TunedReplayReport:
    run_id: str
    fork_index: int
    intervention: str
    original: str
    tuned: str
    changed: bool
    grammar_status: str | None = None  # native | emulated | unavailable (if a skill was used)
    valid: bool | None = None          # did the tuned output satisfy the skill grammar/schema
    attempts: int = 1

    def summary(self) -> str:
        lines = [
            f"tuned replay {self.run_id} @ call {self.fork_index} — intervention: {self.intervention}",
            f"  changed: {self.changed}"
            + (f" · grammar: {self.grammar_status} · valid: {self.valid}"
               if self.grammar_status else ""),
            f"  original: {self.original[:200]!r}",
            f"  tuned:    {self.tuned[:200]!r}",
        ]
        return "\n".join(lines)


def _set_system(body: dict, prompt: str) -> None:
    for m in body.get("messages", []):
        if m.get("role") == "system":
            m["content"] = prompt
            return
    body.setdefault("messages", []).insert(0, {"role": "system", "content": prompt})


def _answer_text(message: dict) -> str:
    return (message.get("content") or "").strip()


async def replay_tuned(
    log: EventLog,
    run_id: str,
    client: OpenAICompatClient,
    caps: Capabilities,
    intervention: Intervention,
    fork_index: int | None = None,
    max_grammar_retries: int = 3,
) -> TunedReplayReport:
    """Re-run the model call at `fork_index` (default: the last one — the answer)
    under `intervention`, with evidence and seed held fixed."""
    calls = log.events(run_id, type=MODEL_CALL)
    if not calls:
        raise ValueError(f"run {run_id} has no model calls to replay")
    if fork_index is None:
        fork_index = len(calls) - 1
    payload = calls[fork_index].payload
    original = canonical_text(payload["response"]["choices"][0]["message"])

    body = copy.deepcopy(payload["request_body"])
    # We re-generate the synthesis, not the tool decision: drop tools so the model
    # answers (and to avoid the logprobs+tools+stream rejection), and drop logprobs.
    for k in ("tools", "tool_choice", "logprobs", "top_logprobs"):
        body.pop(k, None)
    if intervention.seed is not None:
        body["seed"] = intervention.seed
    if intervention.system_prompt is not None:
        _set_system(body, intervention.system_prompt)

    grammar_status = None
    skill = intervention.skill
    if skill is not None:
        from .logits.grammar_stage import GrammarStage
        from .logits.pipeline import LogitPipeline
        plan = LogitPipeline().add(GrammarStage(skill)).resolve(caps)
        body.update(plan.body_params)
        st = plan.status_of("grammar")
        grammar_status = st.value if st else None
    if intervention.extra_body:
        body.update(intervention.extra_body)

    resp = await client.chat_body(body)
    message = resp.raw["choices"][0]["message"]
    tuned = canonical_text(message)
    valid = None
    attempts = 1
    if skill is not None:
        valid = skill.validate_output(_answer_text(message))
        # Tier-0 / emulated grammar: the server didn't enforce it, so validate and
        # retry with shifted seeds until it conforms (bounded).
        if not valid and grammar_status != "native":
            base_seed = body.get("seed", 1)
            for s in range(1, max_grammar_retries + 1):
                body["seed"] = base_seed + s
                resp = await client.chat_body(body)
                message = resp.raw["choices"][0]["message"]
                tuned = canonical_text(message)
                attempts += 1
                valid = skill.validate_output(_answer_text(message))
                if valid:
                    break

    return TunedReplayReport(
        run_id=run_id, fork_index=fork_index, intervention=intervention.label,
        original=original, tuned=tuned,
        changed=tuned.strip() != original.strip(),
        grammar_status=grammar_status, valid=valid, attempts=attempts)
