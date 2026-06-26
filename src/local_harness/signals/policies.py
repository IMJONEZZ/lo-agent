"""Declarative step policies: route on confidence signals.

accept / resample (new seed) / branch (Phase 3 tree search) / escalate
(bigger model) / ask (surface to the user).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .metrics import StepSignals


class Action(str, Enum):
    ACCEPT = "accept"
    RESAMPLE = "resample"
    BRANCH = "branch"
    ESCALATE = "escalate"
    ASK = "ask"


@dataclass
class PolicyDecision:
    action: Action
    reason: str = ""


@dataclass
class StepPolicy:
    min_mean_logprob: float | None = None
    max_entropy: float | None = None
    min_top2_margin: float | None = None
    on_fail: Action = Action.RESAMPLE
    max_retries: int = 2

    def evaluate(self, signals: StepSignals | None) -> PolicyDecision:
        if signals is None:
            return PolicyDecision(Action.ACCEPT, "no logprobs available")
        if self.min_mean_logprob is not None and signals.mean_logprob < self.min_mean_logprob:
            return PolicyDecision(
                self.on_fail, f"mean_logprob {signals.mean_logprob:.3f} < {self.min_mean_logprob}"
            )
        if self.max_entropy is not None and signals.mean_entropy > self.max_entropy:
            return PolicyDecision(
                self.on_fail, f"mean_entropy {signals.mean_entropy:.3f} > {self.max_entropy}"
            )
        if self.min_top2_margin is not None and signals.mean_top2_margin < self.min_top2_margin:
            return PolicyDecision(
                self.on_fail,
                f"top2_margin {signals.mean_top2_margin:.3f} < {self.min_top2_margin}",
            )
        return PolicyDecision(Action.ACCEPT)
