"""The logit pipeline: a middleware chain of token-level controls.

Each stage lowers itself to server params when the connected server supports
it (`compile_http`), or reports how it degrades. The resolved plan — which
stages ran native, emulated, or were dropped — is attached to every event so
a logged generation is fully explicable.

At Tier 4 (native backend, Phase 5) the same stages run as real per-token
logit processors via `process()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..inference.capabilities import Capabilities


class StageStatus(str, Enum):
    NATIVE = "native"        # lowered to server params
    EMULATED = "emulated"    # harness-side workaround (e.g. validate-and-retry)
    UNAVAILABLE = "unavailable"


@dataclass
class StageResolution:
    stage: str
    status: StageStatus
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@runtime_checkable
class LogitStage(Protocol):
    name: str

    def compile_http(self, caps: Capabilities) -> StageResolution: ...

    def process(self, input_ids: Any, scores: Any) -> Any:
        """Tier-4 in-process path (Phase 5). HTTP-only stages may raise."""
        ...


@dataclass
class ResolvedPlan:
    stages: list[StageResolution]
    body_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [
                {"stage": s.stage, "status": s.status.value, "note": s.note} for s in self.stages
            ],
            "body_params": self.body_params,
        }

    def status_of(self, stage_name: str) -> StageStatus | None:
        for s in self.stages:
            if s.stage == stage_name:
                return s.status
        return None


class LogitPipeline:
    def __init__(self, stages: list[LogitStage] | None = None):
        self.stages: list[LogitStage] = list(stages or [])

    def add(self, stage: LogitStage) -> "LogitPipeline":
        self.stages.append(stage)
        return self

    def resolve(self, caps: Capabilities) -> ResolvedPlan:
        resolutions: list[StageResolution] = []
        body: dict[str, Any] = {}
        for stage in self.stages:
            res = stage.compile_http(caps)
            resolutions.append(res)
            if res.status == StageStatus.NATIVE:
                for k, v in res.params.items():
                    if k in body and isinstance(body[k], dict) and isinstance(v, dict):
                        body[k] = {**body[k], **v}
                    else:
                        body[k] = v
        return ResolvedPlan(stages=resolutions, body_params=body)
