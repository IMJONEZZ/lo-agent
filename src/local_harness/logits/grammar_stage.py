"""Grammar constraint as a pipeline stage: native where the server supports
it, validate-and-retry emulation at Tier 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..inference.capabilities import Capabilities
from .pipeline import StageResolution, StageStatus

if TYPE_CHECKING:  # avoid a circular import; Skill is only needed for typing
    from ..skills.skill import Skill


@dataclass
class GrammarStage:
    skill: "Skill"
    name: str = "grammar"
    # Reasoning-model templates auto-open a <think> block; a server-enforced
    # grammar would then constrain the *reasoning* and the parsed content comes
    # back empty. Constrained skills therefore disable thinking by default.
    disable_thinking: bool = True

    def _native(self, params: dict) -> StageResolution:
        if self.disable_thinking:
            params = {**params, "chat_template_kwargs": {"enable_thinking": False}}
        return StageResolution(self.name, StageStatus.NATIVE, params)

    def compile_http(self, caps: Capabilities) -> StageResolution:
        if self.skill.json_schema is not None:
            if caps.server == "llama.cpp":
                return self._native({"json_schema": self.skill.json_schema})
            if caps.grammar == "guided":
                return self._native({"guided_json": self.skill.json_schema})
            return StageResolution(self.name, StageStatus.EMULATED, note="validate-and-retry")
        if self.skill.grammar is not None:
            if caps.grammar == "gbnf":
                return self._native({"grammar": self.skill.grammar.to_gbnf()})
            if caps.grammar == "guided":
                # vLLM's `guided_grammar` is xgrammar-backed and expects GBNF/EBNF,
                # NOT Lark — verified live on .33 (vLLM): a GBNF grammar binds the
                # content channel, the Lark form is silently ignored and the output
                # comes back unconstrained. SGLang's guided_grammar is also GBNF via
                # xgrammar; only the older outlines backend wanted Lark, which we no
                # longer target. Send GBNF.
                return self._native({"guided_grammar": self.skill.grammar.to_gbnf()})
            return StageResolution(self.name, StageStatus.EMULATED, note="validate-and-retry")
        return StageResolution(self.name, StageStatus.NATIVE, {})  # unconstrained skill

    def process(self, input_ids, scores):
        raise NotImplementedError("token-level grammar masking arrives with the native backend")
