"""Proxy engine: one OpenAI-shaped request in, one OpenAI-shaped response out,
with the logit pipeline and guardrails applied in between.

The client thinks it's talking to a smarter, better-behaved model:
- pipeline params (grammar, samplers, bias, thinking control) are compiled to
  upstream-native body params per the probed capabilities
- tool calls embedded in prose are rescued into native tool_calls
- unknown-tool / malformed-args responses are retried internally with nudges
  before the client ever sees them
- grammar/json_schema constraints are validate-and-retried when the upstream
  can't enforce them natively
- think-budget and anti-slop requests route through the raw-completion paths

Every proxied call logs to the event store — `harness replay` works on proxy
traffic exactly like on agent runs.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from ..events.log import GUARDRAIL, MODEL_CALL, RUN_COMPLETED, EventLog
from ..guardrails.validator import ResponseValidator
from ..inference.capabilities import Capabilities, probe
from ..inference.client import OpenAICompatClient
from ..inference.types import Message
from ..logits.antislop import generate_antislop
from ..logits.bias import BiasProfileStore, BiasStage
from ..logits.budget import generate_with_think_budget
from ..logits.grammar_stage import GrammarStage
from ..logits.pipeline import LogitPipeline
from ..logits.samplers import SamplerChain
from ..skills.skill import Skill, SkillRegistry

SKIP_THINK_PREFILL = "<think>\n\n</think>\n\n"


def _synthesize(model: str, content: str, reasoning: str | None = None,
                usage: dict | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": f"harness-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage or {},
    }


class ProxyEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAICompatClient(cfg.upstream_url, cfg.model)
        self.caps: Capabilities = Capabilities()
        self.log = EventLog(cfg.db)
        self.skills = SkillRegistry(cfg.skills_dir)
        self.profiles = BiasProfileStore(cfg.profiles_dir)
        self._bias_cache: dict[str, BiasStage] = {}

    async def start(self) -> None:
        if not self.client.model:
            models = await self.client.list_models()
            self.client.model = models[0] if models else ""
        self.caps = await probe(self.client)

    # --- pipeline assembly -------------------------------------------------

    async def _bias_stage(self, profile_name: str) -> BiasStage:
        if profile_name not in self._bias_cache:
            stage = BiasStage(self.profiles.get(profile_name))
            await stage.resolve_tokens(self.client)
            self._bias_cache[profile_name] = stage
        return self._bias_cache[profile_name]

    async def _resolve_pipeline(self, ext: dict, body: dict) -> tuple[dict, Skill | None, bool]:
        """Returns (body_params, validation_skill, emulated)."""
        pipeline = LogitPipeline()
        skill: Skill | None = None
        emulated = False
        if ext.get("skill"):
            skill = self.skills.get(ext["skill"])
            pipeline.add(GrammarStage(skill))
        elif isinstance(body.get("response_format"), dict):
            rf = body["response_format"]
            schema = (rf.get("json_schema") or {}).get("schema") if rf.get("type") == "json_schema" else None
            if schema:
                skill = Skill(name="client_schema", json_schema=schema)
                if self.caps.grammar is None and self.caps.server != "llama.cpp":
                    body.pop("response_format", None)  # upstream can't enforce; we emulate
                    emulated = True
        if ext.get("samplers"):
            pipeline.add(SamplerChain(ext["samplers"]))
        if ext.get("bias_profile"):
            pipeline.add(await self._bias_stage(ext["bias_profile"]))

        plan = pipeline.resolve(self.caps)
        grammar_status = plan.status_of("grammar")
        emulated = emulated or (grammar_status is not None and grammar_status.value == "emulated")
        return plan.body_params, skill, emulated

    # --- request handling ----------------------------------------------------

    async def handle_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        ext = self.cfg.merged_ext(body.pop("harness", None))
        body.setdefault("model", self.client.model)
        run_id = self.log.create_run(f"proxy:{body['model']}")
        has_tools = bool(body.get("tools"))

        # Raw-completion special paths (llama.cpp upstream, non-tool requests).
        if not has_tools and self.caps.raw_completion:
            messages = [Message.from_dict(m) for m in body.get("messages", [])]
            if ext.get("think_budget"):
                r = await generate_with_think_budget(
                    self.client, messages, think_budget=int(ext["think_budget"]),
                    answer_max_tokens=body.get("max_tokens", 512), seed=body.get("seed"),
                )
                response = _synthesize(body["model"], r.answer, reasoning=r.reasoning)
                self.log.append(run_id, RUN_COMPLETED, {"answer": r.answer})
                return response
            if ext.get("banned_phrases"):
                r = await generate_antislop(
                    self.client, messages, list(ext["banned_phrases"]),
                    max_tokens=body.get("max_tokens", 256), seed=body.get("seed"),
                    prefill=SKIP_THINK_PREFILL,
                )
                response = _synthesize(body["model"], r.text.strip())
                self.log.append(run_id, GUARDRAIL, {
                    "action": "antislop", "kind": "antislop", "rescued": False,
                    "reason": f"{r.rewinds} rewinds",
                })
                self.log.append(run_id, RUN_COMPLETED, {"answer": r.text.strip()})
                return response

        plan_params, skill, emulated = await self._resolve_pipeline(ext, body)
        tool_names = [t["function"]["name"] for t in body.get("tools", [])]
        validator = (
            ResponseValidator(tool_names, rescue_enabled=bool(ext.get("rescue")))
            if tool_names else None
        )

        messages_work = list(body.get("messages", []))
        base_seed = body.get("seed")
        raw: dict[str, Any] = {}

        for attempt in range(self.cfg.max_internal_retries + 1):
            send = {**plan_params, **body, "messages": messages_work}
            if base_seed is not None and attempt:
                send["seed"] = base_seed + 1000 * attempt
            response = await self.client.chat_body(send)
            raw = response.raw
            self.log.append(run_id, MODEL_CALL, {
                "call_index": attempt, "seed": send.get("seed"),
                "request_body": send, "response": raw,
                "timing_ms": response.timing_ms, "logprob_summary": None,
            })
            message = response.message

            if validator is not None:
                v = validator.validate(message)
                if v.rescued:
                    choice = raw["choices"][0]
                    choice["message"]["tool_calls"] = [tc.to_dict() for tc in v.tool_calls]
                    choice["finish_reason"] = "tool_calls"
                    self.log.append(run_id, GUARDRAIL, {
                        "action": "execute", "kind": None, "rescued": True, "reason": None,
                    })
                    break
                if v.nudge is not None:
                    self.log.append(run_id, GUARDRAIL, {
                        "action": "nudge", "kind": v.nudge.kind, "rescued": False,
                        "reason": None,
                    })
                    messages_work = messages_work + [message.to_dict()]
                    if v.nudge.role == "tool" and message.tool_calls:
                        for tc in message.tool_calls:
                            content = v.nudge.content if tc.id == v.nudge.tool_call_id else \
                                "not executed: a sibling tool call in this batch was rejected"
                            messages_work.append({"role": "tool", "content": content,
                                                  "tool_call_id": tc.id})
                    else:
                        messages_work.append({"role": "user", "content": v.nudge.content})
                    continue

            if skill is not None and emulated and not message.tool_calls:
                if not skill.validate_output((message.content or "").strip()):
                    if base_seed is None:
                        base_seed = 1
                    self.log.append(run_id, GUARDRAIL, {
                        "action": "nudge", "kind": "grammar_retry", "rescued": False,
                        "reason": "output failed grammar/schema validation",
                    })
                    continue
            break

        answer = (raw.get("choices", [{}])[0].get("message", {}) or {}).get("content") or ""
        self.log.append(run_id, RUN_COMPLETED, {"answer": answer})
        return raw

    async def forward_models(self) -> dict[str, Any]:
        resp = await self.client.get("/v1/models")
        return resp.json()
