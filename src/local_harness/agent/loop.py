"""Event-sourced agent loop (Phase 1: linear state; the tree arrives in Phase 3).

Every model call and tool execution is appended to the event log before the
loop proceeds, so the log is always a valid checkpoint: `resume()` rebuilds
message state purely from events and continues — including finishing tool
calls that were requested but not yet executed when the process died.

Determinism: each model call uses seed = base_seed + call_index, so a re-run
of the same task on a Tier-1 server reproduces the original trajectory, and
`replay_run` can verify any logged run bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import httpx

from ..events.log import (
    CONTEXT_COMPACTED,
    GUARDRAIL,
    MESSAGE_SNIPPED,
    MODEL_CALL,
    POLICY_TRIGGERED,
    RUN_COMPLETED,
    RUN_FAILED,
    TOOL_CALL,
    USER_MESSAGE,
    EventLog,
)
from typing import Any, Callable

from ..guardrails.guardrails import Guardrails
from ..inference.capabilities import Capabilities
from ..inference.client import OpenAICompatClient
from ..inference.types import GenerationRequest, Message, SamplingParams
from ..signals.metrics import StepSignals
from ..signals.policies import Action, StepPolicy
from .compaction import NUDGE_NAME, compact, estimate_tokens, summarize_and_compact
from .codemode import RUN_CODE_NAME
from .tools import TOOL_SEARCH_NAME, ToolRegistry, tool_search_schema

# Above this many exposed tools, defer the deferrable (MCP/UTCP) ones behind
# tool_search instead of sending all their schemas on every request.
TOOL_DEFER_THRESHOLD = 15

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise agent. Use the available tools to complete the task. "
    "Call tools when you need information or computation; when the task is "
    "complete, reply with the final answer and no tool calls."
)

# Fronted in the system message when code-mode is on. The run_code tool
# description carries the full API reference, but many local models weight the
# system prompt far more than tool schemas (and some chat templates render
# schemas poorly) — without this, they fall back to timid one-call-per-step
# behavior and burn the step budget.
CODE_MODE_SYSTEM_NOTE = (
    "Code mode is ON: your only tool is `run_code` — write Python and `await` "
    "the `tools.*` functions listed in its description. Chain as much work as "
    "possible into ONE run_code block (loops, conditionals, many tool calls): "
    "your number of turns is budgeted, the amount of code per block is not. "
    "`print(...)` anything you need to see; end with `return <value>`.\n"
    "\n"
    "Example of a perfect run_code call — find the TODOs across the source tree "
    "and report the worst files, all in one block:\n"
    "\n"
    "```python\n"
    "hits = await tools.grep(\"TODO\", \"src\")   # returns path:line:text lines\n"
    "counts = {}\n"
    "for line in hits.splitlines():\n"
    "    path = line.split(\":\", 1)[0]\n"
    "    counts[path] = counts.get(path, 0) + 1\n"
    "worst = sorted(counts, key=counts.get, reverse=True)[:3]\n"
    "for path in worst:                          # read only the top files\n"
    "    body = await tools.read_file(path)\n"
    "    print(f\"{path}: {counts[path]} TODOs, {len(body.splitlines())} lines\")\n"
    "return {\"files_with_todos\": len(counts), \"worst\": worst}\n"
    "```\n"
    "\n"
    "Notice: every `tools.*` call is `await`ed; grep, the loop, and the reads "
    "are one block, not one tool per turn; results you want to keep are "
    "`print`ed or returned. Do NOT `import os`/`subprocess`/`open` — reach the "
    "filesystem, shell, and network only through `tools.*`."
)

# Injected as a user turn before the model's last budgeted call, so the run
# ends with an answer instead of a tool call whose result nobody will see.
FINAL_STEP_NOTE = (
    "This is your FINAL turn — the step budget is exhausted after this reply. "
    "Do not call tools. Reply now with your best final answer from what you "
    "already know, noting anything left unverified."
)


import json as _json


def _sanitize_history_message(msg: Message) -> Message:
    """Return a copy whose tool-call arguments are all valid JSON, so the message
    is safe to send back to a strict server. Truncated/garbage arguments (e.g. a
    file body cut off by max_tokens) are replaced with `{}` — the accompanying
    nudge tells the model what went wrong, and history stays server-valid."""
    if not msg.tool_calls:
        return msg
    fixed, changed = [], False
    for tc in msg.tool_calls:
        args = (tc.arguments or "").strip()
        ok = False
        if args:
            try:
                ok = isinstance(_json.loads(args), dict)
            except _json.JSONDecodeError:
                ok = False
        if ok:
            fixed.append(tc)
        else:
            changed = True
            fixed.append(replace(tc, arguments="{}"))
    return msg if not changed else replace(msg, tool_calls=fixed)


@dataclass
class AgentResult:
    run_id: str
    answer: str
    model_calls: int
    status: str  # completed | max_steps


class Agent:
    def __init__(
        self,
        client: OpenAICompatClient,
        tools: ToolRegistry,
        log: EventLog,
        capabilities: Capabilities | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling: SamplingParams | None = None,
        base_seed: int = 1,
        max_steps: int = 95,
        policy: StepPolicy | None = None,
        guardrails_factory: Callable[[], Guardrails] | None = None,
        context_budget: int | None = None,
        compact_fraction: float = 0.85,
        compaction_strategy: str = "summarize",
        on_token: Callable[[str, str], None] | None = None,
        notebook=None,
        exposed_tools: set[str] | None = None,
        retrieval=None,
        on_notice: Callable[[str], None] | None = None,
        on_tool: Callable[[str, str], None] | None = None,
        on_compact: Callable[[str, dict[str, Any]], None] | None = None,
        code_mode: bool = False,
        sandbox=None,
        model: str | None = None,
    ):
        self.client = client
        self.tools = tools
        # Per-agent model override (preset `model:`). None → use the client's
        # model. Sent as the request's model field, so a multi-model server
        # (LM Studio, a router) can serve e.g. plan on a big model, build on a
        # fast one. Capabilities stay those probed for the client's model.
        self.model = model
        # Code-mode: expose ONE run_code tool the model writes Python against,
        # instead of N per-tool schemas (fewer round-trips / tokens). Off by
        # default for the library; the TUI/CLI/server turn it on.
        self.code_mode = code_mode
        self.sandbox = sandbox  # follows code-mode execution (microVM bridge when set)
        self.log = log
        self.caps = capabilities or Capabilities()
        self.system_prompt = system_prompt
        self.sampling = sampling or SamplingParams(temperature=0.2)  # no token cap by default
        self.base_seed = base_seed
        self.max_steps = max_steps
        self.policy = policy
        self.guardrails_factory = guardrails_factory
        # Auto-compaction trigger. An explicit context_budget wins; otherwise lock
        # it at `compact_fraction` (default 0.85) of the model's probed context
        # window — so the harness self-tunes the compaction point to whatever box
        # it connected to, no manual budget needed (Claude-Code-style auto-compact).
        self.compact_fraction = compact_fraction
        self.compaction_strategy = compaction_strategy  # "summarize" | "mechanical"
        self.context_budget = context_budget
        self.auto_budget = False
        if self.context_budget is None and self.caps.context_window:
            self.context_budget = int(self.caps.context_window * compact_fraction)
            self.auto_budget = True
        self.on_compact = on_compact  # (phase, info) progress callback for a compaction UI
        self.on_token = on_token  # (kind, text): "start" | "content" | "reasoning"
        self.notebook = notebook
        self.exposed_tools = exposed_tools  # restrict which tools the model sees (preset)
        self.retrieval = retrieval  # an agent.memory.Memory for retrieval-augmented runs
        self.on_notice = on_notice  # one-off human-readable notices (e.g. degradation)
        self.on_tool = on_tool      # (name, phase) — phase "start"/"done" for a live UI
        self._degraded = False  # latched once we've stripped picky params for this server
        # Confidence calibrated to the sampler chain (llama.cpp post_sampling_probs)
        self._post_sampling = bool(self.caps.logprobs and self.caps.post_sampling_probs)
        self._promoted: set[str] = set()  # tools surfaced via tool_search this run (ToolSearch)
        self.inbox = None  # optional () -> list[str]: peer messages to inject (Coordinator)

    def _notice(self, msg: str) -> None:
        if self.on_notice is not None:
            self.on_notice(msg)

    async def _send(self, body: dict, stream: bool):
        """Send a chat request, degrading gracefully for picky providers.

        Other harnesses work where ours didn't because we add `logprobs`/`seed`
        for determinism + confidence; some OpenAI-compatible servers 4xx on those
        (especially combined with `tools`). On such a rejection we retry once
        without them so the turn still completes — confidence/replay are the only
        casualties, and we say so. The strip is latched per server."""
        async def _do(b):
            if stream and self.on_token is not None:
                return await self.client.chat_body_stream(b, self.on_token)
            return await self.client.chat_body(b)

        picky = ("logprobs", "top_logprobs", "seed")
        if self._degraded:  # already learned this server is picky — don't re-send the bad body
            body = {k: v for k, v in body.items() if k not in picky}
        try:
            return await _do(body)
        except httpx.HTTPStatusError as e:
            stripped = {k: v for k, v in body.items() if k not in picky}
            if e.response.status_code >= 500 or stripped == body:
                raise
            if not self._degraded:
                self._degraded = True
                self._notice("server rejected logprobs/seed — retrying without them; "
                             "confidence overlay and bit-exact replay are off for this server.")
            return await _do(stripped)

    def _tool_schemas(self) -> list[dict]:
        schemas = self.tools.schemas()
        if self.exposed_tools is not None:
            schemas = [s for s in schemas if s["function"]["name"] in self.exposed_tools]
        # Code-mode: replace all per-tool schemas with one run_code tool whose
        # description is a compact API reference for the exposed tools.
        if self.code_mode:
            from .codemode import api_reference, run_code_schema
            exposed = {s["function"]["name"] for s in schemas}
            return [run_code_schema(api_reference(self.tools, exposed))]
        # Lazy tool loading (ToolSearch): once the toolset is large, defer the
        # deferrable (MCP/UTCP) tools the model hasn't searched up yet behind a
        # single tool_search meta-tool, so their schemas don't bloat every request.
        deferrable = self.tools.deferrable_names()
        deferred = [s["function"]["name"] for s in schemas
                    if s["function"]["name"] in deferrable
                    and s["function"]["name"] not in self._promoted]
        if len(schemas) <= TOOL_DEFER_THRESHOLD or not deferred:
            return schemas
        kept = [s for s in schemas if s["function"]["name"] not in deferred]
        return kept + [tool_search_schema(len(deferred))]

    def _call_confidence(self, response) -> float | None:
        signals = StepSignals.from_logprobs(
            response.logprobs or [], post_sampling=self._post_sampling
        )
        return signals.mean_logprob if signals else None

    def _system_message(self, task: str | None = None) -> Message:
        """System prompt + frozen self-editing memory + lessons retrieved for the
        task (run-start, so it stays inside determinism/replay)."""
        content = self.system_prompt
        if self.code_mode:
            content = f"{content}\n\n{CODE_MODE_SYSTEM_NOTE}"
        if self.notebook is not None:
            block = self.notebook.system_block()
            if block:
                content = f"{content}\n\n{block}"
        if self.retrieval is not None and task:
            hits = self.retrieval.recall(task, limit=3)
            if hits:
                lessons = "\n".join(f"- {h.text}" for h in hits)
                content = f"{content}\n\n## Relevant notes from past runs\n{lessons}"
        return Message(role="system", content=content)

    async def run(self, task: str, run_id: str | None = None) -> AgentResult:
        # The session server pre-creates the run (so it can bind run-scoped
        # stream callbacks before the agent starts); the CLI lets us create it.
        if run_id is None:
            run_id = self.log.create_run(task)
        messages = [
            self._system_message(task),
            Message(role="user", content=task),
        ]
        guardrails = self.guardrails_factory() if self.guardrails_factory else None
        return await self._loop(run_id, messages, call_index=0, guardrails=guardrails)

    async def resume(self, run_id: str) -> AgentResult:
        meta = self.log.run(run_id)
        if meta is None:
            raise ValueError(f"unknown run: {run_id}")
        if meta.status == "completed":
            done = self.log.events(run_id, type=RUN_COMPLETED)
            calls = len(self.log.events(run_id, type=MODEL_CALL))
            return AgentResult(run_id, done[-1].payload["answer"], calls, "completed")

        messages, call_index, pending = self._reconstruct(run_id, meta.task)
        self._replay_repl(run_id)  # rebuild persistent REPL state after a restart
        guardrails = self.guardrails_factory() if self.guardrails_factory else None
        if guardrails is not None:
            # Rebuild step state from the log (error budgets restart fresh).
            executed = [e.payload["name"] for e in self.log.events(run_id, type=TOOL_CALL)]
            guardrails.steps.record(executed)
        # Finish tool calls that were requested but not executed before the crash.
        for tc in pending:
            messages.append(await self._execute_tool(run_id, tc))
            if guardrails is not None:
                guardrails.steps.record([tc.name])
        return await self._loop(run_id, messages, call_index, guardrails)

    async def continue_run(self, run_id: str, message: str) -> AgentResult:
        """Continue a finished (or interrupted) run with a new user turn — the
        same run_id, so the conversation and its full event history carry on."""
        meta = self.log.run(run_id)
        if meta is None:
            raise ValueError(f"unknown run: {run_id}")
        self.log.reopen(run_id)
        self.log.append(run_id, USER_MESSAGE, {"content": message})
        messages, call_index, pending = self._reconstruct(run_id, meta.task)
        self._replay_repl(run_id)  # rebuild persistent REPL state after a restart
        guardrails = self.guardrails_factory() if self.guardrails_factory else None
        if guardrails is not None:
            executed = [e.payload["name"] for e in self.log.events(run_id, type=TOOL_CALL)]
            guardrails.steps.record(executed)
        for tc in pending:
            messages.append(await self._execute_tool(run_id, tc))
            if guardrails is not None:
                guardrails.steps.record([tc.name])
        return await self._loop(run_id, messages, call_index, guardrails)

    def _replay_repl(self, run_id: str) -> None:
        """Rebuild persistent REPL state after a process restart by re-running the
        run's logged repl cells in order. Sessions already alive in THIS process are
        skipped, so a same-process continue doesn't double-execute their side effects."""
        import json as _json

        from .tools import _REPL
        live = set(_REPL._ns)  # sessions with live state here → leave them be
        for ev in self.log.events(run_id, type=TOOL_CALL):
            if ev.payload.get("name") != "repl":
                continue
            try:
                args = _json.loads(ev.payload.get("arguments") or "{}")
            except (ValueError, TypeError):
                continue
            session = args.get("session", "default")
            if session in live:
                continue
            _REPL.run(args.get("code", ""), session=session, reset=bool(args.get("reset", False)))

    # --- internals -----------------------------------------------------

    async def _loop(
        self,
        run_id: str,
        messages: list[Message],
        call_index: int,
        guardrails: Guardrails | None = None,
    ) -> AgentResult:
        try:
            while call_index < self.max_steps:
                if self.inbox is not None:  # Coordinator: inject any peer messages
                    for msg in self.inbox():
                        messages.append(Message(role="user", content=msg))
                        self.log.append(run_id, USER_MESSAGE, {"content": msg})
                if (self.context_budget is not None
                        and estimate_tokens(messages) > self.context_budget):
                    messages = await self._compact(run_id, messages)
                if call_index == self.max_steps - 1:
                    # Same shape as an inbox injection, so resume/replay
                    # reconstruction stays faithful.
                    messages.append(Message(role="user", content=FINAL_STEP_NOTE))
                    self.log.append(run_id, USER_MESSAGE, {"content": FINAL_STEP_NOTE,
                                                           "source": "harness"})
                response, call_index = await self._model_call(run_id, messages, call_index)

                if guardrails is None:
                    messages.append(_sanitize_history_message(response.message))
                    if not response.message.tool_calls:
                        answer = response.text
                        self.log.append(run_id, RUN_COMPLETED, {"answer": answer})
                        return AgentResult(run_id, answer, call_index, "completed")
                    conf = self._call_confidence(response)
                    for tc in response.message.tool_calls:
                        messages.append(await self._execute_tool(run_id, tc, confidence=conf))
                    continue

                result = guardrails.check(response.message)
                payload = {
                    "call_index": call_index - 1, "action": result.action,
                    "kind": result.nudge.kind if result.nudge else None,
                    "rescued": result.rescued, "reason": result.reason,
                }
                if result.nudge is not None:
                    payload["nudge"] = {"role": result.nudge.role,
                                        "content": result.nudge.content,
                                        "tool_call_id": result.nudge.tool_call_id}
                if result.rescued:
                    payload["rescued_calls"] = [tc.to_dict() for tc in result.tool_calls]
                self.log.append(run_id, GUARDRAIL, payload)

                if result.action == "fatal":
                    self.log.append(run_id, RUN_FAILED, {"error": f"guardrails: {result.reason}"})
                    return AgentResult(run_id, "", call_index, "fatal")

                if result.action == "final":
                    messages.append(response.message)
                    answer = response.text
                    self.log.append(run_id, RUN_COMPLETED, {"answer": answer})
                    return AgentResult(run_id, answer, call_index, "completed")

                if result.action == "nudge":
                    # A malformed/truncated tool call must NOT go back to the server
                    # verbatim — a tool_call whose arguments aren't valid JSON makes
                    # strict servers (vLLM) 400 the *next* request. Sanitize first.
                    safe = _sanitize_history_message(response.message)
                    messages.append(safe)
                    messages.extend(self._nudge_messages(safe, result.nudge))
                    continue

                # execute — rescued calls are grafted onto the assistant message
                # so the tool-channel pairing stays template-valid.
                message = response.message
                if result.rescued:
                    message = Message(role="assistant", content=message.content,
                                      tool_calls=result.tool_calls)
                messages.append(message)
                conf = self._call_confidence(response)
                had_errors = False
                for tc in result.tool_calls:
                    tool_msg = await self._execute_tool(run_id, tc, confidence=conf)
                    had_errors = had_errors or (tool_msg.content or "").startswith("error:")
                    messages.append(tool_msg)
                fatal = guardrails.record([tc.name for tc in result.tool_calls], had_errors)
                if fatal:
                    self.log.append(run_id, RUN_FAILED, {"error": f"guardrails: {fatal}"})
                    return AgentResult(run_id, "", call_index, "fatal")

            self.log.append(run_id, RUN_FAILED, {"error": "max_steps exceeded"})
            return AgentResult(run_id, "", call_index, "max_steps")
        except httpx.HTTPStatusError as e:
            # Surface the server's actual reason (e.g. vLLM's 400 message), not just
            # the bare status, so failures are diagnosable from the transcript.
            try:
                detail = e.response.text[:400]
            except Exception:
                detail = ""
            self.log.append(run_id, RUN_FAILED, {
                "error": f"HTTP {e.response.status_code} from {e.request.url}: {detail}"})
            raise
        except Exception as e:
            self.log.append(run_id, RUN_FAILED, {"error": f"{type(e).__name__}: {e}"})
            raise

    async def _compact(self, run_id: str, messages: list[Message]) -> list[Message]:
        """Trigger crossed: shrink the in-flight context and log a
        CONTEXT_COMPACTED event. Default strategy summarizes the old turns with a
        model call (Claude-Code parity); "mechanical" skips the call (Tier-0 /
        deterministic-replay safe). Full history stays in the log either way."""
        if self.compaction_strategy == "mechanical":
            before = estimate_tokens(messages)
            if self.on_compact:
                self.on_compact("start", {"method": "mechanical", "before_tokens": before,
                                          "trigger_tokens": self.context_budget,
                                          "context_window": self.caps.context_window, "frac": 0.0})
            new_messages, phase = compact(messages, self.context_budget)
            info = {"method": "mechanical", "before_tokens": before,
                    "after_tokens": estimate_tokens(new_messages),
                    "trigger_tokens": self.context_budget,
                    "context_window": self.caps.context_window, "summary": None, "phase": phase}
            if self.on_compact:
                self.on_compact("done", dict(info, frac=1.0))
            self.log.append(run_id, CONTEXT_COMPACTED, info)
            return new_messages

        new_messages, info = await summarize_and_compact(
            self.client, messages,
            trigger_tokens=self.context_budget,
            context_window=self.caps.context_window,
            on_compact=self.on_compact,
        )
        self.log.append(run_id, CONTEXT_COMPACTED, info)
        return new_messages

    def _nudge_messages(self, assistant_msg: Message, nudge) -> list[Message]:
        """Materialize a Nudge, keeping the tool channel template-valid: every
        tool_call id in the assistant message gets a tool reply."""
        out: list[Message] = []
        if nudge.role == "tool" and assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                content = nudge.content if tc.id == nudge.tool_call_id else \
                    "not executed: a sibling tool call in this batch was rejected"
                out.append(Message(role="tool", content=content, tool_call_id=tc.id,
                                   name=NUDGE_NAME))
        else:
            out.append(Message(role="user", content=nudge.content, name=NUDGE_NAME))
        return out

    async def _model_call(self, run_id: str, messages: list[Message], call_index: int):
        # Uncertainty-aware control flow: a step the policy rejects is
        # resampled with a shifted seed; every attempt is its own MODEL_CALL
        # event so replay reproduces the whole trajectory, retries included.
        attempts = (self.policy.max_retries + 1) if self.policy else 1
        response = None
        for attempt in range(attempts):
            response = await self._one_call(run_id, messages, call_index, attempt)
            if self.policy is None:
                break
            signals = StepSignals.from_logprobs(
            response.logprobs or [], post_sampling=self._post_sampling
        )
            decision = self.policy.evaluate(signals)
            if decision.action == Action.ACCEPT:
                break
            self.log.append(
                run_id,
                POLICY_TRIGGERED,
                {"call_index": call_index, "attempt": attempt, "action": decision.action.value,
                 "reason": decision.reason},
            )
            if decision.action != Action.RESAMPLE:
                break  # BRANCH arrives with the tree (Phase 3); ESCALATE/ASK surface upward
        return response, call_index + 1

    async def _one_call(self, run_id: str, messages: list[Message], call_index: int, attempt: int):
        # Seed is always sent (unsupported servers ignore it); logprobs only
        # when verified, since some servers reject the param outright.
        # Resample attempts shift by 1000 to stay clear of call_index seeds.
        sampling = replace(
            self.sampling,
            seed=self.base_seed + call_index + 1000 * attempt,
            logprobs=self.caps.logprobs,
        )
        if self._post_sampling:
            # Ask for probabilities of the distribution actually sampled from
            # (post sampler chain) — raw logprobs miscalibrate confidence once
            # min_p/XTC/etc. truncate the distribution they describe.
            sampling.extra = {**sampling.extra, "post_sampling_probs": True}
        request = GenerationRequest(
            messages=messages, sampling=sampling, tools=self._tool_schemas()
        )
        body = request.to_body(self.model or self.client.model)
        if self.on_token is not None:
            self.on_token("start", "")
            # The `logprobs + tools + stream` combo is the single least-portable
            # request shape — vLLM and others reject it or stream malformed
            # tool-call deltas, version-dependent — and it's the one thing we did
            # that other harnesses don't. So never stream that combo: on a
            # tool-calling turn, strip logprobs from the streamed body, and only
            # recover them with a deterministic non-streamed re-pass when a policy
            # actually consumes them. Tool-free answer turns still stream logprobs
            # (the confidence overlay), and the re-pass is identical by seed.
            wants_lp = bool(body.get("logprobs") and body.get("tools"))
            if not wants_lp:
                response = await self._send(body, stream=True)
            else:
                stream_body = {k: v for k, v in body.items()
                               if k not in ("logprobs", "top_logprobs")}
                if self.policy is not None:
                    await self._send(stream_body, stream=True)         # typing feel
                    response = await self._send(body, stream=False)    # recover logprobs
                else:
                    response = await self._send(stream_body, stream=True)
        else:
            response = await self._send(body, stream=False)

        signals = StepSignals.from_logprobs(
            response.logprobs or [], post_sampling=self._post_sampling
        )
        self.log.append(
            run_id,
            MODEL_CALL,
            {
                "call_index": call_index,
                "seed": sampling.seed,
                "request_body": body,
                "response": response.raw,
                "timing_ms": response.timing_ms,
                "logprob_summary": signals.to_dict() if signals else None,
            },
        )
        return response

    async def _run_code(self, arguments: str) -> str:
        """Code-mode: run the model's Python with the tools bound, enforcing the
        same exposed-tool policy. Execution follows the sandbox (in-process or VM)."""
        from .codemode import CodeMode
        try:
            code = (_json.loads(arguments) if arguments.strip() else {}).get("code", "")
        except _json.JSONDecodeError:
            code = arguments
        cm = CodeMode(self.tools, exposed=self.exposed_tools, sandbox=self.sandbox)
        return await cm.run(code)

    def _run_tool_search(self, arguments: str) -> str:
        """The tool_search meta-tool: BM25-rank deferred tools for the query, mark
        the matches promoted (so their full schemas appear next step), and return a
        callable summary (names + params + descriptions)."""
        import json as _json
        try:
            query = (_json.loads(arguments) if arguments.strip() else {}).get("query", "")
        except _json.JSONDecodeError:
            query = arguments
        matches = self.tools.search(query, limit=5)
        if not matches:
            return f"tool_search: no tools matched {query!r}. Try different keywords."
        lines = [f"Loaded {len(matches)} tool(s) — you can call them now:"]
        for name, desc in matches:
            self._promoted.add(name)
            tool = self.tools.get(name)
            params = ", ".join(((tool.parameters or {}).get("properties") or {}).keys()) if tool else ""
            lines.append(f"- {name}({params}): {desc}")
        return "\n".join(lines)

    async def _execute_tool(self, run_id: str, tc, confidence: float | None = None) -> Message:
        if self.on_tool is not None:
            self.on_tool(tc.name, "start")   # a live UI shows "running <tool>…"
        try:
            if tc.name == TOOL_SEARCH_NAME:  # agent meta-tool, not in the registry
                result = self._run_tool_search(tc.arguments)
            elif tc.name == RUN_CODE_NAME:   # code-mode: run the model's Python
                result = await self._run_code(tc.arguments)
            else:
                result = await self.tools.execute(tc.name, tc.arguments, confidence=confidence)
        finally:
            if self.on_tool is not None:
                self.on_tool(tc.name, "done")
        self.log.append(
            run_id,
            TOOL_CALL,
            {"tool_call_id": tc.id, "name": tc.name, "arguments": tc.arguments, "result": result},
        )
        return Message(role="tool", content=result, tool_call_id=tc.id, name=tc.name)

    def _reconstruct(self, run_id: str, task: str):
        """Rebuild message state from the event log.

        Returns (messages, call_index, pending_tool_calls) where pending are
        tool calls the last assistant turn requested but never executed.
        """
        messages = [
            self._system_message(task),
            Message(role="user", content=task),
        ]
        call_index = 0
        pending: dict[str, object] = {}

        # /snip: collapse the content of these events in the rebuilt context to free
        # space (lossless — the originals stay in the log; only the projection shrinks).
        events = self.log.events(run_id)
        snipped = {e.payload.get("seq") for e in events if e.type == MESSAGE_SNIPPED}

        last_call_index = None
        for event in events:
            if event.type == MODEL_CALL:
                msg = Message.from_dict(event.payload["response"]["choices"][0]["message"])
                if event.seq in snipped:  # keep tool_calls (channel pairing), drop the prose
                    msg.content = "[snipped to free context]"
                # Policy resamples log multiple MODEL_CALLs for one call_index;
                # only the last attempt belongs in the reconstructed transcript.
                if event.payload["call_index"] == last_call_index and messages:
                    messages.pop()
                last_call_index = event.payload["call_index"]
                messages.append(msg)
                call_index = event.payload["call_index"] + 1
                pending = {tc.id: tc for tc in msg.tool_calls}
            elif event.type == USER_MESSAGE:
                messages.append(Message(role="user", content=event.payload["content"]))
                pending = {}
            elif event.type == TOOL_CALL:
                p = event.payload
                content = "[snipped to free context]" if event.seq in snipped else p["result"]
                messages.append(
                    Message(
                        role="tool",
                        content=content,
                        tool_call_id=p["tool_call_id"],
                        name=p["name"],
                    )
                )
                pending.pop(p["tool_call_id"], None)
            elif event.type == GUARDRAIL:
                p = event.payload
                if p.get("rescued") and p.get("rescued_calls") and messages:
                    # Re-graft rescued calls onto the assistant message so the
                    # tool replies that follow stay paired.
                    from ..inference.types import ToolCallRequest

                    last = messages[-1]
                    if last.role == "assistant" and not last.tool_calls:
                        last.tool_calls = [ToolCallRequest.from_dict(d)
                                           for d in p["rescued_calls"]]
                        pending = {tc.id: tc for tc in last.tool_calls}
                elif p.get("nudge"):
                    n = p["nudge"]
                    if n["role"] == "tool" and messages and messages[-1].tool_calls:
                        for tc in messages[-1].tool_calls:
                            content = n["content"] if tc.id == n.get("tool_call_id") else \
                                "not executed: a sibling tool call in this batch was rejected"
                            messages.append(Message(role="tool", content=content,
                                                    tool_call_id=tc.id, name=NUDGE_NAME))
                        pending = {}
                    else:
                        messages.append(Message(role="user", content=n["content"],
                                                name=NUDGE_NAME))

        return messages, call_index, list(pending.values())
