"""Goal mode (/goal, /guided-goal): the pure part.

A goal is a condition the agent keeps working toward, one round at a time, until
it is met or something stops it. A round is one agent turn. This module decides
what happens next as a pure function of a run's events (`decide`), and holds the
prompts, the judge's request and answer parsing, and the `goal` tool. The session
server (server/sessions.py) does the side effects: it starts rounds, calls the
judge, runs the verify command, logs what each produced, and asks `decide` again.
Because every input is in the event log, a goal can be replayed and resumed, and
nothing here reads the clock.

Events (events/log.py): GOAL_SET starts a goal; GOAL_ROUND opens each round (the
turn after it belongs to that round); GOAL_CLAIM records a call to the `goal`
tool; GOAL_JUDGE and GOAL_VERIFY record the round's check; GOAL_STATUS records a
stop, a clear, or a resume.

The rules:
- After every round the goal is checked. If the model didn't call the goal tool,
  the judge answers true or false to "Is the goal complete based on the work so
  far?". If the model called it with "complete" and the goal has no verify
  command, the judge gives a verdict: met, not yet met, or impossible, with a
  reason. The reason is passed to the next round.
- A verify command decides whenever the goal has one: it runs after a
  "complete" claim, or after the judge answers true. Exit code 0 means met, 1
  means not yet, and any other code means the check itself is broken, so the
  goal pauses.
- A round that ended because its step budget ran out is still checked: running
  out of steps is not the same as finishing.
- At most `cap` rounds run (150 by default); the next one would be over the cap.
  The per-turn step budget still applies inside each round, so a round that keeps
  calling tools ends too.
- STALL_ROUNDS rounds in a row with no new work (no tool calls, or the same tool
  calls as the round before) stop the goal as stalled.
- An interrupted round pauses the goal. A round that fails with an error is
  retried, up to RETRY_ROUNDS failed rounds in a row, then the goal pauses.
- Resuming starts a new stretch: the round count goes back to 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..events.log import (
    GOAL_CLAIM,
    GOAL_JUDGE,
    GOAL_ROUND,
    GOAL_SET,
    GOAL_STATUS,
    GOAL_VERIFY,
    MODEL_CALL,
    RUN_COMPLETED,
    RUN_FAILED,
    TOOL_CALL,
)
from ..skills.ir import Grammar
from .tools import Tool

DEFAULT_CAP = 150     # rounds per stretch
STALL_ROUNDS = 3      # rounds in a row without new work before the goal stops as stalled
RETRY_ROUNDS = 3      # failed rounds in a row before the goal pauses
DEFAULT_VERIFY_TIMEOUT = 600  # seconds the verify command may run (a test suite can take minutes)
GOAL_TOOL_NAME = "goal"
MODES = ("goal", "guided")  # /goal, /guided-goal
JUDGE_SEED_OFFSET = 9_000_000  # judge seeds sit far from the agent's base_seed + call_index

# Goal statuses. ACTIVE is the only one in which rounds run. A goal stopped as
# PAUSED, BLOCKED, IMPOSSIBLE, CAPPED or STALLED stays set: /goal resume starts a
# new stretch. MET is final. CLEARED removes the goal.
ACTIVE = "active"
PAUSED = "paused"
MET = "met"
BLOCKED = "blocked"
IMPOSSIBLE = "impossible"
CAPPED = "capped"
STALLED = "stalled"
CLEARED = "cleared"
STOPPED = frozenset({PAUSED, BLOCKED, IMPOSSIBLE, CAPPED, STALLED})

# How a round's turn ended.
COMPLETED = "completed"    # the model replied without a tool call
OUT_OF_STEPS = "max_steps"  # the turn's step budget ran out
INTERRUPTED = "interrupted"
FAILED = "failed"          # any other error

# Judge modes and answers.
CHECK = "check"      # no claim: true | false
VERDICT = "verdict"  # a "complete" claim and no verify command: met | not yet met | impossible
NOT_YET = "not yet met"


@dataclass
class Round:
    number: int
    source: str  # "goal" (opened by the goal loop's prompt) | "user" (by the user's message)
    start_seq: int
    end: str | None = None  # None while the round's turn is running
    error: str = ""
    work: list[str] = field(default_factory=list)  # the round's tool calls (goal tool excluded)
    claim: dict[str, Any] | None = None   # the last GOAL_CLAIM payload in the round
    judge: dict[str, Any] | None = None   # its GOAL_JUDGE payload
    verify: dict[str, Any] | None = None  # its GOAL_VERIFY payload

    @property
    def idle(self) -> bool:
        return not self.work


@dataclass
class Goal:
    condition: str
    cap: int
    verify: str | None
    mode: str  # "goal" | "guided"
    fields: dict[str, str]
    set_seq: int
    verify_timeout: float = DEFAULT_VERIFY_TIMEOUT  # seconds
    judge_model: str | None = None  # None: the judge runs on the chat's own model
    status: str = ACTIVE
    reason: str = ""  # why it stopped (or was resumed)
    rounds: list[Round] = field(default_factory=list)  # the current stretch's rounds

    @classmethod
    def from_set(cls, p: dict[str, Any], seq: int = 0) -> "Goal":
        """The goal that a GOAL_SET payload describes."""
        return cls(condition=p["condition"], cap=int(p.get("cap") or DEFAULT_CAP),
                   verify=(p.get("verify") or None), mode=p.get("mode") or "goal",
                   fields=dict(p.get("fields") or {}), set_seq=seq,
                   verify_timeout=(p.get("verify_timeout") or DEFAULT_VERIFY_TIMEOUT),
                   judge_model=(p.get("judge_model") or None))

    @property
    def round(self) -> int:
        """Rounds started in the current stretch."""
        return len(self.rounds)

    @property
    def active(self) -> bool:
        return self.status == ACTIVE

    @property
    def last_reason(self) -> str:
        """The most recent reason to show the user: why the goal stopped, else what
        the latest check said."""
        if self.status != ACTIVE and self.reason:
            return self.reason
        for r in reversed(self.rounds):
            if r.verify is not None and r.verify.get("exit_code") != 0:
                return _verify_reason(self, r.verify)
            if r.judge is not None:
                return r.judge.get("reason") or ""
        return ""


@dataclass(frozen=True)
class Decision:
    """What the goal loop should do next.

    action: "idle"   — no active goal; nothing to do
            "wait"   — the current round's turn is still running
            "round"  — start round `round`, opened by `message`
            "judge"  — ask the judge about round `round`, in mode `mode`
            "verify" — run the verify command for round `round`
            "stop"   — log GOAL_STATUS `status` with `reason`
    """
    action: str
    round: int = 0
    message: str = ""
    mode: str = ""
    status: str = ""
    reason: str = ""


# --- setting a goal ----------------------------------------------------------


def goal_set_payload(condition: Any, *, cap: Any = None, verify: Any = None,
                     verify_timeout: Any = None, judge_model: Any = None,
                     mode: Any = "goal", fields: Any = None) -> dict[str, Any]:
    """The GOAL_SET payload for a new goal, from a request's values. Raises
    ValueError, with a message for the user, when they can't make a goal."""
    condition = condition.strip() if isinstance(condition, str) else ""
    if not condition:
        raise ValueError("a goal needs a condition")
    cap = DEFAULT_CAP if cap is None else cap
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("the round cap must be a whole number, at least 1")
    if verify is not None and not isinstance(verify, str):
        raise ValueError("the verify command must be a string")
    verify = (verify or "").strip() or None
    verify_timeout = DEFAULT_VERIFY_TIMEOUT if verify_timeout is None else verify_timeout
    if (isinstance(verify_timeout, bool) or not isinstance(verify_timeout, (int, float))
            or verify_timeout <= 0):
        raise ValueError("the verify timeout must be a positive number of seconds")
    if judge_model is not None and not isinstance(judge_model, str):
        raise ValueError("the judge model must be a string")
    if mode not in MODES:
        raise ValueError(f"the goal mode must be one of {', '.join(MODES)}")
    if fields is not None and not isinstance(fields, dict):
        raise ValueError("the goal's fields must be an object of names to text")
    return {"condition": condition, "cap": cap, "verify": verify,
            "verify_timeout": verify_timeout,
            "judge_model": (judge_model or "").strip() or None,
            "mode": mode, "fields": {str(k): str(v) for k, v in (fields or {}).items()}}


# --- reading the log -------------------------------------------------------


def _round_end(error: str) -> str:
    if error == "max_steps exceeded":
        return OUT_OF_STEPS
    if "interrupted" in error:
        return INTERRUPTED
    return FAILED


def goal_state(events: Iterable) -> Goal | None:
    """The run's current goal, rebuilt from its events; None if there is none
    (never set, or cleared)."""
    goal: Goal | None = None
    current: Round | None = None  # the round that events are attributed to
    for ev in events:
        t, p = ev.type, ev.payload
        if t == GOAL_SET:
            goal = Goal.from_set(p, ev.seq)
            current = None
        elif goal is None:
            continue
        elif t == GOAL_STATUS:
            goal.status = p["status"]
            goal.reason = p.get("reason") or ""
            if goal.status == ACTIVE:  # resumed: a new stretch, counted from 0
                goal.rounds = []
            current = None  # a turn after a stop isn't a goal round
        elif t == GOAL_ROUND:
            current = Round(number=int(p["round"]), source=p.get("source") or "goal",
                            start_seq=ev.seq)
            goal.rounds.append(current)
        elif current is None:
            continue
        elif t == TOOL_CALL:
            if p.get("name") != GOAL_TOOL_NAME:
                current.work.append(f"{p.get('name')}:{p.get('arguments') or ''}")
        elif t == GOAL_CLAIM:
            current.claim = p
        elif t == RUN_COMPLETED:
            if current.end is None:
                current.end = COMPLETED
        elif t == RUN_FAILED:
            if current.end is None:
                current.error = p.get("error") or ""
                current.end = _round_end(current.error)
        elif t == GOAL_JUDGE:
            current.judge = p
        elif t == GOAL_VERIFY:
            current.verify = p
    if goal is not None and goal.status == CLEARED:
        return None
    return goal


def _trailing(rounds: list[Round], pred: Callable[[int, Round], bool]) -> int:
    n = 0
    for i in range(len(rounds) - 1, -1, -1):
        if not pred(i, rounds[i]):
            break
        n += 1
    return n


def _no_new_work(rounds: list[Round]) -> int:
    """Rounds in a row, ending with the latest, that made no tool calls or
    repeated the round before's calls exactly."""
    return _trailing(rounds, lambda i, r: r.idle or (i > 0 and r.work == rounds[i - 1].work))


# --- deciding --------------------------------------------------------------


def decide(events: Iterable) -> Decision:
    """The goal loop's next step, as a pure function of the run's events."""
    goal = goal_state(events)
    if goal is None or not goal.active:
        return Decision("idle")
    if not goal.rounds:
        return Decision("round", round=1, message=kickoff_prompt(goal))
    r = goal.rounds[-1]
    if r.end is None:
        return Decision("wait", round=r.number)
    if r.end == INTERRUPTED:
        return _stop(PAUSED, "the round was interrupted", r)
    if r.end == FAILED:
        failed = _trailing(goal.rounds, lambda i, x: x.end == FAILED)
        if failed >= RETRY_ROUNDS:
            return _stop(PAUSED, f"{failed} rounds in a row failed; the last error was: "
                                 f"{r.error}", r)
        if r.number >= goal.cap:
            return _stop(CAPPED, f"all {goal.cap} rounds were used", r)
        return Decision("round", round=r.number + 1,
                        message=retry_prompt(goal, r.number + 1, r.error))

    # The round's turn ended normally, or ran out of steps: check it.
    claim = r.claim or {}
    if claim.get("kind") == "blocked":
        return _stop(BLOCKED, claim.get("summary") or "the agent reported it is blocked", r)
    if claim.get("kind") == "complete":
        if goal.verify:
            return _after_verify(goal, r)
        if r.judge is None:
            return Decision("judge", round=r.number, mode=VERDICT)
        answer, reason = r.judge.get("answer"), r.judge.get("reason") or ""
        if answer == MET:
            return _stop(MET, reason or "the judge agreed the goal is met", r)
        if answer == IMPOSSIBLE:
            return _stop(IMPOSSIBLE, reason or "the judge found the goal impossible", r)
        return _not_yet(goal, r, reason)
    if r.judge is None:
        return Decision("judge", round=r.number, mode=CHECK)
    if r.judge.get("answer") == "true":
        if goal.verify:
            return _after_verify(goal, r)
        return _stop(MET, "the judge answered true", r)
    return _not_yet(goal, r, "")


def _stop(status: str, reason: str, r: Round) -> Decision:
    return Decision("stop", round=r.number, status=status, reason=reason)


def _verify_reason(goal: Goal, v: dict) -> str:
    out = (v.get("output") or "").strip()
    tail = "\n".join(out.splitlines()[-20:])[-2000:]
    head = f"`{v.get('command') or goal.verify}` exited {v.get('exit_code')}"
    return f"{head}:\n{tail}" if tail else head


def _after_verify(goal: Goal, r: Round) -> Decision:
    if r.verify is None:
        return Decision("verify", round=r.number)
    code = r.verify.get("exit_code")
    if code == 0:
        return _stop(MET, _verify_reason(goal, r.verify), r)
    if code == 1:
        return _not_yet(goal, r, _verify_reason(goal, r.verify))
    return _stop(PAUSED, "the verify command is broken: " + _verify_reason(goal, r.verify), r)


def _not_yet(goal: Goal, r: Round, reason: str) -> Decision:
    if r.number >= goal.cap:
        return _stop(CAPPED, f"all {goal.cap} rounds were used without meeting the goal", r)
    stalled = _no_new_work(goal.rounds)
    if stalled >= STALL_ROUNDS:
        return _stop(STALLED, f"{stalled} rounds in a row made no new tool calls", r)
    return Decision("round", round=r.number + 1, reason=reason,
                    message=continue_prompt(goal, r.number + 1, reason))


# --- prompts (model-visible; deterministic) ---------------------------------


def _fields_block(goal: Goal) -> str:
    if not goal.fields:
        return ""
    return "\n" + "\n".join(f"{k}: {v}" for k, v in goal.fields.items()) + "\n"


def kickoff_prompt(goal: Goal) -> str:
    verify = (f"\nThe goal counts as met when `{goal.verify}` exits with code 0.\n"
              if goal.verify else "")
    return (
        f"Goal: {goal.condition}\n{_fields_block(goal)}{verify}\n"
        "Work toward this goal until it is met. You work in rounds: after each of "
        "your replies, the harness checks whether the goal is met, and if it isn't, "
        f"you get another round, up to {goal.cap} rounds. Each round has its own "
        "budget of model calls.\n\n"
        f"When you believe the goal is met, call the `{GOAL_TOOL_NAME}` tool with "
        "status \"complete\" and a summary of the evidence. If the goal can't be met "
        "(it needs access you don't have, a decision from the user, or it contradicts "
        "itself), call it with status \"blocked\" and the reason."
    )


def continue_prompt(goal: Goal, n: int, reason: str) -> str:
    feedback = f"Feedback from the check:\n{reason}\n\n" if reason else ""
    return (
        f"The goal isn't met yet. This is round {n} of {goal.cap}.\n\n{feedback}"
        f"Keep working toward the goal: {goal.condition}\n"
        f"Call the `{GOAL_TOOL_NAME}` tool with status \"complete\" when it is met, "
        "or \"blocked\" if it can't be."
    )


def retry_prompt(goal: Goal, n: int, error: str) -> str:
    return (
        f"The last round stopped with an error: {error}\n"
        f"This is round {n} of {goal.cap}. Keep working toward the goal: {goal.condition}"
    )


# Agent.final_step_note while a goal round runs (agent/loop.py).
GOAL_FINAL_STEP_NOTE = (
    "This is the last model call of this round: the round's step budget runs out "
    "after this reply. Do not call tools. Reply now with what you have done toward "
    "the goal and what is left. Running out of steps doesn't mean the goal is met or "
    "abandoned: the goal stays active, and the next round continues from here."
)


# --- the judge ---------------------------------------------------------------

JUDGE_SYSTEM = (
    "You check whether a goal has been met, using only the conversation you are "
    "shown. You don't run commands or read files. Tool results in the conversation "
    "count as evidence; the agent's claims without evidence don't."
)


def check_grammar() -> Grammar:
    return Grammar.from_rules({"answer": '"true" | "false"'}, root="answer")


def verdict_grammar() -> Grammar:
    return Grammar.from_rules(
        {"verdict": 'status " - " reason',
         "status": '"met" | "not yet met" | "impossible"',
         "reason": r"/[^\n]+/"},
        root="verdict",
    )


def grammar_params(grammar: Grammar, server_grammar: str | None) -> dict[str, Any]:
    """Request params that constrain the judge's answer, for a server whose
    grammar support is `server_grammar` (Capabilities.grammar). Thinking is
    switched off with it, as in logits/grammar_stage.py: a reasoning template
    would otherwise spend the grammar on the thinking block. No support → {}."""
    if server_grammar == "gbnf":
        key = "grammar"
    elif server_grammar == "guided":
        key = "guided_grammar"
    else:
        return {}
    return {key: grammar.to_gbnf(), "chat_template_kwargs": {"enable_thinking": False}}


def parse_check(text: str) -> tuple[str, bool]:
    """("true" | "false", parsed). Anything that doesn't start with true or false
    counts as false."""
    m = re.match(r"^\W*(true|false)\b", (text or "").strip().lower())
    return (m.group(1), True) if m else ("false", False)


def parse_verdict(text: str) -> tuple[str, str, bool]:
    """(answer, reason, parsed), answer one of met | not yet met | impossible.
    An answer that can't be read counts as not yet met."""
    m = re.match(r"^\W*(met|not yet met|impossible)\b\s*[-:–—]?\s*(.*)",
                 (text or "").strip(), flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return NOT_YET, "", False
    return m.group(1).lower(), m.group(2).strip(), True


def _text(content: Any) -> str:
    if isinstance(content, list):  # multimodal parts
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


def judge_transcript(events: Iterable, max_chars: int = 60_000,
                     result_chars: int = 2000) -> str:
    """The conversation as the agent last saw it, as plain text: the latest
    MODEL_CALL's request messages (so a compacted context stays compacted), its
    reply, and any tool results logged after it. The system prompt is left out.
    Older text is dropped first when it is longer than `max_chars`."""
    events = list(events)
    last = next((i for i in range(len(events) - 1, -1, -1)
                 if events[i].type == MODEL_CALL), None)
    if last is None:
        return ""
    call = events[last].payload
    messages = list(call["request_body"].get("messages") or [])
    messages.append(call["response"]["choices"][0]["message"])
    for ev in events[last + 1:]:
        if ev.type == TOOL_CALL:
            messages.append({"role": "tool", "name": ev.payload.get("name"),
                             "content": ev.payload.get("result")})
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        text = _text(m.get("content")).strip()
        if role == "tool":
            if len(text) > result_chars:
                text = text[:result_chars] + " [cut]"
            lines.append(f"[tool result: {m.get('name') or 'tool'}]\n{text}")
        elif role == "assistant":
            calls = [f"[called {tc['function']['name']}({tc['function'].get('arguments') or ''})]"
                     for tc in (m.get("tool_calls") or [])]
            lines.append("[assistant]\n" + "\n".join(x for x in [text, *calls] if x))
        else:
            lines.append(f"[{role}]\n{text}")
    out = "\n\n".join(lines)
    if len(out) > max_chars:
        out = "[earlier conversation left out]\n\n" + out[-max_chars:]
    return out


def judge_seed(base_seed: int, round_number: int) -> int:
    return base_seed + JUDGE_SEED_OFFSET + round_number


def judge_request(goal: Goal, round_number: int, mode: str, events: Iterable, *,
                  model: str, base_seed: int = 1, server_grammar: str | None = None,
                  claim_summary: str = "") -> dict[str, Any]:
    """The judge's whole chat request body: a fresh context with no tools, on
    `model` (the chat's own model unless a judge model is configured)."""
    transcript = judge_transcript(events)
    if mode == VERDICT:
        claim = f"The agent says the goal is complete: {claim_summary}\n\n" if claim_summary else ""
        question = (
            "Is the goal met? Answer on one line, in one of these forms:\n"
            "met - <the evidence>\n"
            "not yet met - <what is still missing>\n"
            "impossible - <why it can't be met>"
        )
        grammar, max_tokens = verdict_grammar(), 400
    else:
        claim = ""
        question = ("Is the goal complete based on the work so far? "
                    "Answer with one word: true or false.")
        grammar, max_tokens = check_grammar(), 8
    user = (f"Goal: {goal.condition}\n{_fields_block(goal)}\n{claim}"
            f"The conversation so far:\n\n{transcript}\n\n{question}")
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "seed": judge_seed(base_seed, round_number),
    }
    params = grammar_params(grammar, server_grammar)
    if params:  # constrained: the answer is short, so cap it
        body["max_tokens"] = max_tokens
        body.update(params)
    return body


def judge_payload(round_number: int, mode: str, body: dict[str, Any],
                  response: dict[str, Any], timing_ms: float) -> dict[str, Any]:
    """The GOAL_JUDGE payload for a judge call and its raw response."""
    text = _text(response["choices"][0]["message"].get("content"))
    if mode == VERDICT:
        answer, reason, parsed = parse_verdict(text)
    else:
        (answer, parsed), reason = parse_check(text), ""
    return {"round": round_number, "mode": mode, "seed": body.get("seed"),
            "request_body": body, "response": response, "timing_ms": timing_ms,
            "answer": answer, "reason": reason, "parsed": parsed}


# --- the goal tool -----------------------------------------------------------


def goal_tool(log, run_id: str, round_number: int) -> Tool:
    """The `goal` tool, bound to one run and round. It exists only while a goal
    round runs. It logs a GOAL_CLAIM itself, so a call made from inside code mode's
    run_code (which logs only the run_code call) still reaches decide()."""
    def goal(status: str, summary: str = "") -> str:
        status = (status or "").strip().lower()
        if status not in ("complete", "blocked"):
            return 'error: status must be "complete" or "blocked"'
        log.append(run_id, GOAL_CLAIM,
                   {"round": round_number, "kind": status, "summary": summary or ""})
        if status == "complete":
            return ("Recorded: the goal will be checked when this reply ends. "
                    "Finish your reply now.")
        return ("Recorded: the goal will stop and the user will see your reason. "
                "Finish your reply now.")

    return Tool(
        name=GOAL_TOOL_NAME,
        description=(
            "Report on the active goal. Call it with status \"complete\" when you "
            "believe the goal is met, with the evidence as the summary; the goal is "
            "checked after your reply. Call it with status \"blocked\" when the goal "
            "can't be met without the user, with what is blocking it as the summary."),
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["complete", "blocked"]},
                "summary": {"type": "string",
                            "description": "the evidence (complete) or the blocker (blocked)"},
            },
            "required": ["status", "summary"],
        },
        fn=goal,
    )

