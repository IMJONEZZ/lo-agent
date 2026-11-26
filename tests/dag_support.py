"""Composability DAGs: the pool from the LO2 test plan, run on the mock.

The plan (docs/software-testing/LO2-test-plan.md) treats a 3-step round as the
smallest meaningful unit, one lo feature per step, and its criterion 2.2 names
the pool: 2 system prompts × 3 ways to stop × 2 tools = 12 DAGs.

  step 1, system prompt: with or without the project's AGENTS.md
  step 2, tool calling:  memory or session_search, called natively
  step 3, stop:          a normal finish, the step budget used up, or a guardrail
                         stop (the doom-loop cap on identical calls)

Driver: the journey driver (the TUI under Textual's pilot, over the real session
server). Stub: MockLlamaCpp, scripted per DAG as a pure function of the request,
so every run can be replayed. Oracle: the plan's break list (a crash, an error
the user doesn't see, an incorrect output, an unfulfilled promise, not returning
to ready) on top of the journeys' UX floor. Each run is tagged (RUN_TAG) with
its DAG and the option it used at each step, so coverage is counted from the
event log, not from the test code.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from local_harness.events.log import (
    GUARDRAIL,
    MODEL_CALL,
    RUN_COMPLETED,
    RUN_FAILED,
    TOOL_CALL,
    USER_MESSAGE,
    EventLog,
)
from local_harness.sim.scenario import Scenario, Step

from e2e_support import _skip_probe, agent_bodies
from mocks import MockLlamaCpp, chat_response

SUITE = "composability"

# The three steps and their options, in the plan's order.
SYSTEM_PROMPTS = ("agents-md", "no-agents-md")
TOOLS = ("memory", "session_search")
STOPS = ("finish", "step-budget", "guardrail")
DIMENSIONS = {"system_prompt": SYSTEM_PROMPTS, "tool": TOOLS, "stop": STOPS}

MAX_STEPS = 8  # e2e_support.start_server's step budget
LOOP_CAP = 6  # Guardrails' default doom-loop hard cap (identical calls)
LOOP_NUDGE_AT = 3  # ...and the identical call that gets a nudge instead of running

ENTER = "\r"
# What the model is told to construct, so the marker never matches the echoed
# prompt (the journeys' technique; see local_harness/sim/scenarios.py). The
# prompts are short because the driver types them a key at a time (~0.1 s a key).
FINISH_MARKER = "ROUNDDONE"
READY_MARKER = "READYAGAIN"
FOLLOW_UP = "Ready? Say READY+AGAIN."
AGENTS_RULE = "DAG-RULE: always name the file you used."
NOTE = "launch list: docs/launch.md"
SEARCH_WORD = "zephyrine"  # a rare word, seeded in a past session for session_search
SEEDED_SESSION = f"Planned the {SEARCH_WORD} launch: the checklist lives in docs/launch.md."

TASKS = {
    "memory": f"Remember '{NOTE}', then say ROUND+DONE.",
    "session_search": f"Find {SEARCH_WORD} in past sessions, then say ROUND+DONE.",
}
# What the user sees when the round ends: the answer, or the run's error line.
TURN_END_MARKER = {"finish": FINISH_MARKER, "step-budget": "max_steps exceeded",
                   "guardrail": "doom loop"}


@dataclass(frozen=True)
class Dag:
    system_prompt: str
    tool: str
    stop: str

    @property
    def id(self) -> str:
        return f"{self.system_prompt}.{self.tool}.{self.stop}"

    def tags(self) -> dict[str, str]:
        return {"suite": SUITE, "dag": self.id, "system_prompt": self.system_prompt,
                "tool": self.tool, "stop": self.stop}

    # What the round should look like, step by step.
    @property
    def model_calls(self) -> int:
        return {"finish": 2, "step-budget": MAX_STEPS, "guardrail": LOOP_CAP}[self.stop]

    @property
    def tool_runs(self) -> int:
        # The doom-loop nudge replaces the third identical call; the cap stops the sixth.
        return {"finish": 1, "step-budget": MAX_STEPS, "guardrail": LOOP_CAP - 2}[self.stop]


def generate_pool() -> list[Dag]:
    """Every combination of the three steps' options (3-wise = all 12 here)."""
    return [Dag(s, t, x) for s, t, x in itertools.product(SYSTEM_PROMPTS, TOOLS, STOPS)]


POOL = generate_pool()


# --- the stub: the model's side of each DAG ----------------------------------


def _tool_args(dag: Dag, n: int) -> str:
    """The n-th tool call of the round (n from 1). The step-budget DAGs vary the
    arguments every call, so the doom-loop guardrail never fires on them; the
    others repeat the first call exactly."""
    vary = dag.stop == "step-budget" and n > 1
    if dag.tool == "memory":
        return json.dumps({"action": "add", "text": f"{NOTE} (step {n})" if vary else NOTE})
    return json.dumps({"query": f"{SEARCH_WORD} step {n}" if vary else SEARCH_WORD})


def dag_chat_fn(dag: Dag):
    """The scripted model for one DAG: a pure function of the request, so a replay
    of the run's logged requests gets the same answers."""

    def respond(body: dict) -> dict:
        messages = body.get("messages") or []
        if any(m.get("role") == "user" and FOLLOW_UP in str(m.get("content") or "")
               for m in messages):
            return chat_response(content=READY_MARKER)
        n = 1 + sum(1 for m in messages if m.get("role") == "assistant")
        if dag.stop == "finish" and n > 1:
            return chat_response(content=FINISH_MARKER)
        return chat_response(tool_calls=[(f"call-{n}", dag.tool, _tool_args(dag, n))])

    return _skip_probe(respond)


def dag_mock(dag: Dag) -> MockLlamaCpp:
    return MockLlamaCpp(chat_fn=dag_chat_fn(dag))


def prepare_workspace(dag: Dag, workspace: Path, mem_dir: Path) -> None:
    """Step 1's option (AGENTS.md or not) and step 2's data (a past session for
    session_search to find)."""
    if dag.system_prompt == "agents-md":
        (workspace / "AGENTS.md").write_text(f"# Rules\n\n{AGENTS_RULE}\n")
    if dag.tool == "session_search":
        from local_harness.agent.memory import Memory

        mem_dir.mkdir(parents=True, exist_ok=True)
        Memory(mem_dir / "memory.db").store("session", SEEDED_SESSION)


def dag_scenario(dag: Dag) -> Scenario:
    return Scenario(
        f"dag-{dag.id}",
        description=f"Composability DAG {dag.id}: system prompt → tool call → stop, "
                    "then the session takes the next message.",
        needs={"mock-only"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step(TASKS[dag.tool] + ENTER, marker=TURN_END_MARKER[dag.stop], timeout=60.0,
                 label=f"round {dag.id}"),
            Step(FOLLOW_UP + ENTER, marker=READY_MARKER, timeout=30.0,
                 label="back to ready"),
        ],
    )


# --- the oracle: the break list -------------------------------------------------


def _looks_like_crash(error: str) -> bool:
    head = error.split(":", 1)[0]
    return head.endswith(("Error", "Exception")) or head in ("KeyError", "StopIteration")


def find_breaks(dag: Dag, db: str, mock: MockLlamaCpp, mem_dir: Path) -> list[str]:
    """Check one DAG run against the plan's break list; [] means it passed."""
    log = EventLog(db)
    runs = log.runs()
    if len(runs) != 1:
        return [f"expected 1 run, found {len(runs)}"]
    run = runs[0]
    events = log.events(run.run_id)
    breaks: list[str] = []

    # Turns: the round, then the follow-up. Split at the follow-up message.
    follow = next((e.seq for e in events if e.type == USER_MESSAGE
                   and FOLLOW_UP in str(e.payload.get("content"))), None)
    if follow is None:
        return ["the follow-up message never reached the run"]
    round_events = [e for e in events if e.seq < follow]
    ends = [e for e in events if e.type in (RUN_COMPLETED, RUN_FAILED)]

    # Crashes: an end event that is an exception, or the wrong number of ends.
    for e in ends:
        error = str(e.payload.get("error") or "")
        if e.type == RUN_FAILED and _looks_like_crash(error):
            breaks.append(f"crash: {error}")
    if len(ends) != 2:
        breaks.append(f"expected 2 end events (round + follow-up), found {len(ends)}")

    # Step 3, how the round stopped.
    round_end = next((e for e in round_events if e.type in (RUN_COMPLETED, RUN_FAILED)), None)
    if round_end is None:
        breaks.append("the round has no end event")
    elif dag.stop == "finish":
        if round_end.type != RUN_COMPLETED or round_end.payload.get("answer") != FINISH_MARKER:
            breaks.append(f"incorrect output: round ended {round_end.type} {round_end.payload}")
    elif dag.stop == "step-budget":
        if round_end.payload.get("error") != "max_steps exceeded":
            breaks.append(f"incorrect output: round ended {round_end.type} {round_end.payload}")
        notes = [e for e in round_events if e.type == USER_MESSAGE
                 and e.payload.get("source") == "harness"]
        if len(notes) != 1:
            breaks.append(f"expected 1 final-step note before the last call, found {len(notes)}")
    else:  # guardrail
        error = str(round_end.payload.get("error") or "")
        if not error.startswith(f"guardrails: doom loop: {dag.tool} called {LOOP_CAP} times"):
            breaks.append(f"incorrect output: round ended {round_end.type} {round_end.payload}")
        actions = [(e.payload.get("action"), e.payload.get("kind")) for e in round_events
                   if e.type == GUARDRAIL and e.payload.get("action") != "execute"]
        if actions != [("nudge", "loop"), ("fatal", None)]:
            breaks.append(f"guardrail steps were {actions}, expected a loop nudge then fatal")

    calls = [e for e in round_events if e.type == MODEL_CALL]
    if len(calls) != dag.model_calls:
        breaks.append(f"expected {dag.model_calls} model calls in the round, found {len(calls)}")

    # Step 1, the system prompt.
    bodies = agent_bodies(mock)
    if not bodies:
        return breaks + ["the model never received a request"]
    system = str((bodies[0].get("messages") or [{}])[0].get("content") or "")
    has_rule = AGENTS_RULE in system
    if has_rule != (dag.system_prompt == "agents-md"):
        breaks.append(f"incorrect output: AGENTS.md rule in the system prompt = {has_rule}")

    # Step 2, the tool call: parsed, run by its handler, result back to the model.
    runs_of_tool = [e for e in round_events if e.type == TOOL_CALL
                    and e.payload.get("name") == dag.tool]
    if len(runs_of_tool) != dag.tool_runs:
        breaks.append(f"expected {dag.tool} to run {dag.tool_runs} times, "
                      f"found {len(runs_of_tool)}")
    errors = [e.payload.get("result") for e in events if e.type == TOOL_CALL
              and str(e.payload.get("result") or "").startswith("error")]
    if errors:
        breaks.append(f"tool errors: {errors[:2]}")
    if runs_of_tool:
        first = str(runs_of_tool[0].payload.get("result") or "")
        if dag.tool == "memory":
            saved = (mem_dir / "MEMORY.md").read_text() if (mem_dir / "MEMORY.md").exists() else ""
            if first != "ok" or NOTE not in saved:
                breaks.append(f"incorrect output: memory returned {first!r}; MEMORY.md has "
                              f"the note = {NOTE in saved}")
        elif SEEDED_SESSION not in first:
            breaks.append(f"incorrect output: session_search returned {first!r}")
        # The next request must carry the result as a tool message paired with the
        # call. Finding the text anywhere in it isn't enough: a short result such
        # as "ok" also turns up elsewhere in a request (the planted-fault run of
        # 23 Nov missed exactly that).
        second = (bodies[1].get("messages") if len(bodies) > 1 else None) or []
        paired = [m for m in second if m.get("role") == "tool"
                  and m.get("tool_call_id") == "call-1"]
        if not paired or str(paired[0].get("content") or "") != first:
            breaks.append("the tool's result was not sent back to the model, "
                          "paired with its call")

    # Return to ready: the follow-up was answered and the run isn't left running.
    if not ends or ends[-1].type != RUN_COMPLETED or \
            ends[-1].payload.get("answer") != READY_MARKER:
        breaks.append("did not return to ready: the follow-up wasn't answered")
    if log.run(run.run_id).status != "completed":
        breaks.append(f"did not return to ready: run status {log.run(run.run_id).status}")
    return breaks


# --- coverage, counted from the event log -----------------------------------------


def tagged_dags(db_paths: list[str]) -> list[dict]:
    """The tags of every composability run in these logs."""
    found = []
    for db in db_paths:
        log = EventLog(db)
        for meta in log.runs():
            tags = log.run_tags(meta.run_id)
            if tags.get("suite") == SUITE:
                found.append(dict(tags, status=meta.status))
    return found


def write_report(path: str, results: list[dict], tagged: list[dict]) -> None:
    """A Markdown table of the DAG runs, and the coverage counted from their tags."""
    import platform
    import time
    from importlib.metadata import version

    order = {d: i for i, d in enumerate(POOL)}
    rows = sorted(results, key=lambda r: order.get(r["dag"], len(order)))
    passed = sum(1 for r in rows if not r["breaks"])
    lines = [
        "# Composability DAG results",
        "",
        f"Run {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · lo "
        f"{version('lo-agent')} · Python {platform.python_version()} · "
        "model: MockLlamaCpp (scripted per DAG)",
        "",
        "Command: `LO_DAG_REPORT=<this file> pytest tests/test_dag_composability.py`",
        "",
        f"{passed} of {len(rows)} DAGs passed. Each DAG is one round (system prompt → "
        "tool call → stop) followed by one more message, which checks that the session "
        "came back to ready. Replay re-sends every logged model call and compares.",
        "",
        "| DAG | system prompt | tool | stop | model calls | tool runs | round ended "
        "| follow-up answered | replay | result |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["dag"]
        result = "pass" if not r["breaks"] else "BREAK: " + "; ".join(r["breaks"])
        lines.append(f"| `{d.id}` | {d.system_prompt} | {d.tool} | {d.stop} | "
                     f"{r['model_calls']} | {r['tool_runs']} | {r['round_end']} | "
                     f"{'yes' if r['follow_up'] else 'no'} | {r['replay']} | {result} |")
    lines += ["", "## Coverage, counted from the runs' RUN_TAG events", "",
              "| criterion | covered | total | % |", "|---|---|---|---|"]
    for name, (got, total) in coverage(POOL, tagged).items():
        lines.append(f"| {name} | {got} | {total} | {100 * got / total:.0f}% |")
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def coverage(pool: list[Dag], tagged: list[dict]) -> dict[str, tuple[int, int]]:
    """How much of the pool the tagged runs cover: whole DAGs (3-wise), every
    pair of options from two different steps (2-wise), and every option (1-wise)."""
    dims = list(DIMENSIONS)
    seen = {tuple(t.get(d) for d in dims) for t in tagged}
    combos = {tuple(getattr(d, k) for k in dims) for d in pool}
    pairs_needed = {(i, j, a[i], a[j]) for a in combos for i, j in
                    itertools.combinations(range(len(dims)), 2)}
    pairs_seen = {(i, j, a[i], a[j]) for a in seen & combos for i, j in
                  itertools.combinations(range(len(dims)), 2)}
    values_needed = {(i, a[i]) for a in combos for i in range(len(dims))}
    values_seen = {(i, a[i]) for a in seen & combos for i in range(len(dims))}
    return {
        "dags (3-wise)": (len(seen & combos), len(combos)),
        "pairs (2-wise)": (len(pairs_seen), len(pairs_needed)),
        "options (1-wise)": (len(values_seen), len(values_needed)),
    }
