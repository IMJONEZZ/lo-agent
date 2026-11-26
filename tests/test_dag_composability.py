"""Composability DAGs (LO2 test plan, requirement 1) on the mock.

Each of the 12 DAGs in the pool runs through the TUI over the real session
server (the journey driver) with a scripted MockLlamaCpp, gets tagged with its
DAG and options (RUN_TAG), and is checked against the plan's break list; a
replay of its logged model calls must match (the "promise" in the break list).
The oracle is itself checked on runs with a planted break.

Set LO_DAG_REPORT=<path> to write a results table, with the coverage counted
from the runs' tags in their event logs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_harness.events.export import transcript_markdown
from local_harness.events.log import (
    MODEL_CALL,
    RUN_COMPLETED,
    RUN_FAILED,
    TOOL_CALL,
    USER_MESSAGE,
    EventLog,
)
from local_harness.events.replay import replay_run
from local_harness.inference.client import OpenAICompatClient
from local_harness.sim import run_scenario
from local_harness.sim.scenario import ScenarioFailure

from dag_support import (
    AGENTS_RULE,
    FINISH_MARKER,
    FOLLOW_UP,
    NOTE,
    POOL,
    READY_MARKER,
    TASKS,
    Dag,
    coverage,
    dag_mock,
    dag_scenario,
    find_breaks,
    prepare_workspace,
    tagged_dags,
    write_report,
)
from e2e_support import (
    UPSTREAM_MODEL,
    UPSTREAM_URL,
    PilotDriver,
    assert_ux_invariants,
    make_server_app,
    start_server,
    stop_server,
)
from mocks import MockLlamaCpp, chat_response

pytestmark = pytest.mark.e2e

_RESULTS: list[dict] = []  # one row per DAG run, for LO_DAG_REPORT


@pytest.fixture(scope="module", autouse=True)
def dag_report():
    yield
    path = os.environ.get("LO_DAG_REPORT")
    if path and _RESULTS:
        write_report(path, _RESULTS, tagged_dags([r["db"] for r in _RESULTS]))


@pytest.fixture
def dag_env(tmp_path, monkeypatch):
    """An isolated workspace (cwd) and home for one DAG; tears the server down."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no personal ~/.lo/AGENTS.md
    import local_harness.tui.app as tui_app

    monkeypatch.setattr(tui_app, "_CONFIG_PATH", str(tmp_path / "config.json"))
    servers: list = []
    yield workspace, servers
    for s in servers:
        stop_server(s)


@pytest.mark.parametrize("dag", POOL, ids=lambda d: d.id)
async def test_dag(dag: Dag, dag_env):
    workspace, servers = dag_env
    mem_dir = workspace / "mem"  # start_server's memory_dir
    prepare_workspace(dag, workspace, mem_dir)
    mock = dag_mock(dag)
    url, server, db = start_server(workspace, mock)
    servers.append(server)
    app = make_server_app(db, mock, url)

    breaks: list[str] = []
    async with app.run_test(size=(120, 42)) as pilot:
        driver = PilotDriver(app, pilot)
        app.query_one("#prompt").focus()
        try:
            await run_scenario(driver, dag_scenario(dag))
        except ScenarioFailure as e:  # not on screen: swallowed, or never got there
            breaks.append(f"on screen: step {e.step.label!r}: {e.reason}")

    log = EventLog(db)
    runs = log.runs()
    run_id = runs[0].run_id if runs else None
    if run_id is not None:
        log.tag_run(run_id, dag.tags())
        assert_ux_invariants(db)
        breaks += find_breaks(dag, db, mock, mem_dir)
        replay = await replay_run(log, run_id, OpenAICompatClient(
            UPSTREAM_URL, UPSTREAM_MODEL, transport=dag_mock(dag).transport()))
        if not replay.identical:
            breaks.append(f"unfulfilled promise: replay matched {replay.matched}/"
                          f"{replay.total} model calls")
    else:
        replay = None
        breaks.append("no run was created")
    _RESULTS.append(_row(dag, db, log, run_id, replay, breaks))
    assert not breaks, "\n".join(breaks) + "\n\n" + (
        transcript_markdown(log, run_id, full=True) if run_id else "")


def _row(dag: Dag, db: str, log: EventLog, run_id, replay, breaks: list[str]) -> dict:
    events = log.events(run_id) if run_id else []
    ends = [e for e in events if e.type in (RUN_COMPLETED, RUN_FAILED)]
    follow = next((e.seq for e in events if e.type == USER_MESSAGE
                   and FOLLOW_UP in str(e.payload.get("content"))), len(events) + 1)
    round_end = ends[0] if ends else None
    return {
        "dag": dag, "db": db,
        "model_calls": sum(1 for e in events if e.type == MODEL_CALL and e.seq < follow),
        "tool_runs": sum(1 for e in events if e.type == TOOL_CALL and e.seq < follow),
        "round_end": ("none" if round_end is None else
                      "completed" if round_end.type == RUN_COMPLETED else
                      f"failed: {round_end.payload.get('error')}"),
        "follow_up": bool(ends) and ends[-1].payload.get("answer") == READY_MARKER,
        "replay": f"{replay.matched}/{replay.total}" if replay else "-",
        "breaks": list(breaks),
    }


# --- the pool and the coverage count -------------------------------------------


def test_pool_is_every_combination_of_the_three_steps():
    assert len(POOL) == len(set(POOL)) == 12
    full = coverage(POOL, [d.tags() for d in POOL])
    assert full == {"dags (3-wise)": (12, 12), "pairs (2-wise)": (16, 16),
                    "options (1-wise)": (7, 7)}


def test_coverage_is_counted_from_the_tags_in_the_log(tmp_path):
    db = str(tmp_path / "c.db")
    log = EventLog(db)
    for dag in POOL[:3]:  # agents-md + memory, with each of the three stops
        log.tag_run(log.create_run(dag.id), dag.tags())
    log.tag_run(log.create_run("not a DAG"), {"suite": "something-else", "tool": "grep"})
    log.create_run("untagged")
    cov = coverage(POOL, tagged_dags([db]))
    assert cov["dags (3-wise)"] == (3, 12)
    assert cov["options (1-wise)"] == (5, 7)  # agents-md, memory, and all three stops
    assert cov["pairs (2-wise)"] == (7, 16)  # 1 prompt×tool + 3 prompt×stop + 3 tool×stop


# --- the oracle catches planted breaks ------------------------------------------

GOOD = Dag("agents-md", "memory", "finish")


def _synthetic_run(tmp_path: Path, *, round_error: str | None = None,
                   answer_follow_up: bool = True, rule_in_prompt: bool = True,
                   result_sent_back: bool = True, result_elsewhere: bool = False):
    """A GOOD run written straight into a log, with one break planted on request."""
    db = str(tmp_path / "s.db")
    log = EventLog(db)
    rid = log.create_run(TASKS["memory"])
    call = chat_response(tool_calls=[("call-1", "memory",
                                      json.dumps({"action": "add", "text": NOTE}))])
    log.append(rid, MODEL_CALL, {"response": call})
    log.append(rid, TOOL_CALL, {"name": "memory", "arguments": "{}", "result": "ok"})
    log.append(rid, MODEL_CALL, {"response": chat_response(content=FINISH_MARKER)})
    if round_error:
        log.append(rid, RUN_FAILED, {"error": round_error})
    else:
        log.append(rid, RUN_COMPLETED, {"answer": FINISH_MARKER})
    log.append(rid, USER_MESSAGE, {"content": FOLLOW_UP, "source": "user"})
    log.append(rid, MODEL_CALL, {"response": chat_response(content=READY_MARKER)})
    if answer_follow_up:
        log.append(rid, RUN_COMPLETED, {"answer": READY_MARKER})
    mem_dir = tmp_path / "mem"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text(NOTE)
    system = {"role": "system", "content": ("rules: " + AGENTS_RULE) if rule_in_prompt else "x"}
    if result_elsewhere:  # the result's text appears in the request, but not as its result
        system["content"] += " Look it up in the book."
    task = {"role": "user", "content": TASKS["memory"]}
    back = [{"role": "assistant", "content": None}] + (
        [{"role": "tool", "content": "ok", "tool_call_id": "call-1"}] if result_sent_back
        else [])
    mock = MockLlamaCpp()
    mock.chat_bodies = [{"messages": [system, task]}, {"messages": [system, task, *back]}]
    return db, mock, mem_dir


def test_the_oracle_passes_a_good_run(tmp_path):
    assert find_breaks(GOOD, *_synthetic_run(tmp_path)) == []


@pytest.mark.parametrize("plant, expected", [
    ({"round_error": "TypeError: 'NoneType' object is not callable"}, "crash: TypeError"),
    ({"answer_follow_up": False}, "did not return to ready"),
    ({"rule_in_prompt": False}, "AGENTS.md rule in the system prompt = False"),
    ({"result_sent_back": False}, "result was not sent back to the model"),
    # the oracle's own gap, found by planting a fault in lo (23 Nov): a short
    # result ("ok") also matched text elsewhere in the request
    ({"result_sent_back": False, "result_elsewhere": True},
     "result was not sent back to the model"),
])
def test_the_oracle_catches_a_planted_break(tmp_path, plant, expected):
    breaks = find_breaks(GOOD, *_synthetic_run(tmp_path, **plant))
    assert any(expected in b for b in breaks), breaks
