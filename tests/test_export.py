"""The transcript export's full view (`lo export --full`) and run tags.

The default view is the readable summary the TUI's /export writes; the full view
is the instrumentation the LO2 test plan asks for: the whole nested conversation
history, with reasoning, tool results in full, how each turn ended, and every
sub-agent run. Run tags (RUN_TAG) label a run so runs can be counted from the log.
"""

from __future__ import annotations

import pytest

from local_harness.cli import main as cli
from local_harness.events.export import MAX_NESTING, TOOL_RESULT_CHARS, transcript_markdown
from local_harness.events.log import (
    AGENT_SPAWNED,
    CONTEXT_COMPACTED,
    GOAL_JUDGE,
    GOAL_ROUND,
    GOAL_STATUS,
    GOAL_VERIFY,
    GUARDRAIL,
    MODEL_CALL,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_TAG,
    TOOL_CALL,
    USER_MESSAGE,
    EventLog,
)

from mocks import chat_response

LONG_RESULT = "".join(f"line {i:04d}\n" for i in range(600))  # ~6000 chars, past the cut


def _call(log, rid, *, content=None, reasoning=None, field="reasoning_content",
          tool_calls=None):
    resp = chat_response(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        resp["choices"][0]["message"][field] = reasoning
    log.append(rid, MODEL_CALL, {"call_index": 0, "seed": 1, "response": resp})


def _seed(db) -> tuple[EventLog, str]:
    """A run with reasoning, a long tool result, the harness's final-step note,
    a guardrail nudge and a failed end."""
    log = EventLog(db)
    rid = log.create_run("check the build")
    _call(log, rid, reasoning="I should list the files first.",
          tool_calls=[("c1", "list_dir", '{"path": "."}')])
    log.append(rid, TOOL_CALL, {"name": "list_dir", "arguments": '{"path": "."}',
                                "result": LONG_RESULT})
    log.append(rid, GUARDRAIL, {"call_index": 1, "action": "nudge", "kind": "loop",
                                "rescued": False, "reason": None,
                                "nudge": {"role": "tool", "content": "You already ran that.",
                                          "tool_call_id": "c1"}})
    log.append(rid, USER_MESSAGE, {"content": "This is your FINAL turn.", "source": "harness"})
    _call(log, rid, content="Done listing.", reasoning="Wrap up.", field="reasoning")
    log.append(rid, RUN_FAILED, {"error": "max_steps exceeded"})
    return log, rid


def test_default_view_is_unchanged(tmp_path):
    log, rid = _seed(str(tmp_path / "e.db"))
    md = transcript_markdown(log, rid)
    assert md.startswith("# check the build")
    assert "I should list the files first." not in md  # no reasoning
    assert "line 0599" not in md and "line 0001" in md  # the result is cut
    assert len(LONG_RESULT) > TOOL_RESULT_CHARS
    assert "max_steps exceeded" not in md  # the default view doesn't show run ends


def test_full_view_keeps_reasoning_results_and_the_end(tmp_path):
    log, rid = _seed(str(tmp_path / "e.db"))
    md = transcript_markdown(log, rid, full=True)
    assert md.startswith("# check the build\n")
    assert "status: failed" in md
    # reasoning from both field names reasoning parsers use
    assert "> I should list the files first." in md
    assert "> Wrap up." in md
    assert LONG_RESULT.strip() in md  # the whole result, not the first 2000 chars
    assert "## User (harness)" in md and "This is your FINAL turn." in md
    assert "guardrail · nudge (loop)" in md and "> You already ran that." in md
    assert "**turn ended: failed** — max_steps exceeded" in md


def test_full_view_follows_sub_agents_nested_and_guards_cycles(tmp_path):
    log = EventLog(str(tmp_path / "e.db"))
    lead = log.create_run("lead task")
    worker = log.create_run("worker task")
    grandchild = log.create_run("grandchild task")
    log.append(lead, AGENT_SPAWNED, {"child_run_id": worker, "task": "worker task"})
    log.append(lead, AGENT_SPAWNED, {"child_run_id": "gone", "task": "deleted"})
    _call(log, lead, content="LEAD-ANSWER")
    log.append(lead, RUN_COMPLETED, {"answer": "LEAD-ANSWER"})
    _call(log, worker, content="WORKER-ANSWER")
    log.append(worker, AGENT_SPAWNED, {"child_run_id": grandchild, "task": "grandchild task"})
    log.append(worker, AGENT_SPAWNED, {"child_run_id": lead, "task": "a cycle"})
    log.append(worker, RUN_COMPLETED, {"answer": "WORKER-ANSWER"})
    _call(log, grandchild, content="GRANDCHILD-ANSWER")
    log.append(grandchild, RUN_COMPLETED, {"answer": "GRANDCHILD-ANSWER"})

    md = transcript_markdown(log, lead, full=True)
    # each level one heading deeper, in the order the runs were spawned
    assert "\n## sub-agent · worker task\n" in md
    assert "\n### Assistant\n\nWORKER-ANSWER" in md
    assert "\n### sub-agent · grandchild task\n" in md
    assert "\n#### Assistant\n\nGRANDCHILD-ANSWER" in md
    assert md.index("WORKER-ANSWER") < md.index("GRANDCHILD-ANSWER") < md.index("LEAD-ANSWER")
    assert f"sub-agent run `{lead}` is shown above" in md  # the cycle stops
    assert "sub-agent run `gone` isn't in this log" in md
    assert md.count("WORKER-ANSWER") == 1
    # the default view doesn't follow sub-agents
    assert "WORKER-ANSWER" not in transcript_markdown(log, lead)


def test_full_view_fences_survive_backticks_in_results(tmp_path):
    log = EventLog(str(tmp_path / "e.db"))
    rid = log.create_run("t")
    result = "before\n```\ninside\n```\nafter"
    log.append(rid, TOOL_CALL, {"name": "read_file", "arguments": "{}", "result": result})
    md = transcript_markdown(log, rid, full=True)
    assert f"\n````\n{result}\n````\n" in md


def test_run_tags_merge_in_log_order_and_show_in_the_full_view(tmp_path):
    log = EventLog(str(tmp_path / "e.db"))
    rid = log.create_run("t")
    log.tag_run(rid, {"dag": "agents-memory-finish", "tool": "memory"})
    log.tag_run(rid, {"tool": "session_search", "round": 2})
    assert log.run_tags(rid) == {"dag": "agents-memory-finish", "tool": "session_search",
                                 "round": 2}
    assert [e.type for e in log.events(rid)].count(RUN_TAG) == 2
    assert log.run(rid).status == "running"  # a tag isn't an end event
    assert log.run_tags(log.create_run("untagged")) == {}
    with pytest.raises(ValueError):
        log.tag_run(rid, {})
    md = transcript_markdown(log, rid, full=True)
    assert "tags: `dag=agents-memory-finish`, `tool=session_search`, `round=2`" in md


def test_cmd_export_full_flag(tmp_path, capsys):
    db = str(tmp_path / "e.db")
    _, rid = _seed(db)
    cli.cmd_export(cli.build_parser().parse_args(["export", "--db", db, rid, "--stdout",
                                                  "--full"]))
    out = capsys.readouterr().out
    assert "line 0599" in out and "max_steps exceeded" in out
    cli.cmd_export(cli.build_parser().parse_args(["export", "--db", db, rid, "--stdout"]))
    assert "line 0599" not in capsys.readouterr().out


def test_full_view_shows_goal_rounds_compaction_and_a_fatal_guardrail(tmp_path):
    log = EventLog(str(tmp_path / "e.db"))
    rid = log.create_run("make the tests pass")
    log.append(rid, GOAL_ROUND, {"round": 1, "source": "goal"})
    log.append(rid, CONTEXT_COMPACTED, {"method": "summary", "before_tokens": 9000,
                                        "after_tokens": 1200})
    log.append(rid, GUARDRAIL, {"call_index": 3, "action": "fatal", "kind": None,
                                "rescued": False, "reason": "doom loop: grep called 6 times"})
    log.append(rid, GOAL_JUDGE, {"round": 1, "mode": "verdict", "answer": "not yet met",
                                 "reason": "2 tests still fail"})
    log.append(rid, GOAL_VERIFY, {"round": 1, "command": "pytest -q", "exit_code": 1,
                                  "output": "2 failed"})
    log.append(rid, GOAL_STATUS, {"status": "paused", "reason": "paused by the user",
                                  "round": 1})
    md = transcript_markdown(log, rid, full=True)
    assert "\n## Goal round 1\n" in md
    assert "*context compacted: 9000 → 1200 tokens (summary)*" in md
    assert "*guardrail · fatal (fatal)*: doom loop: grep called 6 times" in md
    assert "*goal judge (round 1): not yet met* — 2 tests still fail" in md
    assert "*goal verify (round 1): `pytest -q` exited 1*\n\n```\n2 failed\n```" in md
    assert "*goal paused: paused by the user*" in md


def test_full_view_stops_following_sub_agents_past_the_nesting_limit(tmp_path):
    log = EventLog(str(tmp_path / "e.db"))
    runs = [log.create_run(f"level {i}") for i in range(MAX_NESTING + 2)]
    for parent, child in zip(runs, runs[1:]):
        log.append(parent, AGENT_SPAWNED, {"child_run_id": child, "task": "deeper"})
    md = transcript_markdown(log, runs[0], full=True)
    assert f"sub-agent · level {MAX_NESTING}" in md  # 8 levels below the top are shown
    assert f"sub-agent · level {MAX_NESTING + 1}" not in md
    assert f"sub-agent run `{runs[-1]}` not shown: more than {MAX_NESTING} levels deep" in md
