"""Phase 1 CLI additions: export / cost / usage commands, the transcript-markdown
module, the live `harness run` status helper, and quickstart's probe."""

from __future__ import annotations

import io
import os

from rich.console import Console

from local_harness.cli import main as cli
from local_harness.events.export import transcript_markdown
from local_harness.events.log import (
    EventLog, MODEL_CALL, POLICY_TRIGGERED, RUN_COMPLETED, TOOL_CALL, USER_MESSAGE,
)
from local_harness.inference.capabilities import Capabilities

from mocks import chat_response


def _seed(db) -> tuple[EventLog, str]:
    log = EventLog(db)
    rid = log.create_run("explain Apollo 13")
    log.append(rid, MODEL_CALL, {"call_index": 0, "seed": 1,
               "response": dict(chat_response(content="An oxygen tank exploded."),
                                usage={"prompt_tokens": 1000, "completion_tokens": 200})})
    log.append(rid, TOOL_CALL, {"name": "web_search", "arguments": '{"q":"apollo 13"}',
                                "result": "results about Apollo 13"})
    log.append(rid, POLICY_TRIGGERED, {"call_index": 0, "attempt": 0,
                                       "action": "resample", "reason": "low conf"})
    log.append(rid, RUN_COMPLETED, {"answer": "An oxygen tank exploded; they used the LM as a lifeboat."})
    return log, rid


def test_transcript_markdown_sections(tmp_path):
    log, rid = _seed(str(tmp_path / "h.db"))
    md = transcript_markdown(log, rid)
    assert md.startswith("# explain Apollo 13")
    assert "## Assistant" in md and "An oxygen tank exploded." in md
    assert "### result · web_search" in md and "results about Apollo 13" in md


def test_cmd_export_writes_file(tmp_path, monkeypatch):
    db = str(tmp_path / "h.db")
    _, rid = _seed(db)
    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args(["export", "--db", db, rid])
    cli.cmd_export(args)
    out = tmp_path / f"run-{rid}.md"
    assert out.exists() and "Apollo 13" in out.read_text()


def test_cmd_export_stdout(tmp_path, capsys):
    db = str(tmp_path / "h.db")
    _, rid = _seed(db)
    cli.cmd_export(cli.build_parser().parse_args(["export", "--db", db, rid, "--stdout"]))
    assert "## Assistant" in capsys.readouterr().out


def test_cmd_cost_and_usage(tmp_path, capsys):
    db = str(tmp_path / "h.db")
    _seed(db)
    cli.cmd_cost(cli.build_parser().parse_args(["cost", "--db", db]))
    cost_out = capsys.readouterr().out
    assert "$0.00 spent" in cost_out and "frontier API" in cost_out

    cli.cmd_usage(cli.build_parser().parse_args(["usage", "--db", db]))
    usage_out = capsys.readouterr().out
    assert "calls" in usage_out and "resamples  1" in usage_out and "tokens     1,200" in usage_out


def test_cli_status_renders_tier_and_ctx(tmp_path):
    db = str(tmp_path / "h.db")
    log, rid = _seed(db)
    caps = Capabilities(server="llama.cpp", seed=True, logprobs=True, context_window=262144)
    r = cli._cli_status(log, rid, caps, t0=0.0)  # final form (no spinner)
    console = Console(width=100, record=True, file=io.StringIO())
    console.print(r)
    text = console.export_text()
    assert "tier" in text and "run" in text  # the preset chip + tier segment
