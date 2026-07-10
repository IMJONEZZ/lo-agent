"""The CLI's UX surface: --version, runs filters, config precedence, doctor,
completion scripts, run diff — the verified gaps from the model's own audit of
this codebase (session 46eade64), each locked in with a test."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from local_harness.cli import main as cli
from local_harness.events.log import EventLog, MODEL_CALL


# ── lo --version ─────────────────────────────────────────────────────────────


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        cli.build_parser().parse_args(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("lo ")


# ── run titles: migration + rename ───────────────────────────────────────────


def test_title_column_migrates_old_databases(tmp_path):
    """A pre-title lo.db gains the column on open — no data lost."""
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, task TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running', created_at REAL NOT NULL);
        CREATE TABLE events (
            run_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL,
            payload TEXT NOT NULL, created_at REAL NOT NULL,
            PRIMARY KEY (run_id, seq));
        INSERT INTO runs VALUES ('abc123', 'old task', 'completed', 1.0);
        """
    )
    conn.commit()
    conn.close()
    log = EventLog(db)
    runs = log.runs()
    assert runs[0].run_id == "abc123" and runs[0].title is None
    assert runs[0].label == "old task"  # falls back to the task
    log.rename_run("abc123", "my old run")
    assert log.run("abc123").label == "my old run"
    log.rename_run("abc123", "   ")  # empty clears back to the task
    assert log.run("abc123").title is None


# ── lo runs: filters, --json, relative times ─────────────────────────────────


def _runs_args(db, **kw):
    base = dict(db=db, status=None, search=None, since=None, limit=None, json=False)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def seeded_db(tmp_path):
    db = str(tmp_path / "h.db")
    log = EventLog(db)
    a = log.create_run("first task about apples")
    log.append(a, "run_completed", {"answer": "x"})
    b = log.create_run("second task about bananas")  # stays running
    log.rename_run(b, "fruit stand")
    return db


def _json_rows(capsys):
    return json.loads(capsys.readouterr().out)


def test_runs_json_includes_titles(seeded_db, capsys):
    cli.cmd_runs(_runs_args(seeded_db, json=True))
    rows = _json_rows(capsys)
    assert len(rows) == 2
    assert rows[1]["title"] == "fruit stand"
    assert rows[0]["events"] >= 1


def test_runs_status_filter(seeded_db, capsys):
    cli.cmd_runs(_runs_args(seeded_db, status="completed", json=True))
    rows = _json_rows(capsys)
    assert len(rows) == 1 and "apples" in rows[0]["task"]


def test_runs_search_matches_title_and_task(seeded_db, capsys):
    cli.cmd_runs(_runs_args(seeded_db, search="fruit", json=True))
    assert len(_json_rows(capsys)) == 1  # by title
    cli.cmd_runs(_runs_args(seeded_db, search="apples", json=True))
    assert len(_json_rows(capsys)) == 1  # by task


def test_runs_since_and_limit(seeded_db, capsys):
    cli.cmd_runs(_runs_args(seeded_db, since="1h", json=True))
    assert len(_json_rows(capsys)) == 2
    cli.cmd_runs(_runs_args(seeded_db, limit=1, json=True))
    rows = _json_rows(capsys)
    assert len(rows) == 1 and rows[0]["title"] == "fruit stand"  # the newest
    with pytest.raises(SystemExit):
        cli._parse_since("soonish")


def test_runs_default_output_shows_label_and_relative_time(seeded_db, capsys):
    cli.cmd_runs(_runs_args(seeded_db))
    out = capsys.readouterr().out
    assert "fruit stand" in out and "just now" in out


def test_ago_units():
    now = time.time()
    assert cli._ago(now - 90) == "1m ago"
    assert cli._ago(now - 7200) == "2h ago"
    assert cli._ago(now - 3 * 86400) == "3d ago"


# ── lo config: persistent defaults with flag > env > config precedence ───────


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    path = str(tmp_path / "config.json")
    monkeypatch.setattr(cli, "_CONFIG_PATH", path)
    for var in ("LO_BASE_URL", "LO_MODEL", "LO_DB"):
        monkeypatch.delenv(var, raising=False)
    return path


def _cfg_args(action, key=None, value=None):
    return SimpleNamespace(action=action, key=key, value=value)


def test_config_set_feeds_parser_defaults(cfg_path, monkeypatch, capsys):
    cli.cmd_config(_cfg_args("set", "url", "http://box:8080"))
    args = cli.build_parser().parse_args(["probe"])
    assert args.url == "http://box:8080"
    # env beats config
    monkeypatch.setenv("LO_BASE_URL", "http://env:1")
    args = cli.build_parser().parse_args(["probe"])
    assert args.url == "http://env:1"
    # an explicit flag beats both
    args = cli.build_parser().parse_args(["probe", "--url", "http://flag:2"])
    assert args.url == "http://flag:2"


def test_config_show_get_unset_roundtrip(cfg_path, capsys):
    cli.cmd_config(_cfg_args("set", "model", "glm-5.2"))
    cli.cmd_config(_cfg_args("get", "model"))
    assert '"glm-5.2"' in capsys.readouterr().out
    cli.cmd_config(_cfg_args("show"))
    assert "model" in capsys.readouterr().out
    cli.cmd_config(_cfg_args("unset", "model"))
    cli.cmd_config(_cfg_args("get", "model"))
    assert "null" in capsys.readouterr().out


def test_config_rejects_unknown_keys(cfg_path):
    with pytest.raises(SystemExit):
        cli.cmd_config(_cfg_args("set", "hostnmae", "x"))  # typo caught


def test_config_coerces_booleans(cfg_path, capsys):
    cli.cmd_config(_cfg_args("set", "vim", "true"))
    assert json.load(open(cfg_path))["vim"] is True


def test_config_preserves_tui_keys(cfg_path, capsys):
    """Setting a CLI default must not clobber the TUI's theme/vim settings."""
    with open(cfg_path, "w") as f:
        json.dump({"theme": "osaka-jade"}, f)
    cli.cmd_config(_cfg_args("set", "url", "http://box:8080"))
    cfg = json.load(open(cfg_path))
    assert cfg["theme"] == "osaka-jade" and cfg["url"] == "http://box:8080"


# ── lo doctor ─────────────────────────────────────────────────────────────────


def test_doctor_dead_upstream_diagnoses_and_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_CONFIG_PATH", str(tmp_path / "config.json"))
    args = SimpleNamespace(
        url="http://127.0.0.1:9", model="", db=str(tmp_path / "h.db")
    )
    with pytest.raises(SystemExit) as e:
        asyncio.run(cli.cmd_doctor(args))
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "unreachable" in out and "quickstart" in out  # diagnosis + fix
    assert "event log" in out  # later checks still ran


def test_doctor_healthy_setup_passes(tmp_path, monkeypatch, capsys):
    from e2e_support import start_mock_upstream, stop_server
    from mocks import MockLlamaCpp, chat_response

    monkeypatch.setattr(cli, "_CONFIG_PATH", str(tmp_path / "config.json"))
    mock = MockLlamaCpp(chat_fn=lambda body: chat_response(content="ok"))
    url, server = start_mock_upstream(mock)
    try:
        args = SimpleNamespace(url=url, model="", db=str(tmp_path / "h.db"))
        asyncio.run(cli.cmd_doctor(args))  # no SystemExit → healthy
    finally:
        stop_server(server)
    out = capsys.readouterr().out
    assert "all good" in out and "✓ upstream" in out


# ── lo completion ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shell,needle",
    [
        ("bash", "complete -o default -F _lo_complete lo"),
        ("zsh", "compdef _lo lo"),
        ("fish", "__fish_use_subcommand"),
    ],
)
def test_completion_scripts_cover_commands_and_flags(shell, needle, capsys):
    cli.cmd_completion(SimpleNamespace(shell=shell))
    out = capsys.readouterr().out
    assert needle in out
    assert "doctor" in out and "runs" in out  # generated from the live parser
    assert "--status" in out or "status" in out


# ── lo diff ──────────────────────────────────────────────────────────────────


def _seed_answer(log, task, answer):
    rid = log.create_run(task)
    log.append(rid, MODEL_CALL, {
        "call_index": 0,
        "response": {"choices": [{
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }]},
    })
    return rid


def test_diff_shows_transcript_divergence(tmp_path, capsys):
    db = str(tmp_path / "h.db")
    log = EventLog(db)
    a = _seed_answer(log, "same task", "alpha answer")
    b = _seed_answer(log, "same task", "beta answer")
    with pytest.raises(SystemExit) as e:
        cli.cmd_diff(SimpleNamespace(db=db, run_a=a, run_b=b))
    assert e.value.code == 1  # differ → nonzero, like diff(1)
    out = capsys.readouterr().out
    assert "-alpha answer" in out and "+beta answer" in out


def test_diff_identical_runs_say_so(tmp_path, capsys):
    db = str(tmp_path / "h.db")
    log = EventLog(db)
    a = _seed_answer(log, "same task", "same answer")
    b = _seed_answer(log, "same task", "same answer")
    cli.cmd_diff(SimpleNamespace(db=db, run_a=a, run_b=b))
    assert "identical transcripts" in capsys.readouterr().out


def test_diff_unknown_run_errors(tmp_path):
    db = str(tmp_path / "h.db")
    EventLog(db)
    with pytest.raises(SystemExit):
        cli.cmd_diff(SimpleNamespace(db=db, run_a="nope", run_b="nada"))
