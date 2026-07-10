"""User-simulation journeys: the TUI driven like a user drives it, over the
REAL session server (HTTP + SSE on a localhost socket) with a mock model.

Each test plays a scenario from local_harness.sim.scenarios through the
PilotDriver and then asserts semantic post-conditions (db state, files on
disk, what the model was actually sent). Failures raise ScenarioFailure with
the full screen text embedded.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_harness.events.log import EventLog
from local_harness.sim import SCENARIOS, run_scenario
from local_harness.sim.scenarios import GREET_COMMAND_MD

from e2e_support import (
    PilotDriver,
    agent_bodies,
    assert_ux_invariants,
    make_server_app,
    scripted_chat,
    start_server,
    stop_server,
)
from mocks import MockLlamaCpp, chat_response

pytestmark = pytest.mark.e2e


@pytest.fixture
def sim_env(tmp_path, monkeypatch):
    """Isolated cwd + config for one journey; tears the server down after."""
    monkeypatch.chdir(tmp_path)
    import local_harness.tui.app as tui_app

    monkeypatch.setattr(tui_app, "_CONFIG_PATH", str(tmp_path / "config.json"))
    servers: list = []
    yield tmp_path, servers
    for s in servers:
        stop_server(s)


async def play(name: str, sim_env, mock: MockLlamaCpp | None = None, **server_kw):
    """Start the real server, boot the TUI as its client, run the journey.
    Returns (app, mock, db) for post-condition assertions."""
    tmp_path, servers = sim_env
    scenario = SCENARIOS[name]
    mock = mock or MockLlamaCpp(chat_fn=scripted_chat(scenario))
    url, server, db = start_server(tmp_path, mock, **server_kw)
    servers.append(server)
    app = make_server_app(db, mock, url)
    async with app.run_test(size=(120, 42)) as pilot:
        driver = PilotDriver(app, pilot)
        app.query_one("#prompt").focus()
        await run_scenario(driver, scenario)
    assert_ux_invariants(db)  # the UX floor holds after EVERY journey
    return app, mock, db


async def test_first_run_welcome(sim_env):
    app, _, _ = await play("first-run-welcome", sim_env)
    assert app.caps is not None  # probed through the real /health route


async def test_chat_turn_streaming(sim_env):
    _, mock, db = await play("chat-turn-streaming", sim_env)
    runs = EventLog(db).runs()
    assert len(runs) == 1 and runs[0].status == "completed"
    assert mock.chat_calls >= 1  # the answer really came through the server


async def test_slash_autocomplete_and_help(sim_env):
    await play("slash-autocomplete", sim_env)


async def test_shell_mode_buffers_output_into_next_turn(sim_env):
    _, mock, _ = await play("shell-mode-buffering", sim_env)
    sent = json.dumps(mock.chat_bodies[-1]["messages"])
    assert "sim-shell-ok" in sent  # the ! output rode into the model context


async def test_plan_approve_build(sim_env):
    _, mock, db = await play("plan-approve-build", sim_env)
    # Server-side preset enforcement: the plan turn's exposed toolset is
    # read-only — write_file must not be offered to the model.
    bodies = agent_bodies(mock)
    plan_tools = [t["function"]["name"] for t in (bodies[0].get("tools") or [])]
    assert plan_tools and "write_file" not in plan_tools
    # The approved build turn regains full tools.
    build_tools = [t["function"]["name"] for t in (bodies[-1].get("tools") or [])]
    assert "write_file" in build_tools
    assert EventLog(db).runs()[-1].status == "completed"


async def test_permission_modal_allow_runs_tool(sim_env):
    tmp_path, _ = sim_env
    await play("permission-allow", sim_env, chat_delay=0.5)
    assert (tmp_path / "perm-test.txt").read_text() == "PERMOK"


async def test_permission_modal_deny_blocks_tool(sim_env):
    tmp_path, _ = sim_env
    await play("permission-deny", sim_env, chat_delay=0.5)
    assert not (tmp_path / "perm-test.txt").exists()


async def test_history_bulk_delete(sim_env):
    """The consistent problem area: wipe EVERY conversation (including the
    active one) through the history sidebar, then start fresh."""
    app, _, db = await play("history-bulk-delete", sim_env)
    runs = EventLog(db).runs()
    # 3 created + all deleted + 1 fresh turn afterwards
    assert len(runs) == 1 and runs[0].status == "completed"
    assert app.active == runs[0].run_id


async def test_rewind_picker_truncates(sim_env):
    app, _, db = await play("rewind-picker", sim_env)
    log = EventLog(db)
    runs = log.runs()
    # rewind archives the removed tail as a new (archive) run
    assert len(runs) >= 2
    original = runs[0].run_id
    # The view must stay on the rewound conversation, NOT jump to the archive
    # (regression: follow-latest used to switch to the archive run).
    assert app.active == original
    events = json.dumps([e.payload for e in log.events(original)])
    assert "SECOND-ANSWER" not in events  # the second turn is gone from context


async def test_custom_file_command(sim_env):
    tmp_path, _ = sim_env
    cmd_dir = tmp_path / ".lo" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "greet.md").write_text(GREET_COMMAND_MD)
    mock = MockLlamaCpp(chat_fn=lambda body: chat_response(content="GREET-ACK"))
    await play("custom-file-command", sim_env, mock=mock)


async def test_mode_preset_switch(sim_env):
    app, _, _ = await play("mode-preset-switch", sim_env)
    assert app._preset.name == "build"


async def test_theme_switch_applies_and_persists(sim_env):
    tmp_path, _ = sim_env
    await play("theme-switch", sim_env)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg.get("theme") == "osaka-jade"


async def test_export_transcript(sim_env):
    tmp_path, _ = sim_env
    await play("export-transcript", sim_env)
    exports = list(tmp_path.glob("run-*.md"))
    assert exports and "BANANA" in exports[0].read_text()


async def test_codemode_import_recovery(sim_env):
    """The journey distilled from real failed session 46eade64: a model that
    reflexively writes `import os` must be TAUGHT by the error and finish the
    simple query in the same turn — not spin until the user hits Esc."""
    _, mock, db = await play("codemode-import-recovery", sim_env)
    runs = EventLog(db).runs()
    assert runs[-1].status == "completed"
    # The model really was shown the teaching error before it corrected.
    sent = json.dumps(agent_bodies(mock)[-1]["messages"])
    assert "isn't available in code mode" in sent


async def test_codemode_crash_loop_breaker(sim_env):
    """A model that never self-corrects gets cut off by the tool-error budget
    with the reason on screen — bounded failure, not an infinite loop."""
    from local_harness.events.log import TOOL_CALL

    _, _, db = await play("codemode-crash-loop-breaker", sim_env)
    log = EventLog(db)
    run = log.runs()[-1]
    assert run.status == "failed"
    n_attempts = sum(
        1 for e in log.events(run.run_id, type=TOOL_CALL)
        if e.payload.get("name") == "run_code"
    )
    assert n_attempts <= 3  # budget of 2 + the batch that trips it


async def test_history_filter_and_rename(sim_env):
    """'/' narrows the sidebar to matching conversations; 'r' gives one a
    human name that persists in the event log (and lo runs)."""
    _, _, db = await play("history-filter-rename", sim_env)
    runs = EventLog(db).runs()
    assert any(r.title == "standup notes" for r in runs)


async def test_inspect_model_calls(sim_env):
    """/inspect surfaces per-call timing/token/logprob stats already in the log."""
    await play("inspect-model-calls", sim_env)


async def test_upstream_down_notice(sim_env):
    class DeadUpstream(MockLlamaCpp):
        def handler(self, request):
            import httpx

            raise httpx.ConnectError("connection refused")

    await play("upstream-down-notice", sim_env, mock=DeadUpstream())
