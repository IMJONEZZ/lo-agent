"""Live tier: real `lo tui` in a PTY against a REAL model endpoint.

    LO_LIVE_URL=http://192.168.1.185:8080 uv run pytest -m live

Optionally LO_LIVE_MODEL pins the served model. These journeys exercise real
model behavior (does plan mode actually produce an approvable plan? does the
permission modal really gate a live tool call?), so markers are answer text
the scenario instructs the model to CONSTRUCT, timeouts are stretched, and a
run ends with history-bulk-delete, wiping the conversations it created.
"""

from __future__ import annotations

import os

import pytest

from local_harness.sim import SCENARIOS, run_scenario
from local_harness.sim.pty_driver import PtyTui, tui_command
from local_harness.sim.scenarios import GREET_COMMAND_MD, scaled

LIVE_URL = os.environ.get("LO_LIVE_URL", "")
LIVE_MODEL = os.environ.get("LO_LIVE_MODEL", "")
SCALE = float(os.environ.get("LO_LIVE_TIMEOUT_SCALE", "8"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE_URL, reason="set LO_LIVE_URL to a model endpoint"),
]


async def play_live(name: str, tmp_path):
    scenario = scaled(SCENARIOS[name], SCALE)
    cmd_dir = tmp_path / ".lo" / "commands"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    (cmd_dir / "greet.md").write_text(GREET_COMMAND_MD)
    driver = PtyTui(
        tui_command(LIVE_URL, str(tmp_path / "h.db"), model=LIVE_MODEL),
        env={"HOME": str(tmp_path)},
        cwd=str(tmp_path),
    )
    try:
        assert await driver.boot("local_harness", timeout=90), driver.dump()
        await run_scenario(driver, scenario)
    finally:
        driver.close()
    return tmp_path


async def test_live_chat_turn_streaming(tmp_path):
    await play_live("chat-turn-streaming", tmp_path)


async def test_live_plan_approve_build(tmp_path):
    await play_live("plan-approve-build", tmp_path)


async def test_live_permission_allow(tmp_path):
    await play_live("permission-allow", tmp_path)
    assert (tmp_path / "perm-test.txt").exists()


async def test_live_permission_deny(tmp_path):
    await play_live("permission-deny", tmp_path)
    assert not (tmp_path / "perm-test.txt").exists()


async def test_live_custom_file_command(tmp_path):
    await play_live("custom-file-command", tmp_path)


async def test_live_interrupt_mid_stream(tmp_path):
    await play_live("interrupt-mid-stream", tmp_path)


async def test_live_history_bulk_delete(tmp_path):
    """Last: creates three real conversations and deletes every one of them
    through the sidebar — the recurring problem area, against a real stream."""
    await play_live("history-bulk-delete", tmp_path)
