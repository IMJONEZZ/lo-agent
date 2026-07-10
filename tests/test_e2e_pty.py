"""PTY smoke tier: the real `lo tui` process in a real pseudo-terminal, its
embedded server pointed at MockLlamaCpp behind a real localhost socket.

The only layer that catches "the TUI crashes at real-terminal boot" and key
encoding bugs the Pilot can't see. Opt in with `pytest -m pty`.
"""

from __future__ import annotations

import pytest

from local_harness.sim import SCENARIOS, run_scenario
from local_harness.sim.pty_driver import PtyTui, tui_command

from e2e_support import scripted_chat, start_mock_upstream, stop_server
from mocks import MockLlamaCpp

pytestmark = pytest.mark.pty


async def play_pty(name: str, tmp_path):
    scenario = SCENARIOS[name]
    mock = MockLlamaCpp(chat_fn=scripted_chat(scenario))
    url, upstream = start_mock_upstream(mock)
    driver = PtyTui(
        tui_command(url, str(tmp_path / "h.db")),
        env={"HOME": str(tmp_path)},  # isolate ~/.lo config + memory
        cwd=str(tmp_path),
    )
    try:
        assert await driver.boot("local_harness"), driver.dump()
        await run_scenario(driver, scenario)
    finally:
        driver.close()
        stop_server(upstream)


async def test_pty_first_run_welcome(tmp_path):
    await play_pty("first-run-welcome", tmp_path)


async def test_pty_chat_turn_streaming(tmp_path):
    await play_pty("chat-turn-streaming", tmp_path)
