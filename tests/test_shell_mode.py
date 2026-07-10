"""Part C — interactive `!` shell mode in the TUI."""

import asyncio

from textual.widgets import Input

from local_harness.inference.client import OpenAICompatClient
from local_harness.skills.skill import BUILTIN_SKILLS_DIR
from local_harness.tui.app import HarnessApp

from mocks import MockLlamaCpp, chat_response

SKILLS_DIR = str(BUILTIN_SKILLS_DIR)


def _app(db):
    mock = MockLlamaCpp(script={424242: chat_response(content="p")})
    client = OpenAICompatClient("http://upstream", "test-model", transport=mock.transport())
    return HarnessApp(client, db, skills_dir=SKILLS_DIR)


async def test_bang_toggles_shell_mode_and_ui(tmp_path):
    app = _app(str(tmp_path / "h.db"))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        box = app.query_one("#prompt", Input)

        box.value = "!echo hi"  # leading ! → shell mode on
        await pilot.pause()
        assert app._shell_mode is True
        assert box.has_class("shell-mode")
        assert "shell command" in box.placeholder

        box.value = "hello"  # no leading ! → shell mode off
        await pilot.pause()
        assert app._shell_mode is False
        assert not box.has_class("shell-mode")
        assert box.placeholder == app._PROMPT_PLACEHOLDER


async def test_submit_shell_runs_via_sandbox_and_buffers(tmp_path):
    app = _app(str(tmp_path / "h.db"))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)

        class FakeSandbox:
            kind = "host"

            async def exec(self, cmd, timeout=30):
                return (f"OUT:{cmd}", 0)

        app._sandbox = FakeSandbox()
        await app._submit_shell("ls -la")
        # buffered for the next turn (not a model turn itself)
        assert app._pending_shell == [("ls -la", "OUT:ls -la")]


async def test_pending_shell_prepended_to_next_turn(tmp_path):
    from local_harness.events.log import MODEL_CALL

    db = str(tmp_path / "h.db")
    mock = MockLlamaCpp(
        script={424242: chat_response(content="p"), 1: chat_response(content="ok")}
    )
    client = OpenAICompatClient("http://upstream", "test-model", transport=mock.transport())
    app = HarnessApp(client, db, skills_dir=SKILLS_DIR)
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        app._pending_shell = [("pwd", "/repo")]
        box = app.query_one(Input)
        box.focus()
        box.value = "what does pwd say?"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(100):
            await pilot.pause(0.05)
            runs = app.event_log.runs()
            if runs and runs[-1].status == "completed":
                break
        # the buffer was consumed and folded into the turn that reached the model
        assert app._pending_shell == []
        run_id = app.event_log.runs()[-1].run_id
        calls = app.event_log.events(run_id, type=MODEL_CALL)
        msgs = calls[0].payload["request_body"]["messages"]
        joined = "\n".join(m.get("content") or "" for m in msgs)
        assert "shell output" in joined and "/repo" in joined
