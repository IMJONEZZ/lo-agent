"""TUI: render functions (pure) + headless app tests via Textual's pilot."""

import asyncio
import io

from rich.console import Console
from textual.widgets import DataTable, Input

from local_harness.events.log import (
    GUARDRAIL,
    MODEL_CALL,
    RUN_COMPLETED,
    RUN_FAILED,
    TOOL_CALL,
    Event,
    EventLog,
)
from local_harness.inference.client import OpenAICompatClient
from local_harness.tui.app import HarnessApp
from local_harness.events.log import POLICY_TRIGGERED, RUN_STARTED, USER_MESSAGE
from local_harness.tui.render import (
    ability_glyphs, banner_body, chat_render_event, confidence_text, frontier_saved,
    render_event, status_bar, status_text, tokens_of, usage_panel, welcome_panel,
)
from local_harness.inference.capabilities import Capabilities

from mocks import MockLlamaCpp, chat_response


def text_of(renderable) -> str:
    console = Console(width=100, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


def ev(type: str, payload: dict) -> Event:
    return Event(run_id="r", seq=0, type=type, payload=payload, created_at=0.0)


def test_render_model_call_shows_content_tools_and_confidence():
    payload = {
        "call_index": 2, "seed": 3, "timing_ms": 123.4, "request_body": {},
        "logprob_summary": {"n_tokens": 2, "mean_logprob": -0.27, "min_logprob": -0.5,
                            "mean_entropy": 0.1, "mean_top2_margin": 1.0},
        "response": chat_response(content="the answer",
                                  tool_calls=[("c1", "calculator", '{"expression": "6*7"}')]),
    }
    out = text_of(render_event(ev(MODEL_CALL, payload)))
    assert "the answer" in out
    assert "calculator" in out
    assert "seed 3" in out and "logprob -0.27" in out


def test_render_tool_call_marks_errors():
    ok = text_of(render_event(ev(TOOL_CALL, {
        "name": "calculator", "arguments": '{"expression": "6*7"}', "result": "42"})))
    assert "calculator" in ok and "42" in ok
    err = text_of(render_event(ev(TOOL_CALL, {
        "name": "read_file", "arguments": "{}", "result": "error: no such file"})))
    assert "error:" in err


def test_render_guardrail_variants():
    rescued = text_of(render_event(ev(GUARDRAIL, {
        "action": "execute", "rescued": True,
        "rescued_calls": [{"id": "x", "type": "function",
                           "function": {"name": "calculator", "arguments": "{}"}}]})))
    assert "rescued" in rescued and "calculator" in rescued

    nudged = text_of(render_event(ev(GUARDRAIL, {
        "action": "nudge", "kind": "unknown_tool", "rescued": False,
        "nudge": {"role": "tool", "content": "tool does not exist", "tool_call_id": "x"}})))
    assert "does not exist" in nudged

    # routine final/execute checks render nothing
    assert render_event(ev(GUARDRAIL, {"action": "final", "rescued": False})) is None

    fatal = text_of(render_event(ev(GUARDRAIL, {
        "action": "fatal", "rescued": False, "reason": "error budget exhausted"})))
    assert "budget" in fatal


def test_render_terminal_events_and_status():
    assert "all good" in text_of(render_event(ev(RUN_COMPLETED, {"answer": "all good"})))
    assert "boom" in text_of(render_event(ev(RUN_FAILED, {"error": "boom"})))
    # Osaka Jade palette: completed=jade, failed=rose (assert against the palette
    # constants so a future re-skin doesn't require touching this test).
    from local_harness.tui.render import JADE, ROSE
    assert status_text("completed").style == JADE
    assert status_text("failed").style == ROSE


def test_chat_render_user_assistant_and_badges():
    started = text_of(chat_render_event(ev(RUN_STARTED, {"task": "sum 2 and 2"})))
    assert "›" in started and "sum 2 and 2" in started

    payload = {
        "seed": 3, "timing_ms": 1200.0,
        "logprob_summary": {"n_tokens": 2, "mean_logprob": -0.21, "min_logprob": -0.4,
                            "mean_entropy": 0.05, "mean_top2_margin": 1.0},
        "grammar_valid": True,
        "response": chat_response(content="SELECT 1;"),
    }
    out = text_of(chat_render_event(ev(MODEL_CALL, payload)))
    assert "SELECT 1;" in out
    assert "seed 3" in out and "logprob -0.21" in out and "1.2s" in out
    assert "grammar-valid" in out  # feature badge surfaces when present


def test_welcome_panel_lists_unlocked_features():
    caps = Capabilities(server="llama.cpp", seed=True, logprobs=True, grammar="gbnf",
                        logit_bias=True, sampler_zoo={"dry", "min_p", "xtc"},
                        raw_completion=True, kv_snapshot=True)
    out = text_of(welcome_panel("qwen3.6-27b", caps))
    assert "qwen3.6-27b" in out and "tier 3" in out
    assert "exact replay" in out and "grammar skills" in out and "KV-fork" in out

    bare = Capabilities(server="generic")  # tier 0
    bare_out = text_of(welcome_panel("some-model", bare))
    assert "basic agent loop" in bare_out


async def test_slash_autocomplete_filters_and_completes(tmp_path):
    from textual.widgets import Input, OptionList
    app = make_app(str(tmp_path / "h.db"),
                   MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        box = app.query_one("#prompt", Input); box.focus()
        for ch in "/he":
            await pilot.press(ch)
        await pilot.pause()
        menu = app.query_one("#slashmenu", OptionList)
        assert menu.has_class("visible")
        ids = [menu.get_option_at_index(i).id for i in range(menu.option_count)]
        assert ids == ["help"]                       # filtered to the match
        await pilot.press("tab")                     # complete
        await pilot.pause()
        assert box.value == "/help " and not menu.has_class("visible")


async def test_resample_ghosts_the_rejected_attempt(tmp_path):
    app = make_app(str(tmp_path / "h.db"),
                   MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        # simulate the agent streaming an attempt, then a resample restarting
        app._on_token("start", "")
        app._on_token("content", "first attempt")
        assert app._live_ghost == ""
        app._on_token("start", "")            # resample fires
        assert app._live_ghost == "first attempt"   # the rejected attempt is ghosted
        app._on_token("content", "better answer")
        assert app._live_content == "better answer" and app._live_ghost == "first attempt"


def _caps(**kw):
    base = dict(server="llamacpp", model="m", seed=True, logprobs=True, grammar="gbnf",
                logit_bias=True, sampler_zoo={"min_p", "dry"}, parallel_n=True)
    base.update(kw)
    return Capabilities(**base)


def test_status_bar_surfaces_zero_cost_tier_and_determinism():
    out = text_of(status_bar(preset="build", tier=3, glyphs=ability_glyphs(_caps()),
                             saved=0.21, deterministic=True, learn="idle"))
    assert "build" in out and "tier 3" in out
    assert "$0.00 spent" in out and "$0.21 saved" in out
    assert "deterministic" in out and "idle-learn" in out
    # best-effort wording when the server can't pin a seed
    assert "best-effort" in text_of(status_bar(
        preset="plan", tier=1, glyphs=[], saved=0.0, deterministic=False, learn="off"))


def test_banner_body_lists_capabilities_tools_skills():
    out = text_of(banner_body("qwen", _caps(), tools=["read_file", "bash"],
                              skills=["yes_no"], memory_summary="3 memories",
                              preset_name="build", preset_blurb="full toolset"))
    assert "qwen" in out and "tier 3" in out
    assert "read_file" in out and "yes_no" in out
    assert "3 memories" in out and "$0.00 spent" in out


def test_usage_panel_reports_cost_confidence_resamples():
    stats = {"calls": 2, "tokens": 1000, "saved": 0.05,
             "conf": [10, 3, 1], "mean_lp": -0.4, "resamples": 1}
    out = text_of(usage_panel(stats, model="m", tier=3, deterministic=True,
                              learn_summary="+1 lessons · $0 spent"))
    assert "$0.00 spent" in out and "bit-exact" in out
    assert "resample" in out and "+1 lessons" in out


def test_tokens_and_frontier_saved():
    p = {"response": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}}
    assert tokens_of(p) == (100, 50)
    assert frontier_saved(100, 50) > 0
    assert tokens_of({}) == (0, 0)


def test_confidence_text_varies_style_by_logprob():
    t = confidence_text([{"token": "sure ", "logprob": -0.01},
                         {"token": "unsure", "logprob": -3.5}])
    assert t.plain == "sure unsure"
    styles = {str(s.style) for s in t.spans}
    assert len(styles) >= 2  # confident and low-confidence tokens rendered differently


def test_chat_render_shows_followup_user_message():
    # a continued turn (USER_MESSAGE) must appear in the stream, not vanish
    out = text_of(chat_render_event(ev(USER_MESSAGE, {"content": "now double it"})))
    assert "now double it" in out


def test_chat_render_resample_is_visible():
    out = text_of(chat_render_event(ev(POLICY_TRIGGERED,
        {"action": "resample", "reason": "low confidence", "attempt": 0})))
    assert "resample" in out and "trying again" in out
    # run_completed adds nothing in chat (answer already shown as the last turn)
    assert chat_render_event(ev(RUN_COMPLETED, {"answer": "x"})) is None


def seed_old_run(db: str) -> str:
    log = EventLog(db)
    run_id = log.create_run("old task")
    log.append(run_id, MODEL_CALL, {
        "call_index": 0, "seed": 1, "request_body": {}, "timing_ms": 1.0,
        "logprob_summary": None, "response": chat_response(content="prior answer")})
    log.append(run_id, RUN_COMPLETED, {"answer": "prior answer"})
    log.close()
    return run_id


from local_harness.skills.skill import BUILTIN_SKILLS_DIR

SKILLS_DIR = str(BUILTIN_SKILLS_DIR)


def make_app(db: str, mock: MockLlamaCpp) -> HarnessApp:
    client = OpenAICompatClient("http://upstream", "test-model", transport=mock.transport())
    return HarnessApp(client, db, skills_dir=SKILLS_DIR)


async def test_app_undo_aliases_rewind(tmp_path):
    db = str(tmp_path / "h.db")
    run_id = seed_old_run(db)  # RUN_STARTED, MODEL_CALL, RUN_COMPLETED
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        assert app.active == run_id
        before = len(app.event_log.events(run_id))
        runs_before = len(app.event_log.runs())
        box = app.query_one(Input)
        box.focus()
        box.value = "/undo"          # now an alias of /rewind → opens the picker
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        from local_harness.tui.app import RewindScreen
        assert isinstance(app.screen, RewindScreen)
        await pilot.press("enter")   # select the highlighted rewind point (the original answer)
        await pilot.pause()
        after = app.event_log.events(run_id)
        assert len(after) < before
        assert not any(e.type == MODEL_CALL for e in after)   # the turn was rewound away
        assert len(app.event_log.runs()) == runs_before + 1   # tail archived as a new run (lossless)


async def test_history_delete_keeps_table_focused(tmp_path):
    # Deleting a run must keep the history sidebar open AND focused, so you can
    # delete the next one without closing/reopening (the reported lockup).
    db = str(tmp_path / "h.db")
    r1 = seed_old_run(db)
    r2 = seed_old_run(db)
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        runs = app.query_one("#runs", DataTable)
        pane = app.query_one("#runspane")
        app.action_toggle_runs()          # ^t → open sidebar
        await pilot.pause()
        assert pane.has_class("visible") and app.focused is runs
        before = len(app.event_log.runs())
        app.action_delete_run()           # delete the highlighted run
        await pilot.pause()
        assert len(app.event_log.runs()) == before - 1
        # still open, still focused → a second delete works immediately
        assert pane.has_class("visible") and app.focused is runs
        assert r1 in (r1, r2)  # (both were valid run ids)


async def test_history_open_closes_sidebar(tmp_path):
    # Opening a run closes the sidebar and returns focus to the prompt.
    db = str(tmp_path / "h.db")
    seed_old_run(db)
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        pane = app.query_one("#runspane")
        app.action_toggle_runs()
        await pilot.pause()
        assert pane.has_class("visible")
        await pilot.press("enter")        # open the highlighted run
        await pilot.pause()
        assert not pane.has_class("visible")
        assert isinstance(app.focused, Input)


async def test_app_runs_skill_via_slash(tmp_path):
    db = str(tmp_path / "h.db")
    # the skill generates at seed 1; grammar-constrained 'yes_no' accepts "yes"
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p"),
                                            1: chat_response(content="yes")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        box = app.query_one(Input)
        box.focus()
        box.value = "/yes_no Is water wet?"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(100):
            await pilot.pause(0.05)
            runs = app.event_log.runs()
            if runs and runs[-1].status == "completed":
                break
        runs = app.event_log.runs()
        assert runs[-1].task.startswith("/yes_no")
        call = app.event_log.events(runs[-1].run_id, type=MODEL_CALL)[0]
        assert call.payload["grammar_valid"] is True      # the grammar-valid badge fires
        done = app.event_log.events(runs[-1].run_id, type=RUN_COMPLETED)
        assert done[-1].payload["answer"] == "yes"


async def test_app_lists_existing_runs_and_follows_latest(tmp_path):
    db = str(tmp_path / "h.db")
    run_id = seed_old_run(db)
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        assert app.query_one(DataTable).row_count == 1
        assert app.active == run_id
        assert app._rendered == 3  # run_started + model_call + run_completed


async def test_tui_wires_default_permissions(tmp_path):
    app = make_app(str(tmp_path / "h.db"),
                   MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        perms = app._tool_registry.permissions
        assert perms is not None
        assert perms.decide("bash") == "ask" and perms.decide("write_file") == "ask"
        assert perms.decide("read_file") == "allow"


async def test_allow_all_disables_permissions(tmp_path):
    client = OpenAICompatClient("http://upstream", "test-model",
                                transport=MockLlamaCpp(script={424242: chat_response(content="p")}).transport())
    app = HarnessApp(client, str(tmp_path / "h.db"), skills_dir=SKILLS_DIR, allow_all=True)
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        assert app._tool_registry.permissions is None


async def test_app_deletes_run_from_history(tmp_path):
    db = str(tmp_path / "h.db")
    r1 = seed_old_run(db)
    r2 = seed_old_run(db)
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        await pilot.press("ctrl+t")  # reveal history sidebar
        await pilot.pause()
        app.query_one(DataTable).move_cursor(row=0)  # highlight the older run (r1)
        await pilot.pause()
        await pilot.press("delete")
        await pilot.pause()
        ids = [r.run_id for r in app.event_log.runs()]
        assert r1 not in ids and r2 in ids and len(ids) == 1


async def test_app_continues_a_loaded_run(tmp_path):
    db = str(tmp_path / "h.db")
    run_id = seed_old_run(db)  # one completed run, model call at seed 1
    # continuation runs at call_index 1 -> seed 2
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p"),
                                            2: chat_response(content="continued answer")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        assert app.active == run_id and app._active_status() == "completed"
        box = app.query_one(Input)
        box.focus()
        box.value = "a follow-up"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(100):
            await pilot.pause(0.05)
            if app.event_log.events(run_id, type=USER_MESSAGE) \
               and app.event_log.run(run_id).status == "completed":
                break
        assert len(app.event_log.runs()) == 1  # same conversation, not a new run
        done = app.event_log.events(run_id, type=RUN_COMPLETED)
        assert done[-1].payload["answer"] == "continued answer"


async def test_app_launches_agent_from_input(tmp_path):
    db = str(tmp_path / "h.db")  # empty: a fresh submit starts a new run
    # agent's first call uses seed base_seed(1) + call_index(0) = 1
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p"),
                                            1: chat_response(content="fresh answer")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        box = app.query_one(Input)
        box.focus()
        box.value = "say something"
        await pilot.pause()
        await pilot.press("enter")

        for _ in range(100):
            await pilot.pause(0.05)
            runs = app.event_log.runs()
            if len(runs) == 1 and runs[-1].status == "completed":
                break
        runs = app.event_log.runs()
        assert len(runs) == 1 and runs[-1].status == "completed"

        await pilot.pause(0.6)  # let the poll tick select + render the new run
        assert app.active == runs[-1].run_id
        assert app.query_one(DataTable).row_count == 1
        done = app.event_log.events(runs[-1].run_id, type=RUN_COMPLETED)
        assert done[-1].payload["answer"] == "fresh answer"


async def test_connect_modal_input_not_submitted_as_message(tmp_path):
    # a provider URL typed into the Connect modal must not become a chat message
    db = str(tmp_path / "h.db")
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._caps_ready.wait(), timeout=5)
        await pilot.pause()
        from local_harness.tui.app import ConnectScreen
        app.action_connect()
        await pilot.pause()
        assert isinstance(app.screen, ConnectScreen)
        app.screen.query_one("#c_url", Input).value = "http://127.0.0.1:1"  # unreachable, harmless
        await pilot.press("enter")
        await pilot.pause()
        tasks = [r.task for r in app.event_log.runs()]
        assert "http://127.0.0.1:1" not in tasks      # not submitted as a turn


async def test_ctrl_c_twice_to_quit(tmp_path):
    db = str(tmp_path / "h.db")
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await pilot.pause()
        exited = []
        app.exit = lambda *a, **k: exited.append(1)
        app.action_confirm_quit()
        assert exited == []                            # first ^C only arms
        app.action_confirm_quit()
        assert exited == [1]                            # second ^C (within 2s) exits


async def test_interrupt_marks_running_run_failed(tmp_path):
    db = str(tmp_path / "h.db")
    app = make_app(db, MockLlamaCpp(script={424242: chat_response(content="p")}))
    async with app.run_test() as pilot:
        await pilot.pause()
        rid = app.event_log.create_run("a turn")
        app.active = rid
        app._runs_state = [(rid, "running", "a turn")]

        class _W:
            cancelled = False
            def cancel(self):
                self.cancelled = True
        w = _W()
        app._active_worker = w
        app.action_interrupt()
        await pilot.pause()
        assert w.cancelled                              # the in-process worker was cancelled
        assert any(e.type == RUN_FAILED for e in app.event_log.events(rid))  # marked failed → rewindable
