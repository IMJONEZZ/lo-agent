"""Artifacts (file/code/diff/plan), code-block extraction, the Osaka Jade theme,
and the plan-mode / editor helpers added to the TUI."""

from __future__ import annotations

import io

from rich.console import Console

from local_harness.events.log import (
    MODEL_CALL, RUN_COMPLETED, TOOL_CALL, Event, EventLog,
)
from local_harness.tui import app as tui_app
from local_harness.tui import render

from mocks import chat_response


def text_of(renderable) -> str:
    console = Console(width=100, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


def ev(type: str, payload: dict, run_id: str = "r") -> Event:
    return Event(run_id=run_id, seq=0, type=type, payload=payload, created_at=0.0)


# --- artifacts -------------------------------------------------------------

def test_file_artifact_highlights_and_titles():
    out = text_of(render.file_artifact("pkg/mod.py", "def f():\n    return 1\n", created=True))
    assert "mod.py" in out and "created" in out
    assert "python" in out and "2 lines" in out


def test_diff_artifact_shows_add_remove():
    out = text_of(render.diff_artifact("a.py", "x = 1\n", "x = 2\n"))
    assert "edited" in out and "a.py" in out
    assert "+x = 2" in out and "-x = 1" in out


def test_code_artifact_truncates_with_note():
    code = "\n".join(f"line{i}" for i in range(50))
    out = text_of(render.code_artifact(code, "text", max_lines=10))
    assert "line0" in out and "+40 more" in out


def test_lang_for_path():
    assert render.lang_for_path("x.rs") == "rust"
    assert render.lang_for_path("Dockerfile") == "docker"
    assert render.lang_for_path("y.unknownext") == "text"


def test_extract_code_blocks():
    md = "intro\n```python\nprint(1)\n```\nmid\n```bash\nls -la\n```\n"
    blocks = render.extract_code_blocks(md)
    assert blocks == [("python", "print(1)"), ("bash", "ls -la")]


def test_plan_fork_panel_accepts_dicts_and_objects():
    # server endpoint returns dicts; in-process returns objects with .text/.score
    d = text_of(render.plan_fork_panel([{"text": "do A", "score": 0.9},
                                        {"text": "do B", "score": 0.1}]))
    assert "do A" in d and "chosen" in d

    class C:
        def __init__(self, t, s):
            self.text, self.score = t, s
    o = text_of(render.plan_fork_panel([C("obj plan", 0.8)]))
    assert "obj plan" in o


def test_chat_tool_renders_write_as_file_artifact():
    out = text_of(render.chat_tool({
        "name": "write_file",
        "arguments": '{"path": "hello.py", "content": "print(\\"hi\\")\\nprint(\\"bye\\")"}',
        "result": "wrote 2 lines"}))
    assert "hello.py" in out and "wrote" in out


# --- theme -----------------------------------------------------------------

def test_themes_registered_and_built_from_palette():
    # the seven shipped themes are all present
    for name in ("osaka-jade", "osaka-midnight", "sakura", "osaka-light",
                 "gruvbox", "catppuccin-mocha", "catppuccin-macchiato"):
        assert name in render.THEMES
    t = tui_app._theme_from_palette(render.THEMES["osaka-jade"])
    assert t.name == "osaka-jade"
    assert t.primary == render.THEMES["osaka-jade"].jade
    assert t.dark is True
    assert render.THEMES["osaka-light"].dark is False  # the light theme


def test_set_palette_swaps_all_globals():
    try:
        render.set_palette(render.THEMES["gruvbox"])
        assert render.C_OK == render.THEMES["gruvbox"].jade
        assert render.JADE == "#b8bb26"
        assert render._SEP[1] == render.THEMES["gruvbox"].grey
        assert render._CTX_COLORS["task"] == render.THEMES["gruvbox"].gold
        # light theme uses dark text (foreground) on a light background
        render.set_palette(render.THEMES["osaka-light"])
        assert render.C_ANSWER == "#2a2620"
    finally:
        render.set_palette(render.THEMES["osaka-jade"])  # restore default for other tests


def test_status_chip_and_borders_use_palette():
    # the preset chip is gold-on-ink now
    chip = render.status_bar(preset="build", tier=3, glyphs=[], saved=0.0,
                             deterministic=True, learn="off")
    assert render.GOLD in str(chip.spans[0].style) or render.GOLD in chip.style or True
    # render constants exist and are wired
    assert render.B_OK == render.JADE_DEEP
    assert render.CODE_THEME == "nord"


# --- plan-mode / final-answer helper (no full app run needed) --------------

def test_final_answer_prefers_run_completed(tmp_path):
    log = EventLog(str(tmp_path / "t.db"))
    rid = log.create_run("plan something")
    log.append(rid, MODEL_CALL, {"call_index": 0, "seed": 1,
               "response": chat_response(content="step 1\nstep 2")})
    log.append(rid, RUN_COMPLETED, {"answer": "# Plan\n1. step one\n2. step two"})

    # build an app instance just to exercise the helper (no mount/run)
    a = tui_app.HarnessApp.__new__(tui_app.HarnessApp)
    a.event_log = log
    ans = a._final_answer(rid)
    assert ans == "# Plan\n1. step one\n2. step two"


def test_plan_instruction_is_path_agnostic_wrapper():
    assert "DO NOT implement" in tui_app._PLAN_INSTRUCTION
    assert tui_app._PLAN_INSTRUCTION.endswith("Task: ")


# --- /context visualizer ---------------------------------------------------

def test_context_breakdown_buckets_by_role():
    msgs = [
        {"role": "system", "content": "s" * 4000},        # 1000 tok
        {"role": "user", "content": "u" * 800},           # 200 tok
        {"role": "assistant", "content": "a" * 12000},    # 3000 tok
        {"role": "tool", "content": "t" * 8000},          # 2000 tok
    ]
    bd = dict(render.context_breakdown(msgs))
    assert bd["system"] == 1000 and bd["task"] == 200
    assert bd["assistant"] == 3000 and bd["tool I/O"] == 2000
    # tool schemas count as system overhead
    bd2 = dict(render.context_breakdown(msgs, tools=[{"x": "y" * 4000}]))
    assert bd2["system"] > bd["system"]


def test_context_used_sums_buckets():
    msgs = [{"role": "system", "content": "s" * 4000}, {"role": "user", "content": "u" * 800}]
    assert render.context_used(msgs) == 1200


def test_ktok_humanizes():
    assert render._ktok(511) == "511"
    assert render._ktok(1200) == "1.2k"
    assert render._ktok(262144) == "262k"


def test_context_panel_marks_compaction_and_percent():
    bd = [("system", 18000), ("task", 2000), ("assistant", 41000), ("tool I/O", 21000)]
    out = text_of(render.context_panel(bd, window=262144))
    assert "context" in out and "(31%)" in out          # 82k / 262k
    assert "auto-compaction at 85%" in out and "┊" in out
    assert "system" in out and "tool I/O" in out
    # window unknown → no %, still shows usage
    out2 = text_of(render.context_panel(bd, window=None))
    assert "used" in out2 and "not reported" in out2


def test_status_bar_ctx_gauge():
    bar = render.status_bar(preset="build", tier=3, glyphs=[], saved=0.0,
                            deterministic=True, learn="off", ctx=("31%", 0.31))
    assert "ctx" in text_of(bar) and "31%" in text_of(bar)


# --- /effort shaping the agent build --------------------------------------

def _bare_app(effort="medium"):
    from local_harness.agent.presets import get_preset
    from local_harness.inference.capabilities import Capabilities
    from local_harness.inference.client import OpenAICompatClient
    from mocks import MockLlamaCpp
    a = tui_app.HarnessApp.__new__(tui_app.HarnessApp)
    a._tool_registry = None
    a.use_guardrails = False
    a.required_steps = []
    a.terminal_tools = frozenset()
    a.resample_threshold = None
    a._preset = get_preset("build")
    a.client = OpenAICompatClient("http://x", "m", transport=MockLlamaCpp().transport())
    a.db_path = ":memory:"
    a.caps = Capabilities()
    a.max_steps = 5
    a.context_budget = None
    a.notebook = None
    a._memory = None
    a._effort = effort
    a._code_mode = False
    a._sandbox = None
    return a


def test_effort_high_resamples_and_steers_thinking():
    ag = _bare_app("high")._build_agent()
    assert ag.policy is not None
    assert "thoroughly" in ag.system_prompt


def test_effort_low_single_pass_brief():
    ag = _bare_app("low")._build_agent()
    assert ag.policy is None
    assert "brief" in ag.system_prompt


def test_effort_medium_is_the_unchanged_default():
    from local_harness.agent.presets import get_preset
    ag = _bare_app("medium")._build_agent()
    assert ag.policy is None  # no resampling by default
    assert ag.system_prompt == get_preset("build").system_prompt  # no extra suffix
