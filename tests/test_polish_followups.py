"""Follow-ups: REPL replay-on-resume, grammar-enforced review findings, and the
softened (no-black) themes."""

from __future__ import annotations

import json

from local_harness.agent.loop import Agent
from local_harness.agent.tools import _REPL, ToolRegistry, repl
from local_harness.events.log import EventLog, TOOL_CALL
from local_harness.inference.capabilities import Capabilities
from local_harness.tui import render
from local_harness.tui.app import _findings_skill


# --- REPL replay-on-resume -------------------------------------------------

def test_repl_replay_rebuilds_state_and_guards_against_double_run():
    log = EventLog(":memory:")
    rid = log.create_run("compute")
    for code in ("acc = 0", "acc += 5"):
        log.append(rid, TOOL_CALL, {"name": "repl", "result": "[no output]",
                                    "arguments": json.dumps({"code": code, "session": "rsess"})})
    _REPL._ns.pop("rsess", None)  # simulate a fresh process — state lost
    agent = Agent(None, ToolRegistry([]), log, capabilities=Capabilities())
    try:
        agent._replay_repl(rid)
        assert repl("acc", session="rsess") == "5\n"     # state rebuilt by replay
        agent._replay_repl(rid)                          # live now → must NOT re-run
        assert repl("acc", session="rsess") == "5\n"     # still 5, no double +=5
    finally:
        _REPL._ns.pop("rsess", None)


# --- grammar-enforced review findings --------------------------------------

def test_review_findings_grammar_validates():
    skill = _findings_skill("review")
    assert skill.validate_output("src/a.py:10 — major — needs a null check")
    assert skill.validate_output("No findings.")
    assert skill.validate_output("a.py:1 — nit — x\nb.py:2 — blocker — y")
    assert not skill.validate_output("just prose with no structure at all")


def test_security_findings_use_security_severities():
    sec = _findings_skill("security-review")
    assert sec.validate_output("auth.py:5 — critical — SQL injection")
    assert not sec.validate_output("auth.py:5 — major — x")  # 'major' isn't a security severity


def test_findings_panel_renders_severities():
    import io
    from rich.console import Console
    out = io.StringIO()
    Console(width=80, file=out).print(
        render.findings_panel("src/x.py:3 — blocker — boom\nsrc/y.py:9 — nit — tidy", "review"))
    text = out.getvalue()
    assert "blocker" in text and "src/x.py:3" in text and "grammar-validated" in text


# --- softened themes -------------------------------------------------------

def test_dark_themes_avoid_pure_black():
    for name, p in render.THEMES.items():
        if not p.dark:
            continue
        r, g, b = int(p.bg[1:3], 16), int(p.bg[3:5], 16), int(p.bg[5:7], 16)
        assert max(r, g, b) >= 0x16, f"{name} background too dark/black: {p.bg}"
