"""Plant one fault at a time in a copy of lo's source, and run the composability
DAG suite (tests/test_dag_composability.py) against each.

It measures whether the DAGs find real faults. Each fault is a one-line change
that breaks one item of the LO2 test plan's break list, at one step of the
round. The copy goes ahead of the installed package on PYTHONPATH, so the tests
import the broken code. The repo itself is never changed.

    uv run --group dev python tests/run_dag_faults.py [--out report.md] [FAULT ...]

Not collected by pytest (no test_ prefix). Takes a few minutes per fault.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# name -> (file under src/local_harness, the original text, the faulty text, what it breaks)
FAULTS = {
    "F1-error-not-shown": (
        "tui/render.py",
        "    if ev.type == RUN_FAILED:\n"
        "        return Text(f\"  ✗ {p.get('error')}\", style=\"bold \" + ROSE)",
        "    if ev.type == RUN_FAILED:\n        return None",
        "the chat no longer shows a failed turn's error (an error the user doesn't see)"),
    "F2-agents-md-dropped": (
        "agent/notebook.py",
        "        if self.repo_dir is None:\n            return \"\"",
        "        if True:\n            return \"\"",
        "AGENTS.md never reaches the system prompt (an incorrect output at step 1)"),
    "F3-no-end-event-on-budget": (
        "agent/loop.py",
        "            self.log.append(run_id, RUN_FAILED, {\"error\": \"max_steps exceeded\"})\n",
        "",
        "a turn that uses up its step budget logs no end event (not returning to ready)"),
    "F4-tool-result-not-sent-back": (
        "agent/loop.py",
        "                    messages.append(tool_msg)\n                fatal = guardrails.record(",
        "                fatal = guardrails.record(",
        "a tool's result isn't sent back to the model (step 2 → step 3 broken)"),
    "F5-doom-loop-cap-off": (
        "guardrails/loops.py",
        "            if n >= self.hard_cap:",
        "            if n >= self.hard_cap * 100:",
        "the doom-loop guardrail never stops a run (step 3's guardrail stop broken)"),
}


def run_fault(name: str, work: Path) -> tuple[int, int, list[tuple[str, str]]]:
    rel, old, new, _ = FAULTS[name]
    copy = work / name
    shutil.copytree(REPO / "src", copy / "src", ignore=shutil.ignore_patterns("__pycache__"))
    target = copy / "src" / "local_harness" / rel
    text = target.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{name}: the original text isn't in {rel} exactly once; update FAULTS")
    target.write_text(text.replace(old, new))
    xml = copy / "junit.xml"
    env = dict(os.environ, PYTHONPATH=str(copy / "src"))
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_dag_composability.py", f"--junitxml={xml}"],
                   cwd=REPO, env=env, capture_output=True, text=True)
    cases = [c for c in ET.parse(xml).getroot().iter("testcase")
             if c.get("name", "").startswith("test_dag[")]
    failed = []
    for case in cases:
        bad = case.find("failure") if case.find("failure") is not None else case.find("error")
        if bad is not None:
            dag = case.get("name").split("[", 1)[1].rstrip("]")
            failed.append((dag, (bad.get("message") or "").splitlines()[0][:200]))
    return len(failed), len(cases), failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("faults", nargs="*", default=list(FAULTS), help="faults to plant (default: all)")
    ap.add_argument("--out", help="write the Markdown results here as well as printing them")
    args = ap.parse_args()
    lines = ["| Fault | What it breaks | DAGs that failed | Which |", "|---|---|---|---|"]
    with tempfile.TemporaryDirectory(prefix="lo-dag-faults-") as tmp:
        for name in args.faults:
            n_failed, total, failed = run_fault(name, Path(tmp))
            which = "<br>".join(f"`{dag}`: {msg}" for dag, msg in failed) or "none"
            lines.append(f"| {name} | {FAULTS[name][3]} | {n_failed} of {total} | {which} |")
            print(f"{name}: {n_failed}/{total} DAGs failed", flush=True)
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        Path(args.out).write_text(report)


if __name__ == "__main__":
    main()
