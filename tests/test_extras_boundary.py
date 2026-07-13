"""The lens extra (numpy/gguf) is optional — a base install (e.g. Homebrew
before 0.2.2_1, `pip install lo-agent`) must never see a raw
ModuleNotFoundError. This walks every module with numpy/gguf blocked in a
subprocess and pins which modules are ALLOWED to need them; anything new that
grows a top-level numpy import outside that set fails here, not on a user's
box."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

# Modules whose top-level imports may legitimately require the lens extra.
# The deal: nothing on the agent/TUI/CLI startup path, and every user-facing
# entry point into these guards with a friendly install hint first.
ALLOWED_LENS_ONLY = (
    "local_harness.jlens.",  # math/service/fitting submodules (pkg init is lazy)
    "local_harness.signals.jspace",
    "local_harness.tui.lens_screen",
)

# Entry points that must import cleanly on a base install.
MUST_IMPORT = [
    "local_harness.cli.main",
    "local_harness.agent.loop",
    "local_harness.agent.tools",
    "local_harness.tui.app",
    "local_harness.jlens",
    "local_harness.jlens.cli",
    "local_harness.jlens.manager",
    "local_harness.guardrails.concept_watch",
]

_SWEEP = r"""
import importlib, importlib.abc, json, pkgutil, sys

BLOCKED = ("numpy", "gguf")

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, Blocker())

import local_harness

failed = {}
ok = []
for m in pkgutil.walk_packages(local_harness.__path__, prefix="local_harness."):
    if "__main__" in m.name:
        continue
    try:
        importlib.import_module(m.name)
        ok.append(m.name)
    except ModuleNotFoundError as e:
        failed[m.name] = e.name or str(e)
    except Exception as e:  # any other import-time crash is also a finding
        failed[m.name] = f"{type(e).__name__}: {e}"

# the CLI guard must exit with the install hint, not a traceback
from local_harness.jlens import cli
from types import SimpleNamespace
guard = None
try:
    cli.run(SimpleNamespace(lens_action="fit", hf=None))
except SystemExit as e:
    guard = str(e)

print(json.dumps({"failed": failed, "ok": ok, "guard": guard}))
"""


@pytest.fixture(scope="module")
def sweep():
    r = subprocess.run([sys.executable, "-c", _SWEEP], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, f"sweep subprocess died:\n{r.stderr[-2000:]}"
    return json.loads(r.stdout.splitlines()[-1])


def test_entry_points_import_without_lens_extra(sweep):
    missing = [m for m in MUST_IMPORT if m not in sweep["ok"]]
    assert not missing, (
        f"base-install entry points failed to import: "
        f"{ {m: sweep['failed'].get(m) for m in missing} }")


def test_only_known_modules_need_lens_extra(sweep):
    offenders = {
        mod: why for mod, why in sweep["failed"].items()
        if why in ("numpy", "gguf")
        and not mod.startswith(ALLOWED_LENS_ONLY[0])
        and mod not in ALLOWED_LENS_ONLY[1:]
    }
    assert not offenders, (
        f"modules newly requiring numpy/gguf at import time: {offenders}. "
        "Either lazy-import inside the function and guard the entry point "
        "(see jlens.cli.run), or add to ALLOWED_LENS_ONLY consciously.")


def test_lens_cli_guard_message(sweep):
    guard = sweep["guard"]
    assert guard, "`lo lens fit` without numpy did not exit via the guard"
    assert "lens" in guard and "uv" in guard, f"unhelpful guard message: {guard!r}"
    assert "Traceback" not in guard
