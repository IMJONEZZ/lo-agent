"""User-simulation layer: scripted user journeys driven through the real TUI.

One scenario vocabulary (`Step`/`Scenario` in `scenario.py`), multiple drivers:
Textual's Pilot headlessly (tests/e2e_support.py), a PTY subprocess
(`pty_driver.py`), and the latter pointed at a live fleet box. The catalog of
journeys lives in `scenarios.py`; `lo simulate` runs them from the CLI.
"""

from .scenario import Driver, Scenario, ScenarioFailure, Step, run_scenario  # noqa: F401
from .scenarios import SCENARIOS  # noqa: F401
