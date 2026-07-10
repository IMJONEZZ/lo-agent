"""The scenario DSL — "type X, expect Y on screen" — shared by every driver.

A `Step` is one user action plus what the user should see afterwards. The
marker technique comes from demos/tui_record.py: Textual redraws continuously,
so we never wait for quiet output — we wait for a distinctive substring to
appear in the screen text produced *after* the step's keys were sent, then let
the panel settle. Markers must be words from the RESPONSE, not the prompt
(the echoed input matches instantly otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class Step:
    keys: str = ""  # raw keystrokes; "\r" = enter, "\x1b" = escape
    marker: str | None = None  # case-insensitive substring that must appear
    absent: str | None = None  # substring that must NOT appear after settle
    timeout: float = 20.0
    settle: float = 0.0  # extra seconds to let the screen finish painting
    label: str = ""
    # Pilot-only hooks (the PTY driver skips both):
    check: Callable[[Any], None] | None = None  # arbitrary assertions on the app
    on_permission: str | None = None  # "allow" | "deny" an expected modal


@dataclass
class Scenario:
    name: str
    steps: list[Step]
    # Capability tags a target must satisfy: "mock-only" (needs a scripted
    # model), "live-ok" (safe against a real endpoint), "interactive-permissions",
    # "fs" (reads/writes the cwd). Drivers and `lo simulate` filter on these.
    needs: set[str] = field(default_factory=set)
    description: str = ""

    @property
    def live_ok(self) -> bool:
        return "live-ok" in self.needs and "mock-only" not in self.needs


class ScenarioFailure(AssertionError):
    """A step's expectation was not met; carries the screen for diagnosis."""

    def __init__(self, scenario: str, step: Step, reason: str, screen: str):
        self.scenario, self.step, self.reason, self.screen = (
            scenario,
            step,
            reason,
            screen,
        )
        label = step.label or step.keys.strip() or "<no keys>"
        super().__init__(
            f"[{scenario}] step {label!r}: {reason}\n--- screen ---\n{screen}"
        )


@runtime_checkable
class Driver(Protocol):
    """What a scenario needs from whatever is holding the TUI."""

    async def send(self, keys: str) -> None: ...

    def mark(self) -> int:
        """Offset into the output stream; markers are searched after it."""
        ...

    async def wait_marker(self, marker: str, timeout: float, since: int) -> bool: ...

    def new_text(self, since: int) -> str:
        """Plain screen text produced since `since`."""
        ...

    async def settle(self, seconds: float) -> None: ...

    def dump(self) -> str:
        """Current screen as plain text, for failure messages."""
        ...


async def run_scenario(driver: Driver, scenario: Scenario, on_step=None) -> None:
    """Play every step; raise ScenarioFailure (with the screen) on the first miss.

    Drivers may optionally expose `app` (for Step.check) and
    `answer_permission(allow) -> awaitable` (for Step.on_permission); steps
    using those hooks are skipped on drivers without them.
    `on_step(step, elapsed_seconds)` is called after each successful step.
    """
    from time import monotonic

    for step in scenario.steps:
        t0 = monotonic()
        since = driver.mark()
        if step.keys:
            await driver.send(step.keys)
        else:
            # A keyless step asserts on the whole session so far ("the welcome
            # card is on screen"), not just on output provoked by a keypress.
            since = 0
        if step.on_permission is not None and hasattr(driver, "answer_permission"):
            ok = await driver.answer_permission(step.on_permission == "allow")
            if not ok:
                raise ScenarioFailure(
                    scenario.name, step, "expected a permission modal; none appeared",
                    driver.dump(),
                )
        if step.marker is not None:
            if not await driver.wait_marker(step.marker, step.timeout, since):
                raise ScenarioFailure(
                    scenario.name,
                    step,
                    f"marker {step.marker!r} not seen within {step.timeout}s",
                    driver.dump(),
                )
        if step.settle:
            await driver.settle(step.settle)
        if step.absent is not None:
            if step.absent.lower() in driver.new_text(since).lower():
                raise ScenarioFailure(
                    scenario.name,
                    step,
                    f"forbidden text {step.absent!r} appeared",
                    driver.dump(),
                )
        if step.check is not None and getattr(driver, "app", None) is not None:
            try:
                step.check(driver.app)
            except AssertionError as e:
                raise ScenarioFailure(
                    scenario.name, step, f"check failed: {e}", driver.dump()
                ) from e
        if on_step is not None:
            on_step(step, monotonic() - t0)
