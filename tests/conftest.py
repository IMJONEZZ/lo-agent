# Ensures tests/ is importable (for mocks.py) under pytest's rootdir insertion.

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """When a user-simulation step fails, save the full screen text to
    test-artifacts/<test>.screen.txt so the failure is diagnosable at a glance."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed or call.excinfo is None:
        return
    exc = call.excinfo.value
    screen = getattr(exc, "screen", None)
    if not screen:
        return
    from pathlib import Path

    art = Path(item.config.rootpath) / "test-artifacts"
    art.mkdir(exist_ok=True)
    path = art / f"{item.name}.screen.txt"
    path.write_text(screen)
    report.sections.append(("simulated screen", f"saved to {path}"))
