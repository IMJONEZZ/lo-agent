"""Headless session server — the OpenCode-parity layer.

`SessionManager` owns agent runs and drives them against an `EventBus`; clients
(TUI, `harness tail`, a future web view) subscribe to the bus over SSE and all
observe one live session. The manager is HTTP-free and unit-testable on its own;
`server.app.create_server_app` wraps it in a Starlette app.
"""

from .sessions import SessionManager

__all__ = ["SessionManager"]
