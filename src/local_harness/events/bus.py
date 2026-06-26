"""Pub/sub over the event log — the substrate for the OpenCode-style server.

OpenCode's UX comes from a headless server with a pub/sub bus: many clients
(TUI, web, `lo tail`) subscribe over SSE and all observe one live session.
Our `EventLog` is already the append-only store; this layers a broadcaster on top.

Two channels, and the split is the whole point:

  PERSISTENT events  — anything written to the EventLog. The bus registers a log
    listener, so the agent loop needs NO bus awareness: it writes to the log as
    it always has, and subscribers get those events live. They're also replayable
    (pillar A) and reach late subscribers via catch-up.
  EPHEMERAL events   — TOKEN_DELTA / REASONING_DELTA / TOOL_PROGRESS, broadcast to
    live subscribers ONLY via `publish_delta`, never written to SQLite. So the
    live token stream reaches clients without polluting the replayable log.

A subscriber registers its queue BEFORE reading catch-up, so nothing published
mid-catch-up is lost; persisted events already seen in catch-up are de-duped by seq.
All subscribers live in the server process (the one that owns the EventLog);
remote clients attach over SSE, not by opening the DB.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from .log import RUN_COMPLETED, RUN_FAILED, Event, EventLog

# Ephemeral event types — broadcast live, never persisted, seq is always -1.
TOKEN_DELTA = "token_delta"  # payload: {text}
REASONING_DELTA = "reasoning_delta"  # payload: {text}
TOOL_PROGRESS = "tool_progress"  # payload: {name, phase}  (start|done)
NOTICE = "notice"  # payload: {message} — a one-off human-readable hint
PERMISSION_REQUEST = "permission_request"  # payload: {request_id, tool, arguments} — ask the client to approve a tool

# The terminal event types a one-shot follow should stop on.
TERMINAL = {RUN_COMPLETED, RUN_FAILED}


class EventBus:
    def __init__(self, log: EventLog):
        self.log = log
        self._subs: dict[str, set[asyncio.Queue]] = {}
        log.add_listener(self._broadcast)  # persisted events flow through here

    # --- publish ---------------------------------------------------------

    def create_run(self, task: str) -> str:
        """Create a run; its RUN_STARTED broadcasts via the log listener."""
        return self.log.create_run(task)

    def publish_delta(self, run_id: str, type: str, payload: dict) -> None:
        """Broadcast an EPHEMERAL event (token/reasoning/tool progress) to live
        subscribers only — never persisted, so replay stays clean."""
        self._broadcast(
            Event(
                run_id=run_id,
                seq=-1,
                type=type,
                payload=payload,
                created_at=time.time(),
            )
        )

    def _broadcast(self, ev: Event) -> None:
        for q in list(self._subs.get(ev.run_id, ())):
            q.put_nowait(ev)

    # --- subscribe -------------------------------------------------------

    async def subscribe(
        self,
        run_id: str,
        *,
        replay: bool = True,
        stop_on: set[str] | None = None,
    ) -> AsyncIterator[Event]:
        """Yield events for a run: catch-up from seq 0 (if replay), then live.

        `stop_on` ends the stream after yielding an event of one of those types
        (e.g. TERMINAL for a one-shot follow). The queue is registered before
        catch-up is read, so events arriving during catch-up are buffered and
        delivered after; any already seen in catch-up are skipped by seq."""
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(run_id, set()).add(q)
        try:
            last_seq = -1
            if replay:
                for ev in self.log.events(run_id):
                    last_seq = ev.seq
                    yield ev
                    if stop_on and ev.type in stop_on:
                        return
            while True:
                ev = await q.get()
                if ev.seq >= 0 and ev.seq <= last_seq:
                    continue  # already delivered during catch-up
                yield ev
                if stop_on and ev.type in stop_on:
                    return
        finally:
            subs = self._subs.get(run_id)
            if subs is not None:
                subs.discard(q)
                if not subs:
                    self._subs.pop(run_id, None)

    def subscriber_count(self, run_id: str) -> int:
        return len(self._subs.get(run_id, ()))
