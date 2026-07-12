"""Append-only SQLite event log — pillar A's substrate.

Every model call, tool call, and agent decision is an event row. The log is
simultaneously: the replay source, the crash-resume checkpoint, the
observability trace, and (later) the corpus for background skill induction.
Rows are never updated or deleted; run status lives in its own table.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Event types
RUN_STARTED = "run_started"
MODEL_CALL = "model_call"        # payload: {request_body, response, seed, timing_ms, logprob_summary}
TOOL_CALL = "tool_call"          # payload: {name, arguments, result, error}
POLICY_TRIGGERED = "policy_triggered"  # payload: {call_index, attempt, action, reason}
GUARDRAIL = "guardrail"          # payload: {call_index, action, kind, rescued, reason, nudge?, rescued_calls?}
RUN_COMPLETED = "run_completed"  # payload: {answer}
RUN_FAILED = "run_failed"        # payload: {error}
USER_MESSAGE = "user_message"    # payload: {content} — a follow-up turn continuing a run
CONTEXT_COMPACTED = "context_compacted"  # payload: {method, before_tokens, after_tokens, trigger_tokens, summary?}
MESSAGE_SNIPPED = "message_snipped"  # payload: {seq} — collapse that event's content in future context (lossless)
AGENT_SPAWNED = "agent_spawned"      # payload: {child_run_id, task} — a worker the lead fanned out (Coordinator)
JLENS_INTERVENTION = "jlens_intervention"  # payload: {specs, lens_hash, target} — a J-space edit set (Rung 6; replayable)


@dataclass
class Event:
    run_id: str
    seq: int
    type: str
    payload: dict[str, Any]
    created_at: float


@dataclass
class RunMeta:
    run_id: str
    task: str
    status: str  # running | completed | failed
    created_at: float
    title: str | None = None  # user-given name; falls back to task in UIs

    @property
    def label(self) -> str:
        """What to show for this run: the user's title, else the task."""
        return self.title or self.task


class EventLog:
    def __init__(self, path: str | Path):
        self.path = str(path)
        # Listeners are called with each persisted Event right after commit — the
        # EventBus registers one to broadcast persisted events to live SSE
        # subscribers, so the agent loop needs no bus awareness: it just writes
        # to the log as it always has.
        self._listeners: list[Callable[[Event], None]] = []
        # check_same_thread=False: the TUI's embedded server is built on the main
        # thread but runs (and uses this connection) on a uvicorn worker thread.
        # Each connection is still touched by a single thread after creation, so
        # this is safe; WAL handles the separate TUI-thread reader connection.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id     TEXT PRIMARY KEY,
                task       TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'running',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                run_id     TEXT NOT NULL,
                seq        INTEGER NOT NULL,
                type       TEXT NOT NULL,
                payload    TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, seq)
            );
            """
        )
        # Migration: `title` (user-given run name) arrived after v0.1.21.
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(runs)")]
        if "title" not in cols:
            self._conn.execute("ALTER TABLE runs ADD COLUMN title TEXT")
        self._conn.commit()

    def create_run(self, task: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO runs (run_id, task, status, created_at) VALUES (?, ?, 'running', ?)",
            (run_id, task, time.time()),
        )
        self._conn.commit()
        self.append(run_id, RUN_STARTED, {"task": task})
        return run_id

    def add_listener(self, fn: Callable[[Event], None]) -> None:
        """Register a callback invoked with each Event right after it commits."""
        self._listeners.append(fn)

    def append(self, run_id: str, type: str, payload: dict[str, Any]) -> int:
        cur = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE run_id = ?", (run_id,)
        )
        seq = cur.fetchone()[0]
        created_at = time.time()
        self._conn.execute(
            "INSERT INTO events (run_id, seq, type, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, seq, type, json.dumps(payload), created_at),
        )
        if type == RUN_COMPLETED:
            self._set_status(run_id, "completed")
        elif type == RUN_FAILED:
            self._set_status(run_id, "failed")
        self._conn.commit()
        if self._listeners:
            event = Event(run_id=run_id, seq=seq, type=type, payload=payload,
                          created_at=created_at)
            for fn in self._listeners:
                fn(event)
        return seq

    def events(self, run_id: str, type: str | None = None) -> list[Event]:
        q = "SELECT run_id, seq, type, payload, created_at FROM events WHERE run_id = ?"
        params: list[Any] = [run_id]
        if type is not None:
            q += " AND type = ?"
            params.append(type)
        q += " ORDER BY seq"
        return [
            Event(run_id=r[0], seq=r[1], type=r[2], payload=json.loads(r[3]), created_at=r[4])
            for r in self._conn.execute(q, params)
        ]

    def run(self, run_id: str) -> RunMeta | None:
        row = self._conn.execute(
            "SELECT run_id, task, status, created_at, title FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return RunMeta(*row) if row else None

    def event_count(self, run_id: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()[0]

    def runs(self) -> list[RunMeta]:
        return [
            RunMeta(*row)
            for row in self._conn.execute(
                "SELECT run_id, task, status, created_at, title FROM runs "
                "ORDER BY created_at"
            )
        ]

    def rename_run(self, run_id: str, title: str) -> None:
        """Give a run a human name (shown instead of the task in listings).
        An empty title clears the name back to the task."""
        self._conn.execute(
            "UPDATE runs SET title = ? WHERE run_id = ?",
            (title.strip() or None, run_id),
        )
        self._conn.commit()

    def delete_events_from(self, run_id: str, from_seq: int) -> None:
        """Drop events at or after a sequence number (used by /undo to remove
        the last turn). A user-initiated edit of conversation history."""
        self._conn.execute(
            "DELETE FROM events WHERE run_id = ? AND seq >= ?", (run_id, from_seq))
        self._conn.commit()

    def rewind_points(self, run_id: str) -> list[tuple[int, str, str]]:
        """Candidate /rewind boundaries as (seq, kind, preview), earliest first:
        the original answer (first model call), then each follow-up user turn.
        Selecting one means 'remove from here onward'."""
        events = self.events(run_id)
        points: list[tuple[int, str, str]] = []
        first_call = next((e for e in events if e.type == MODEL_CALL), None)
        if first_call is not None:
            meta = self.run(run_id)
            points.append((first_call.seq, "answer", (meta.task if meta else "")[:60]))
        for e in events:
            if e.type == USER_MESSAGE:
                points.append((e.seq, "follow-up", (e.payload.get("content") or "")[:60]))
        return points

    def rewind(self, run_id: str, from_seq: int) -> str | None:
        """Roll a conversation back to before `from_seq`: archive the removed tail
        into a fresh run (so the rewind is itself reversible — lossless), then drop
        it from `run_id`. Returns the archive run_id, or None if nothing to remove."""
        removed = [e for e in self.events(run_id) if e.seq >= from_seq]
        if not removed:
            return None
        meta = self.run(run_id)
        archive_id = uuid.uuid4().hex[:12]
        self.ensure_run(archive_id, f"[rewound from {run_id[:8]}] {meta.task if meta else ''}"[:120])
        for e in removed:
            self.import_event(archive_id, e.type, e.payload)
        self.delete_events_from(run_id, from_seq)
        return archive_id

    def delete_run(self, run_id: str) -> None:
        """Remove a run and all its events — a user-initiated history action
        (distinct from the append-only event stream during a live run)."""
        self._conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()

    def reopen(self, run_id: str) -> None:
        """Mark a finished run active again (continuing the conversation)."""
        self._set_status(run_id, "running")
        self._conn.commit()

    def ensure_run(self, run_id: str, task: str = "") -> None:
        """Insert a run row with a GIVEN id if absent — used by the TUI's
        `--server` client to mirror a server-assigned run into its local log."""
        if self._conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone():
            return
        self._conn.execute(
            "INSERT INTO runs (run_id, task, status, created_at) VALUES (?, ?, 'running', ?)",
            (run_id, task, time.time()))
        self._conn.commit()

    def import_event(self, run_id: str, type: str, payload: dict[str, Any]) -> int:
        """Append a persisted event received from a remote server (mirroring).
        Identical to append; named for intent at the call site."""
        return self.append(run_id, type, payload)

    def _set_status(self, run_id: str, status: str) -> None:
        self._conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def close(self) -> None:
        self._conn.close()
