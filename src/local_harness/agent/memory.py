"""Cross-session memory: SQLite FTS5 full-text recall (no embedding model
required — works offline against any backend)."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryEntry:
    id: int
    kind: str   # episode | lesson | note | skill
    text: str
    run_id: str | None
    created_at: float
    # Agreement fraction in [0,1]: how often the model produced this same answer
    # across K resamples when it was written. This is a sample-consistency signal
    # (semantic-entropy-lite), NOT a token logprob — a fact the model agrees with
    # itself on is worth keeping; one it doesn't isn't. (The SQL column is named
    # `confidence` for backward compatibility with existing local memory DBs.)
    agreement: float | None = None


class Memory:
    def __init__(self, path: str | Path):
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
                kind, text, run_id, created_at UNINDEXED, confidence UNINDEXED
            );
            """
        )
        self._conn.commit()

    def store(self, kind: str, text: str, run_id: str | None = None,
              agreement: float | None = None) -> None:
        self._conn.execute(
            "INSERT INTO memories (kind, text, run_id, created_at, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (kind, text, run_id, time.time(), agreement),
        )
        self._conn.commit()

    def recall(self, query: str, limit: int = 5, kind: str | None = None) -> list[MemoryEntry]:
        words = re.findall(r"[A-Za-z0-9_]+", query)
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        q = ("SELECT rowid, kind, text, run_id, created_at, confidence "
             "FROM memories WHERE memories MATCH ?")
        params: list = [match]
        if kind:
            q += " AND kind = ?"
            params.append(kind)
        q += " ORDER BY bm25(memories) LIMIT ?"
        params.append(limit)
        return [MemoryEntry(*row) for row in self._conn.execute(q, params)]

    def has_run(self, run_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM memories WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone()
        return row is not None

    def count(self, kind: str | None = None) -> int:
        if kind:
            return self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE kind = ?", (kind,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
