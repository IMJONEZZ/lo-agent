"""Deterministic replay: re-issue every logged model call with its recorded
seed and params, then compare outputs.

On a Tier-1+ server (verified seed support) replay is bit-identical and the
transcript hashes match. Mismatches mean the server, model, or sampler state
changed — exactly what you want surfaced before trusting a regression test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..inference.client import OpenAICompatClient
from ..inference.types import canonical_text
from .log import MODEL_CALL, EventLog


def transcript_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class Mismatch:
    seq: int
    original: str
    replayed: str


@dataclass
class ReplayReport:
    run_id: str
    total: int = 0
    matched: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    original_hash: str = ""
    replay_hash: str = ""

    @property
    def identical(self) -> bool:
        return self.total > 0 and self.matched == self.total

    def summary(self) -> str:
        status = "IDENTICAL" if self.identical else "DIVERGED"
        lines = [
            f"replay {self.run_id}: {status} ({self.matched}/{self.total} model calls matched)",
            f"  original hash: {self.original_hash}",
            f"  replay hash:   {self.replay_hash}",
        ]
        for m in self.mismatches[:5]:
            lines.append(f"  seq {m.seq}: {m.original[:60]!r} != {m.replayed[:60]!r}")
        return "\n".join(lines)


def _response_text(payload: dict) -> str:
    return canonical_text(payload["response"]["choices"][0]["message"])


async def replay_run(log: EventLog, run_id: str, client: OpenAICompatClient) -> ReplayReport:
    report = ReplayReport(run_id=run_id)
    originals: list[str] = []
    replays: list[str] = []

    for event in log.events(run_id, type=MODEL_CALL):
        body = event.payload["request_body"]
        original = _response_text(event.payload)
        response = await client.chat_body(body)
        replayed = _response_text({"response": response.raw})

        report.total += 1
        originals.append(original)
        replays.append(replayed)
        if original == replayed:
            report.matched += 1
        else:
            report.mismatches.append(Mismatch(seq=event.seq, original=original, replayed=replayed))

    report.original_hash = transcript_hash(originals)
    report.replay_hash = transcript_hash(replays)
    return report
