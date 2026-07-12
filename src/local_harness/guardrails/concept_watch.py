"""Concept-watch guardrail: surface J-space concept surges before output.

The paper's monitoring result — bug/malice/deception concepts are visible in
the J-space workspace before (or even without) surfacing in the output — made
into a guardrail. Given a watch-list of concept tokens and a lens service, it
checks whether any watched concept's rank surged into the workspace band during
a turn, and emits a NOTICE-shaped finding.

Off by default; opt-in per session. [B]-tagged (a measured internal signal).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConceptAlert:
    piece: str
    min_rank: int
    final_rank: int
    surfaced: bool          # did it reach the output too, or stay latent?

    def message(self) -> str:
        where = "in the output" if self.surfaced else "latent (workspace only)"
        return (f"J-space concept {self.piece!r} surged to rank {self.min_rank} "
                f"mid-stack — {where}")


async def watch_concepts(lens_url: str, *, prompt=None, tokens=None,
                         concepts: list[str], band: int = 100,
                         pos: int = -1) -> list[ConceptAlert]:
    """Return an alert for each watched concept that entered the workspace band."""
    from ..signals.jspace import rank_trajectories

    trajs = await rank_trajectories(lens_url, prompt=prompt, tokens=tokens,
                                    track_pieces=concepts, pos=pos, workspace_band=band)
    alerts = []
    for t in trajs:
        if t.min_rank < band:
            alerts.append(ConceptAlert(piece=t.piece, min_rank=t.min_rank,
                                       final_rank=t.final_rank,
                                       surfaced=t.final_rank < band))
    return alerts
