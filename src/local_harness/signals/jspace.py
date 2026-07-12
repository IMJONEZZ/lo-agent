"""J-space uncertainty signal: runner-up-hypothesis rank trajectories.

The paper's yen-vs-euro pattern — a competing answer that hovers in the
workspace band before the model commits — is an uncertainty signal that lives
*below* the token distribution. This queries a lens service for a tracked
token's rank across layers at a position and summarizes how strongly a
runner-up contended.

[B]-tagged (access-ladder): a measured internal signal with stated
assumptions, not a correctness guarantee.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np


@dataclass
class RankTrajectory:
    token: int
    piece: str
    ranks: list[int]          # rank per readout layer at the target position
    min_rank: int             # closest it ever came to the top
    final_rank: int           # rank at the output layer
    contended: bool           # did it enter the workspace band (< threshold) mid-stack?

    def to_dict(self) -> dict:
        return {"token": self.token, "piece": self.piece, "ranks": self.ranks,
                "min_rank": self.min_rank, "final_rank": self.final_rank,
                "contended": self.contended}


def _decode_ranks(b64, shape):
    return np.frombuffer(base64.b64decode(b64), "<i4").reshape(shape)


async def rank_trajectories(lens_url: str, *, prompt=None, tokens=None,
                            track_pieces: list[str], pos: int = -1,
                            workspace_band: int = 200) -> list[RankTrajectory]:
    """For each piece in ``track_pieces``, its rank across layers at ``pos``.

    ``contended`` = the token dipped into the workspace band (rank <
    workspace_band) at some intermediate layer even if it lost at the output —
    the runner-up-hypothesis marker.
    """
    import httpx

    async with httpx.AsyncClient(timeout=600) as c:
        body = {"stride": 1, "top_n": 1}
        if tokens:
            body["tokens"] = tokens
        else:
            body["prompt"] = prompt
        sl = (await c.post(lens_url.rstrip("/") + "/lens/slice", json=body)).json()
        # resolve pieces → ids
        ids, pieces = [], []
        for p in track_pieces:
            res = (await c.get(lens_url.rstrip("/") + "/lens/search_tokens",
                               params={"q": p.strip(), "limit": 20})).json()["results"]
            exact = [r for r in res if r["piece"] == p] or res[:1]
            if exact:
                ids.append(exact[0]["token"])
                pieces.append(exact[0]["piece"])
        if not ids:
            return []
        rr = (await c.post(lens_url.rstrip("/") + "/lens/ranks",
                           json={"ctx_id": sl["ctx_id"], "token_ids": ids})).json()

    T, L, n = rr["shape"]
    arr = _decode_ranks(rr["ranks"], (T, L, n))
    p = pos if pos >= 0 else T - 1
    out = []
    for j, (tid, piece) in enumerate(zip(ids, pieces)):
        ranks = [int(arr[p, li, j]) for li in range(L)]
        # exclude the final layer when judging "contended mid-stack"
        mid = ranks[:-1] or ranks
        out.append(RankTrajectory(
            token=tid, piece=piece, ranks=ranks, min_rank=min(ranks),
            final_rank=ranks[-1],
            contended=(min(mid) < workspace_band and ranks[-1] >= workspace_band)))
    return out
