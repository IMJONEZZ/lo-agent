"""J-space intervention specs + named concept sets (Rung 6, write side).

A spec is the UI/CLI-level description the lens service translates into native
residual edits. Position-range doctrine (found live on qwen3.6): STEER defaults
to prompt-positions-only; ABLATE/SWAP default to all positions. Named sets are
persisted under ``profiles/`` (the BiasProfileStore pattern) so a concept edit
is reusable and recordable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Spec:
    """One J-space edit. `token`/`token_b` are display pieces; `token_id`/
    `token_b_id` the ids the service actually uses (resolved via search_tokens)."""
    type: str                     # 'steer' | 'ablate' | 'swap'
    token_id: int | None = None
    token: str | None = None      # display piece (informational)
    token_b_id: int | None = None  # swap only
    token_b: str | None = None
    alpha: float = 2.0            # steer only
    layers: list[int] | None = None   # [lo, hi] inclusive; None = all fitted
    pos: list[int] | None = None      # [start, end]; None = doctrine default

    def to_service(self) -> dict:
        """The dict the lens service /lens/* endpoints expect."""
        d: dict = {"type": self.type}
        if self.layers is not None:
            d["layers"] = self.layers
        if self.pos is not None:
            d["pos"] = self.pos
        if self.type == "swap":
            d["token_a"] = self.token_id
            d["token_b"] = self.token_b_id
        else:
            d["token_id"] = self.token_id
            if self.type == "steer":
                d["alpha"] = self.alpha
        return d

    def describe(self) -> str:
        if self.type == "swap":
            return f"swap {self.token!r}⇄{self.token_b!r}"
        if self.type == "steer":
            return f"steer {self.token!r} α={self.alpha}"
        return f"ablate {self.token!r}"


@dataclass
class ConceptSet:
    """A named, reusable collection of specs."""
    name: str
    specs: list[Spec] = field(default_factory=list)

    def to_service(self) -> list[dict]:
        return [s.to_service() for s in self.specs]

    def to_dict(self) -> dict:
        return {"name": self.name, "specs": [asdict(s) for s in self.specs]}

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptSet":
        return cls(name=d["name"], specs=[Spec(**s) for s in d.get("specs", [])])


class ConceptStore:
    """Named concept sets on disk under ``<profiles_dir>/jlens``."""

    def __init__(self, profiles_dir: str | Path = "profiles"):
        self.dir = Path(profiles_dir) / "jlens"

    def _path(self, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in "-_") or "unnamed"
        return self.dir / f"{safe}.json"

    def save(self, cs: ConceptSet) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._path(cs.name)
        p.write_text(json.dumps(cs.to_dict(), indent=2))
        return p

    def load(self, name: str) -> ConceptSet:
        p = self._path(name)
        if not p.is_file():
            raise FileNotFoundError(f"no concept set {name!r} in {self.dir}")
        return ConceptSet.from_dict(json.loads(p.read_text()))

    def list(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))


def lens_hash(lens_path: str | None) -> str:
    """A short content hash of the lens file so intervention events are
    attributable to the exact lens used (replay determinism)."""
    import hashlib

    if not lens_path or not Path(lens_path).is_file():
        return "identity"
    h = hashlib.sha256()
    with open(lens_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
