"""Detect overused LLM 'slop' rhetorical structures (see docs/slop-structures.md).

Surface-regex (R-tier) patterns only, curated for precision — a few false
positives that get rewritten are acceptable, but we avoid the high-FP templates
(bare "from X to Y", "that said", rule-of-three) that need real parsing.

Feeds the anti-slop loop: scan emitted text, and on a match drive the same
rewind/resample the fixed-phrase path uses (logits/antislop.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# name -> case-insensitive pattern. Each is a structure that is almost always
# slop in the doubled/closed form matched here (not the bare contrastive "but").
_PATTERNS: dict[str, str] = {
    # A. negation–antithesis
    "not_x_its_y": r"\b(?:it'?s|that'?s|this is) not\b[^.?!]{1,40}?[,—]\s*(?:it'?s|that'?s|this is)\b",
    "not_just_but": r"\bnot just\b[^.?!]{1,40}?\bbut\b",
    "not_only_but": r"\bnot only\b[^.?!]{1,40}?\bbut(?:\s+also)?\b",
    "not_about_about": r"\bnot about\b[^.?!]{1,40}?\b(?:it'?s|it is) about\b",
    "no_x_no_y_just": r"\bno \w+,\s*no \w+[,—]?\s*(?:just|only)\b",
    "less_more": r"\bless \w+,\s*more \w+\b",
    # E. throat-clearing openers
    "heres_the": r"\bhere'?s (?:the thing|the kicker|why)\b",
    "worth_noting": r"\b(?:it'?s worth (?:noting|remembering)|it'?s important to note)\b",
    "the_truth_is": r"\bthe (?:uncomfortable|hard|simple|honest) truth is\b",
    "in_a_world": r"\bin (?:a world where|today'?s [\w-]+ world)\b",
    "end_of_day": r"\bat the end of the day\b",
    # D. em-dash trailing participle
    "em_dash_participle": r"—\s*(?:ensuring|allowing|enabling|empowering)\b",
    "easier_than_ever": r"\b(?:easier|simpler|faster|smarter|better) than ever\b",
    # H. product-copy reframes
    "x_meets_y": r"\bwhere \w+ meets \w+\b",
    "more_than_just": r"\bmore than just (?:a |an )?\w+\b",
    "say_goodbye": r"\b(?:say goodbye to|gone are the days of)\b",
    # F. scope-sweeper
    "whether_youre": r"\bwhether you'?re (?:a |an )?\w+ or (?:a |an )?\w+\b",
    # I. CTA closers
    "possibilities_endless": r"\bpossibilities are endless\b",
    "game_changer": r"\bgame[- ]changer\b",
    "future_is_here": r"\bfuture of \w+ is here\b",
}

_COMPILED = {name: re.compile(pat, re.IGNORECASE) for name, pat in _PATTERNS.items()}


@dataclass
class SlopMatch:
    name: str        # which structure, e.g. "not_x_its_y"
    text: str        # the matched span
    start: int
    end: int


class SlopDetector:
    """Scan text for slop structures. `names` selects a subset (default: all)."""

    def __init__(self, names: list[str] | None = None):
        self._pats = {n: _COMPILED[n] for n in (names or _COMPILED)}

    @property
    def pattern_names(self) -> list[str]:
        return list(self._pats)

    def scan(self, text: str) -> list[SlopMatch]:
        out: list[SlopMatch] = []
        for name, rx in self._pats.items():
            for m in rx.finditer(text):
                out.append(SlopMatch(name, m.group(0), m.start(), m.end()))
        return sorted(out, key=lambda s: s.start)

    def first_match(self, text: str) -> SlopMatch | None:
        """Earliest slop occurrence — the rewind point for the anti-slop loop."""
        matches = self.scan(text)
        return matches[0] if matches else None
