"""Grammar-structured memory facts: `subject | predicate | object`.

A fact written through `fact_grammar()` is valid by construction (constrained
decoding), so structured memory stays machine-queryable and consistent instead
of being freeform prose to re-parse. `parse_fact` recovers the fields from a
stored line; store such facts in Memory with kind='fact'.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..skills.ir import Grammar


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str

    def format(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"


def parse_fact(text: str) -> Fact | None:
    parts = [p.strip() for p in text.split("|")]
    if len(parts) != 3 or any(not p for p in parts):
        return None
    return Fact(*parts)


def fact_grammar() -> Grammar:
    """`subject | predicate | object` — three '|'-separated non-empty fields."""
    return Grammar.from_rules(
        {"fact": 'field "|" field "|" field', "field": r"/[^|\n]+/"},
        root="fact",
    )
