"""Guidance/Outlines-style builders over the harness grammar IR.

Both guidance (`select`, `one_or_more`, `gen`, …) and Outlines (`Regex`, `Choice`,
`KleenePlus`, type leaves, …) expose an AST that is structurally isomorphic to our
`skills.ir` (`Lit/Rx/Choice/Seq/Repeat/Ref`). Rather than vendor either runtime,
this is a thin builder that produces *our* IR — so the same composed grammar
compiles to GBNF (llama.cpp), Lark/guided_grammar (vLLM/SGLang), or the packrat
validator (Tier-0 validate-and-retry), exactly like every other skill.

    from local_harness.structured import select, one_or_more, lit, INT

    verdict = lit("VERDICT: ") + select(["guilty", "not guilty"])
    skill = verdict.skill("verdict")           # → a Skill you can hand to replay_tuned

See docs/guidance-outlines-integration.md for the full capability mapping.
"""

from .builder import (
    G, lit, regex, select, choice, seq, one_or_more, zero_or_more, optional,
    exactly, at_least, at_most, between, rule, json_schema,
    INT, FLOAT, BOOL, WORD, REST_OF_LINE,
)

__all__ = [
    "G", "lit", "regex", "select", "choice", "seq",
    "one_or_more", "zero_or_more", "optional",
    "exactly", "at_least", "at_most", "between", "rule", "json_schema",
    "INT", "FLOAT", "BOOL", "WORD", "REST_OF_LINE",
]
