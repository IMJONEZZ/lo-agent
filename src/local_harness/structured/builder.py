"""The builder layer — `G` wraps an IR node with guidance/Outlines ergonomics.

Composition: `a + b` → Seq, `a | b` → Choice. Bounded repeats (`exactly`,
`at_least`, `at_most`, `between`) are EXPANDED at build time into `* + ?` forms,
because GBNF has no `{m,n}` quantifier — keeping every node emittable to GBNF,
Lark, and the validator without special-casing the emitters.
"""

from __future__ import annotations

from typing import Any

from ..skills.ir import Choice, Grammar, Lit, Ref, Repeat, Rx, Seq
from ..skills.skill import Skill


def _node(x: "G | str") -> Any:
    """Coerce a builder term or a bare string (→ literal) to an IR node."""
    if isinstance(x, G):
        return x.node
    if isinstance(x, str):
        return Lit(x)
    raise TypeError(f"expected a G term or str, got {type(x).__name__}")


class G:
    """A grammar term wrapping one IR node, composable with + and |."""

    __slots__ = ("node",)

    def __init__(self, node: Any):
        self.node = node

    def __add__(self, other: "G | str") -> "G":
        return G(Seq((self.node, _node(other))))

    def __radd__(self, other: "G | str") -> "G":
        return G(Seq((_node(other), self.node)))

    def __or__(self, other: "G | str") -> "G":
        return G(Choice((self.node, _node(other))))

    def __ror__(self, other: "G | str") -> "G":
        return G(Choice((_node(other), self.node)))

    # --- terminal forms ---------------------------------------------------

    def grammar(self, root: str = "root") -> Grammar:
        return Grammar(rules={root: self.node}, root=root)

    def to_gbnf(self) -> str:
        return self.grammar().to_gbnf()

    def to_lark(self) -> str:
        return self.grammar().to_lark()

    def validate(self, text: str) -> bool:
        return self.grammar().validate(text)

    def skill(self, name: str, description: str = "", **kw) -> Skill:
        return Skill(name=name, description=description, grammar=self.grammar(), **kw)


# --- leaves & combinators ---------------------------------------------------


def lit(text: str) -> G:
    return G(Lit(text))


def regex(pattern: str) -> G:
    return G(Rx(pattern))


def seq(*terms: "G | str") -> G:
    return G(Seq(tuple(_node(t) for t in terms)))


def choice(*terms: "G | str") -> G:
    return G(Choice(tuple(_node(t) for t in terms)))


def select(options: list["G | str"]) -> G:
    """guidance.select / Outlines Choice — one of the given options."""
    return G(Choice(tuple(_node(o) for o in options)))


def one_or_more(t: "G | str") -> G:
    return G(Repeat(_node(t), min=1, max=None))


def zero_or_more(t: "G | str") -> G:
    return G(Repeat(_node(t), min=0, max=None))


def optional(t: "G | str") -> G:
    return G(Repeat(_node(t), min=0, max=1))


def exactly(n: int, t: "G | str") -> G:
    """Exactly n repeats. Expanded to a Seq so it stays GBNF-emittable."""
    node = _node(t)
    if n <= 0:
        raise ValueError("exactly(n) needs n >= 1")
    return G(node if n == 1 else Seq(tuple(node for _ in range(n))))


def at_least(n: int, t: "G | str") -> G:
    """n or more: n required copies then a `*` tail."""
    node = _node(t)
    if n <= 0:
        return zero_or_more(t)
    tail = Repeat(node, min=0, max=None)
    return G(Seq(tuple(node for _ in range(n)) + (tail,)))


def at_most(n: int, t: "G | str") -> G:
    """0..n: n optional copies (`t? t? … t?`)."""
    node = _node(t)
    if n <= 0:
        raise ValueError("at_most(n) needs n >= 1")
    opt = Repeat(node, min=0, max=1)
    return G(Seq(tuple(opt for _ in range(n))))


def between(m: int, n: int, t: "G | str") -> G:
    """m..n: m required then (n-m) optional."""
    if m < 0 or n < m:
        raise ValueError("between(m, n) needs 0 <= m <= n")
    node = _node(t)
    opt = Repeat(node, min=0, max=1)
    return G(Seq(tuple(node for _ in range(m)) + tuple(opt for _ in range(n - m))))


def rule(name: str) -> G:
    """A reference to a named rule (compose full grammars via Grammar.merge)."""
    return G(Ref(name))


def json_schema(schema_or_model: Any, name: str = "json", description: str = "") -> Skill:
    """A JSON-constrained skill from a JSON-schema dict or a Pydantic model.
    JSON rides the server's json_schema / guided_json path (and the validator)."""
    schema = (schema_or_model.model_json_schema()
              if hasattr(schema_or_model, "model_json_schema") else schema_or_model)
    return Skill(name=name, description=description, json_schema=schema)


# --- type leaves (ported from Outlines' python_types_to_terms) --------------
# Built structurally (not as one regex) so they emit to GBNF too — llama.cpp's
# GBNF only supports char-class *sequences*, not a leading `-?` or a `.` literal.

_DIGIT = regex(r"[0-9]")
INT = optional("-") + _DIGIT + zero_or_more(_DIGIT)
FLOAT = INT + lit(".") + _DIGIT + zero_or_more(_DIGIT)
BOOL = choice("true", "false")
WORD = regex(r"[a-zA-Z][a-zA-Z]*")          # a char-class sequence — GBNF-safe
REST_OF_LINE = regex(r"[^\n][^\n]*")        # char classes — GBNF-safe
