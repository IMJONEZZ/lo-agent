"""Grammar IR: one AST, three targets.

Productions are written in a small EBNF dialect:

    select_stmt = '"SELECT " select_list " FROM " ident clause? ";"'
    select_list = '"*" | ident ("," " "? ident)*'
    ident       = '/[a-zA-Z_][a-zA-Z0-9_]*/'

Atoms: "literal", rule_name, /regex/ (single re-matchable unit), ( group ).
Suffixes: * + ?. Alternation: |.

The same AST compiles to GBNF (llama.cpp `grammar` param), Lark (vLLM/SGLang
`guided_grammar`), and drives a packrat validator used both for the Tier-0
validate-and-retry fallback and to verify server-constrained output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class GrammarError(Exception):
    pass


# --- AST ---------------------------------------------------------------------


@dataclass(frozen=True)
class Lit:
    text: str


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class Rx:
    pattern: str  # an re pattern matched greedily at the current position


@dataclass(frozen=True)
class Seq:
    items: tuple


@dataclass(frozen=True)
class Choice:
    options: tuple


@dataclass(frozen=True)
class Repeat:
    item: Any
    min: int  # 0 for * and ?, 1 for +
    max: int | None  # None = unbounded, 1 for ?


# --- production parser -------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<string>"(?:[^"\\]|\\.)*")
      | (?P<regex>/(?:[^/\\]|\\.)+/)
      | (?P<name>[a-zA-Z_][a-zA-Z0-9_]*)
      | (?P<op>[()|*+?])
    )""",
    re.VERBOSE,
)


def _tokenize(src: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if m is None:
            if src[pos:].strip() == "":
                break
            raise GrammarError(f"bad production syntax at: {src[pos:pos+20]!r}")
        pos = m.end()
        for kind in ("string", "regex", "name", "op"):
            if m.group(kind) is not None:
                tokens.append((kind, m.group(kind)))
                break
    return tokens


def parse_production(src: str):
    """Parse one production body into an AST node."""
    tokens = _tokenize(src)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def parse_choice():
        nonlocal pos
        options = [parse_seq()]
        while peek() == ("op", "|"):
            pos += 1
            options.append(parse_seq())
        return options[0] if len(options) == 1 else Choice(tuple(options))

    def parse_seq():
        nonlocal pos
        items = []
        while True:
            kind, val = peek()
            if kind is None or (kind == "op" and val in ")|"):
                break
            items.append(parse_suffixed())
        if not items:
            raise GrammarError(f"empty sequence in production: {src!r}")
        return items[0] if len(items) == 1 else Seq(tuple(items))

    def parse_suffixed():
        nonlocal pos
        atom = parse_atom()
        kind, val = peek()
        if kind == "op" and val in "*+?":
            pos += 1
            return Repeat(atom, min=1 if val == "+" else 0, max=1 if val == "?" else None)
        return atom

    def parse_atom():
        nonlocal pos
        kind, val = peek()
        pos += 1
        if kind == "string":
            return Lit(val[1:-1].encode().decode("unicode_escape"))
        if kind == "regex":
            pattern = val[1:-1].replace("\\/", "/")
            try:
                re.compile(pattern)
            except re.error as e:
                raise GrammarError(f"bad regex {pattern!r}: {e}") from e
            return Rx(pattern)
        if kind == "name":
            return Ref(val)
        if kind == "op" and val == "(":
            inner = parse_choice()
            k, v = peek()
            if (k, v) != ("op", ")"):
                raise GrammarError(f"unclosed group in production: {src!r}")
            pos += 1
            return inner
        raise GrammarError(f"unexpected token {val!r} in production: {src!r}")

    node = parse_choice()
    if pos != len(tokens):
        raise GrammarError(f"trailing tokens in production: {src!r}")
    return node


# --- grammar -----------------------------------------------------------------


@dataclass
class Grammar:
    rules: dict[str, Any]  # name -> AST node
    root: str

    @classmethod
    def from_rules(cls, rules: dict[str, str], root: str, check: bool = True) -> "Grammar":
        g = cls(rules={name: parse_production(body) for name, body in rules.items()}, root=root)
        if check:  # skills with imports defer the check until composition
            g.check()
        return g

    def check(self) -> None:
        if self.root not in self.rules:
            raise GrammarError(f"root rule {self.root!r} not defined")
        for name, node in self.rules.items():
            for ref in _refs(node):
                if ref not in self.rules:
                    raise GrammarError(f"rule {name!r} references undefined rule {ref!r}")

    def merge(self, other: "Grammar") -> "Grammar":
        """Compose: import another grammar's rules (conflicts must be identical)."""
        merged = dict(other.rules)
        for name, node in self.rules.items():
            if name in merged and merged[name] != node:
                raise GrammarError(f"conflicting definitions for rule {name!r}")
            merged[name] = node
        return Grammar(rules=merged, root=self.root)

    # --- emitters --------------------------------------------------------

    def to_gbnf(self) -> str:
        lines = [f"root ::= {_gbnf(self.rules[self.root])}"]
        for name, node in self.rules.items():
            if name != self.root:
                lines.append(f"{_safe(name)} ::= {_gbnf(node)}")
        return "\n".join(lines)

    def to_lark(self) -> str:
        # Lark rule names keep underscores (dashes are GBNF-only).
        lines = [f"start: {_lark(self.rules[self.root])}"]
        for name, node in self.rules.items():
            if name != self.root:
                lines.append(f"{name}: {_lark(node)}")
        return "\n".join(lines)

    # --- validation (packrat recursive descent) ---------------------------

    def validate(self, text: str) -> bool:
        return any(end == len(text) for end in self._match(self.rules[self.root], text, 0, {}))

    def _match(self, node, text: str, pos: int, memo: dict) -> list[int]:
        """All end positions reachable by matching `node` at `pos`."""
        key = (id(node), pos)
        if key in memo:
            return memo[key]
        memo[key] = []  # guards left recursion: re-entry yields no new positions
        if isinstance(node, Lit):
            out = [pos + len(node.text)] if text.startswith(node.text, pos) else []
        elif isinstance(node, Rx):
            m = re.compile(node.pattern).match(text, pos)
            out = [m.end()] if m else []
        elif isinstance(node, Ref):
            out = self._match(self.rules[node.name], text, pos, memo)
        elif isinstance(node, Choice):
            out = sorted({e for opt in node.options for e in self._match(opt, text, pos, memo)})
        elif isinstance(node, Seq):
            ends = [pos]
            for item in node.items:
                ends = sorted({e2 for e in ends for e2 in self._match(item, text, e, memo)})
                if not ends:
                    break
            out = ends
        elif isinstance(node, Repeat):
            out, frontier, count = [], [pos], 0
            if node.min == 0:
                out.append(pos)
            while frontier and (node.max is None or count < node.max):
                frontier = sorted(
                    {e2 for e in frontier for e2 in self._match(node.item, text, e, memo) if e2 > e}
                )
                count += 1
                if count >= node.min:
                    out.extend(frontier)
            out = sorted(set(out))
        else:  # pragma: no cover
            raise GrammarError(f"unknown node {node!r}")
        memo[key] = out
        return out


def _refs(node):
    if isinstance(node, Ref):
        yield node.name
    elif isinstance(node, Seq):
        for i in node.items:
            yield from _refs(i)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _refs(o)
    elif isinstance(node, Repeat):
        yield from _refs(node.item)


def _safe(name: str) -> str:
    return name.replace("_", "-")  # GBNF rule names disallow underscores


def _gbnf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")


def _gbnf(node, parenthesize: bool = False) -> str:
    if isinstance(node, Lit):
        return f'"{_gbnf_escape(node.text)}"'
    if isinstance(node, Ref):
        return _safe(node.name)
    if isinstance(node, Rx):
        return _regex_to_gbnf(node.pattern)
    if isinstance(node, Seq):
        s = " ".join(_gbnf(i, True) for i in node.items)
        return f"({s})" if parenthesize else s
    if isinstance(node, Choice):
        s = " | ".join(_gbnf(o) for o in node.options)
        return f"({s})"
    if isinstance(node, Repeat):
        suffix = {(0, None): "*", (1, None): "+", (0, 1): "?"}[(node.min, node.max)]
        return f"{_gbnf(node.item, True)}{suffix}"
    raise GrammarError(f"cannot emit {node!r}")


_CLASS_SEQ = re.compile(r"^(\[[^\[\]]*\][*+?]?)+$")


def _regex_to_gbnf(pattern: str) -> str:
    """GBNF supports sequences of char classes with * + ? suffixes
    (e.g. [a-zA-Z_][a-zA-Z0-9_]*); reject anything fancier."""
    if not _CLASS_SEQ.match(pattern):
        raise GrammarError(
            f"regex {pattern!r} too complex for GBNF: use char-class sequences like "
            "[a-z][a-z0-9]* or express structure with grammar rules"
        )
    return pattern


def _lark(node, parenthesize: bool = False) -> str:
    if isinstance(node, Lit):
        return '"' + node.text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(node, Ref):
        return node.name
    if isinstance(node, Rx):
        return f"/{node.pattern}/"
    if isinstance(node, Seq):
        s = " ".join(_lark(i, True) for i in node.items)
        return f"({s})" if parenthesize else s
    if isinstance(node, Choice):
        return "(" + " | ".join(_lark(o) for o in node.options) + ")"
    if isinstance(node, Repeat):
        suffix = {(0, None): "*", (1, None): "+", (0, 1): "?"}[(node.min, node.max)]
        return f"{_lark(node.item, True)}{suffix}"
    raise GrammarError(f"cannot emit {node!r}")
