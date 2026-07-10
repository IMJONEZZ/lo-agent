"""Minimal, dependency-free frontmatter parser for markdown command/agent files.

Handles the flat YAML frontmatter that OpenCode / Claude-Code / lo command and
agent files use — no nesting needed, so we avoid a pyyaml dependency (which would
also complicate packaging + determinism). Supports:

    ---
    description: A one-line description
    model: qwen3
    tags: [a, b, c]          # inline list
    steps:                   # block list
      - first
      - second
    enabled: true            # bool / int / float / null coerced
    prompt: |                # block scalar (literal)
      line one
      line two
    ---
    <body text after the closing fence>

Anything the parser doesn't understand is kept as a raw string, never an error —
a malformed frontmatter should degrade to "no metadata", not crash a load.
"""

from __future__ import annotations

_FENCE = "---"


def _coerce(value: str):
    """Coerce a scalar string to bool/int/float/None, else strip quotes."""
    v = value.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_inline_list(value: str) -> list:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    return [_coerce(item) for item in inner.split(",")]


def _parse_inline_map(value: str) -> dict:
    """`{a: true, b: false}` → {'a': True, 'b': False} — OpenCode's tools form."""
    inner = value.strip()[1:-1].strip()
    if not inner:
        return {}
    out: dict = {}
    for part in inner.split(","):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        k = k.strip()
        if k:
            out[k] = _coerce(v)
    return out


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata, body). If there is no leading `---` fence, metadata is
    empty and the whole text is the body."""
    if text is None:
        return {}, ""
    # A frontmatter block must be the very first thing (allowing a leading BOM /
    # blank lines) and open with a `---` line.
    stripped = text.lstrip("﻿")
    lines = stripped.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FENCE:
        return {}, text

    # Find the closing fence.
    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FENCE:
            close = i
            break
    if close is None:
        return {}, text  # unterminated → treat as plain body

    meta_lines = [ln.rstrip("\n") for ln in lines[1:close]]
    body = "".join(lines[close + 1 :])
    return _parse_flat_yaml(meta_lines), body.lstrip("\n")


def _parse_flat_yaml(lines: list[str]) -> dict:
    meta: dict = {}
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if not key:
            continue

        # Block scalar: `key: |` or `key: >` — collect indented following lines.
        if rest in ("|", ">", "|-", ">-"):
            block: list[str] = []
            while i < n:
                nxt = lines[i]
                if nxt.strip() and not nxt.startswith((" ", "\t")):
                    break
                block.append(nxt)
                i += 1
            # de-indent by the common leading whitespace
            dedented = _dedent(block)
            sep = "\n" if rest.startswith("|") else " "
            val = sep.join(s.rstrip() for s in dedented).strip()
            meta[key] = val
            continue

        # Inline list: `key: [a, b]`
        if rest.startswith("[") and rest.endswith("]"):
            meta[key] = _parse_inline_list(rest)
            continue

        # Inline map: `key: {a: true, b: false}` (OpenCode's tools form)
        if rest.startswith("{") and rest.endswith("}"):
            meta[key] = _parse_inline_map(rest)
            continue

        # Block list: `key:` followed by `- item` lines.
        if rest == "":
            items: list = []
            while i < n:
                nxt = lines[i].strip()
                if not nxt.startswith("- "):
                    if not nxt or lines[i].startswith((" ", "\t")):
                        # blank or deeper indent that isn't a list item → stop
                        if not nxt:
                            i += 1
                            continue
                    break
                items.append(_coerce(nxt[2:]))
                i += 1
            meta[key] = items if items else None
            continue

        meta[key] = _coerce(rest)
    return meta


def _dedent(lines: list[str]) -> list[str]:
    indents = [
        len(ln) - len(ln.lstrip(" \t")) for ln in lines if ln.strip()
    ]
    if not indents:
        return [ln.strip() for ln in lines]
    cut = min(indents)
    return [ln[cut:] if len(ln) >= cut else ln for ln in lines]
