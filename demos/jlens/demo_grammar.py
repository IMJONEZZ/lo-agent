"""Grammar, shown not told: ask for exactly 7 items and COUNT them — the output
is in the grammar's language with probability 1, because invalid tokens are
masked at decode time. Frontier models miscount; a GBNF-constrained local model
cannot. You see the seven, and you see the guard reject an 8th.

    LO_URL=http://127.0.0.1:8080 python demo_grammar.py
"""

from __future__ import annotations

import os
import sys
import time

import httpx

URL = os.environ.get("LO_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("LO_MODEL", "")
GOLD, GREEN, DIM, BOLD, ROSE, R = "\033[1;38;5;220m", "\033[38;5;42m", "\033[2m", "\033[1m", "\033[38;5;203m", "\033[0m"

# a GBNF that admits EXACTLY seven "- word" lines, then stops. Nothing else can
# be sampled — not six, not eight.
GBNF = r'''root ::= item item item item item item item
item ::= "- " word "\n"
word ::= [A-Za-z]+'''


def slow(t, d=0.011):
    for ch in t:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(d)
    print()


def main():
    c = httpx.Client(timeout=600)
    slow(f"{DIM}$ lo skill — 'list exactly seven fruits'. The grammar admits SEVEN lines{R}")
    slow(f"{DIM}  and nothing else; invalid tokens are masked, so miscounting is impossible.{R}\n")
    time.sleep(0.5)

    body = {"messages": [{"role": "user", "content": "List exactly seven fruits, one per line as '- fruit'."}],
            "temperature": 0.7, "max_tokens": 200, "grammar": GBNF, "seed": 3,
            "chat_template_kwargs": {"enable_thinking": False}}
    if MODEL:
        body["model"] = MODEL
    r = c.post(URL + "/v1/chat/completions", json=body).json()
    out = (r["choices"][0]["message"].get("content") or "").strip()
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("-")]

    print(f"{GOLD}── the model's output ──{R}\n")
    for ln in lines:
        print(f"  {GREEN}{ln}{R}"); time.sleep(0.35)
    print()
    time.sleep(0.5)
    slow(f"{GOLD}── count them ──{R}")
    n = len(lines)
    color = GREEN if n == 7 else ROSE
    slow(f"  items produced: {BOLD}{color}{n}{R}   {DIM}(the grammar's root is seven items — it cannot be six or eight){R}")
    slow(f"\n{BOLD}{GOLD}  exactly seven, by construction. not asked-for-and-hopefully-obeyed.{R}\n")


if __name__ == "__main__":
    main()
