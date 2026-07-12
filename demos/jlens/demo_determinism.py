"""Determinism, shown not told: run the SAME prompt+seed twice and watch the
reasoning trace AND the final sentence come out character-for-character identical.

No "byte-exact ✓" label — you read both runs and see they match. A frontier API
(best-effort seed, no logged distribution) drifts; a local seed is reproducible.

    LO_URL=http://127.0.0.1:8080 python demo_determinism.py
"""

from __future__ import annotations

import os
import sys
import time

import httpx

URL = os.environ.get("LO_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("LO_MODEL", "")
PROMPT = "In one short sentence, name the planet known as the Red Planet and why."

GOLD, GREEN, DIM, BOLD, ROSE, R = "\033[1;38;5;220m", "\033[38;5;42m", "\033[2m", "\033[1m", "\033[38;5;203m", "\033[0m"


def slow(t, d=0.01):
    for ch in t:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(d)
    print()


def run(seed):
    c = httpx.Client(timeout=600)
    body = {"messages": [{"role": "user", "content": PROMPT}], "seed": seed,
            "temperature": 0.7, "max_tokens": 1200}
    if MODEL:
        body["model"] = MODEL
    r = c.post(URL + "/v1/chat/completions", json=body).json()
    msg = r["choices"][0]["message"]
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    content = (msg.get("content") or "").strip()
    return reasoning, content


def show(label, reasoning, content, color):
    print(f"{BOLD}{color}{label}{R}")
    if reasoning:
        head = reasoning.replace("\n", " ")[:150]
        print(f"  {DIM}reasoning:{R} {head}{'…' if len(reasoning) > 150 else ''}")
    print(f"  {DIM}answer:{R}    {BOLD}{content}{R}\n")


def main():
    slow(f"{DIM}$ lo run — same prompt, same seed, twice. A local seed is reproducible;{R}")
    slow(f"{DIM}  a frontier best-effort seed is not. Read both and see for yourself.{R}\n")
    time.sleep(0.5)

    print(f"{GOLD}── run 1 (seed 7) ──{R}")
    r1, c1 = run(7)
    show("run 1", r1, c1, GREEN)
    time.sleep(0.8)

    print(f"{GOLD}── run 2 (seed 7, fresh request) ──{R}")
    r2, c2 = run(7)
    show("run 2", r2, c2, GREEN)
    time.sleep(0.8)

    same_r = r1 == r2
    same_c = c1 == c2
    print(f"{GOLD}── compare, character by character ──{R}")
    time.sleep(0.4)
    slow(f"  reasoning trace:  {(GREEN + 'identical') if same_r else (ROSE + 'DIVERGED')}{R}"
         f"  {DIM}({len(r1)} chars){R}")
    slow(f"  final sentence:   {(GREEN + 'identical') if same_c else (ROSE + 'DIVERGED')}{R}")
    if same_r and same_c:
        slow(f"\n{BOLD}{GOLD}  every token, both runs — the same. that's what 'replayable' means.{R}\n")
    else:
        slow(f"\n{DIM}  (this endpoint's seed isn't bit-stable — lo detects that and says so){R}\n")


if __name__ == "__main__":
    main()
