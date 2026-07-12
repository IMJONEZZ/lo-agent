"""KV-cache tree search, shown not told: fork the same prompt into N candidate
answers (each a prefix-cache hit — the shared prompt is decoded once), then let
a verifier score them and pick the best. You see the candidates and the winner;
the frontier can't fork a cache, so it would bill you N full prompts.

    LO_URL=http://127.0.0.1:8080 LO_MODEL=... python demo_bestof.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

import httpx

URL = os.environ.get("LO_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("LO_MODEL", "")
GOLD, GREEN, DIM, BOLD, ROSE, R = "\033[1;38;5;220m", "\033[38;5;42m", "\033[2m", "\033[1m", "\033[38;5;203m", "\033[0m"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
TASK = "Write a single vivid, concrete opening line for a story about a lighthouse keeper. One sentence."


def slow(t, d=0.011):
    for ch in t:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(d)
    print()


async def gen(client, prompt, seed, max_tokens=60, **extra):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 1.0,
            "max_tokens": max_tokens, "seed": seed, "cache_prompt": True, **NOTHINK, **extra}
    if MODEL:
        body["model"] = MODEL
    r = await client.post(URL + "/v1/chat/completions", json=body)
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


async def judge(client, cands):
    """A comparative verifier: pick the single best of N (grammar-forced index)."""
    listing = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cands))
    n = len(cands)
    j = await gen(client, f"Which ONE of these {n} story openings is the most vivid and "
                          f"concrete? Reply with ONLY its number (1-{n}).\n\n{listing}", 0,
                  max_tokens=4, grammar=f'root ::= [1-{n}]')
    digits = "".join(c for c in j if c.isdigit())
    return (int(digits) - 1) if digits and 1 <= int(digits) <= n else 0


async def main():
    client = httpx.AsyncClient(timeout=120)
    slow(f"{DIM}$ lo — fork the prompt into 4 candidates (prefix-cache hits, decoded once),{R}")
    slow(f"{DIM}  then a verifier picks the best. You pick nothing; you watch it pick.{R}\n")
    time.sleep(0.5)
    slow(f"  {DIM}task: {TASK}{R}\n")

    cands = await asyncio.gather(*[gen(client, TASK, s) for s in (1, 2, 3, 4)])
    winner = await judge(client, cands)

    print(f"{GOLD}── 4 candidates (forked from one cached prefix) ──{R}\n")
    for i, cand in enumerate(cands):
        mark = f"{BOLD}{GREEN}★ best{R}" if i == winner else f"{DIM}  #{i+1} {R}"
        col = GREEN if i == winner else DIM
        print(f"  {mark} {col}{cand[:74]}{R}")
        time.sleep(0.4)
    print()
    time.sleep(0.4)
    slow(f"{GOLD}── the verifier selected ──{R}")
    slow(f"  {BOLD}{GREEN}{cands[winner]}{R}")
    slow(f"\n{BOLD}{GOLD}  best-of-N for the price of one prompt + N short completions. the shared{R}")
    slow(f"{BOLD}{GOLD}  prefix was decoded once — a frontier API bills all N in full.{R}\n")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
