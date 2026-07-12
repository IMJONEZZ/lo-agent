"""Uncertainty, shown not told: sample the same question five times. On a
question with one right answer, the samples AGREE — you can trust it. On an
open question, they SCATTER — the model is guessing, and lo can route that to
'ask' or 'abstain'. Consensus IS the confidence signal; you read the spread.

Local tokens are free, so five samples cost nothing extra.

    LO_URL=http://127.0.0.1:8080 LO_MODEL=... python demo_consistency.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import Counter

import httpx

URL = os.environ.get("LO_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("LO_MODEL", "")
GOLD, GREEN, DIM, BOLD, ROSE, R = "\033[1;38;5;220m", "\033[38;5;42m", "\033[2m", "\033[1m", "\033[38;5;203m", "\033[0m"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def slow(t, d=0.011):
    for ch in t:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(d)
    print()


async def sample(client, prompt, seed):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 1.0,
            "max_tokens": 24, "seed": seed, **NOTHINK}
    if MODEL:
        body["model"] = MODEL
    r = await client.post(URL + "/v1/chat/completions", json=body)
    return (r.json()["choices"][0]["message"].get("content") or "").strip().rstrip(".").strip()


async def run(client, label, prompt, extract):
    print(f"{GOLD}── {label} ──{R}")
    slow(f"  {DIM}{prompt}{R}")
    answers = await asyncio.gather(*[sample(client, prompt, s) for s in range(5)])
    keyed = [extract(a) for a in answers]
    for i, (a, k) in enumerate(zip(answers, keyed), 1):
        print(f"  {DIM}sample {i}:{R} {a[:48]}")
        time.sleep(0.25)
    top, n = Counter(keyed).most_common(1)[0]
    agree = n / len(keyed)
    color = GREEN if agree >= 0.8 else (GOLD if agree >= 0.5 else ROSE)
    verdict = "trust it" if agree >= 0.8 else ("shaky" if agree >= 0.5 else "the model is guessing")
    slow(f"  → agreement: {BOLD}{color}{int(agree*100)}%{R} on {top!r}  {DIM}— {verdict}{R}\n")
    return agree


async def main():
    client = httpx.AsyncClient(timeout=120)
    slow(f"{DIM}$ lo — five samples of each question. free tokens, so it costs nothing.{R}")
    slow(f"{DIM}  watch the SPREAD, not a confidence number:{R}\n")
    time.sleep(0.5)

    await run(client, "a question with one answer", "What is the capital of France? One word.",
              lambda a: a.lower().split()[0] if a.split() else "")
    time.sleep(0.4)
    await run(client, "an open question with no single answer",
              "Pick one number between 1 and 100. Just the number.",
              lambda a: "".join(ch for ch in a if ch.isdigit())[:3])

    slow(f"{BOLD}{GOLD}  same machinery, opposite spread — consensus is the confidence, and it's{R}")
    slow(f"{BOLD}{GOLD}  something a single sample (or a frontier API) can't give you.{R}\n")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
