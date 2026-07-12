"""Anti-slop, shown not told: 'tapestry' is the model's favorite AI-slop word.
Unbanned, it reaches for it. With lo's anti-slop on, when 'tapestry' starts to
form lo rewinds the KV cache to before it and re-samples with its first token
masked — the phrase CANNOT occur. Same prompt, same seed; read both.

    LO_URL=http://127.0.0.1:8080 LO_MODEL=... python demo_antislop.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time

from local_harness.inference.client import OpenAICompatClient
from local_harness.inference.types import Message
from local_harness.logits.antislop import generate_antislop

URL = os.environ.get("LO_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("LO_MODEL", "")
BANNED = ["tapestry"]
PROMPT = "Write one flowery, inspirational sentence about how a diverse community weaves many cultures together."
NOTHINK = "<think>\n\n</think>\n\n"
GOLD, GREEN, DIM, BOLD, ROSE, R = "\033[1;38;5;220m", "\033[38;5;42m", "\033[2m", "\033[1m", "\033[38;5;203m", "\033[0m"


def slow(t, d=0.011):
    for ch in t:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(d)
    print()


async def main():
    client = OpenAICompatClient(URL, MODEL)
    msgs = [Message(role="user", content=PROMPT)]

    slow(f"{DIM}$ lo — 'tapestry' is the classic LLM-slop word. First, no ban:{R}\n")
    time.sleep(0.4)
    base = await generate_antislop(client, msgs, [], max_tokens=70, seed=11, prefill=NOTHINK)
    txt = base.text.strip()
    hl = re.sub("(tapestry)", ROSE + BOLD + r"\1" + R + DIM, txt, flags=re.I)
    print(f"{GOLD}── no ban ──{R}")
    slow(f"  {DIM}{hl}{R}", d=0.005)
    print()
    time.sleep(0.7)
    slow(f"{DIM}  there it is. now ban it — same prompt, same seed, KV-rewind on any hit:{R}\n")
    time.sleep(0.3)

    res = await generate_antislop(client, msgs, BANNED, max_tokens=70, seed=11, prefill=NOTHINK)
    out = res.text.strip()
    print(f"{GOLD}── 'tapestry' banned ({res.rewinds} KV-rewind{'s' if res.rewinds != 1 else ''}) ──{R}")
    slow(f"  {GREEN}{out}{R}", d=0.005)
    print()
    time.sleep(0.5)
    present = bool(re.search("tapestry", out, re.I))
    print(f"{GOLD}── scan ──{R}")
    slow(f"  'tapestry':  {(ROSE + 'found') if present else (GREEN + 'absent')}{R}"
         f"   {DIM}rewinds: {res.rewinds}{R}")
    if not present:
        slow(f"\n{BOLD}{GOLD}  the model wrote around it and stayed fluent — impossible by construction.{R}\n")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
