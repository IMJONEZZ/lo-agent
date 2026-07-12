"""J-Lens demo (show-don't-tell): SEE the J-space, then STEER it.

Drives the live lens service and prints, in sequence:
  1. the position×layer heatmap of a real turn — the model's own reasoning,
     laid out layer by layer (you watch ' Euro' win at the output);
  2. a rank trajectory — ' euro' and ' lira' contend in the workspace band
     before the model commits (the runner-up hypotheses, visible);
  3. an A/B: ablate the ' Euro' concept and watch the SAME prompt produce a
     different currency — no label needed, you see the sentence change.

Run against a lens service:  LO_LENS_URL=http://127.0.0.1:8092 python demo_jlens.py
"""

from __future__ import annotations

import base64
import os
import sys
import time

import httpx
import numpy as np

B = os.environ.get("LO_LENS_URL", "http://127.0.0.1:8092")
PROMPT = ("Fact: The capital of Japan is Tokyo.\n"
          "Fact: The currency used in the country shaped like a boot is")

GREEN, GOLD, DIM, RED, BOLD, R = "\033[38;5;42m", "\033[38;5;220m", "\033[2m", "\033[38;5;203m", "\033[1m", "\033[0m"


def slow(text: str, d: float = 0.012) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(d)
    print()


def hdr(t: str) -> None:
    print(f"\n{BOLD}{GOLD}── {t} ──{R}\n")
    time.sleep(0.6)


def _decode(b64, shape):
    return np.frombuffer(base64.b64decode(b64), "<i4").reshape(shape)


def main() -> None:
    c = httpx.Client(timeout=600)
    vocab = c.get(B + "/lens/vocab").json()["pieces"]

    print(f"{DIM}$ lo lens — read and steer the residual stream (access-ladder Rung 6){R}")
    slow(f"{DIM}  prompt:{R} …the currency used in the country shaped like a boot is ▮")

    hdr("1. SEE the J-space  —  what the model is thinking, layer by layer")
    sl = c.post(B + "/lens/slice", json={"prompt": PROMPT, "stride": 8, "top_n": 3}).json()
    T, layers = len(sl["tokens"]), sl["layers"]
    top = _decode(sl["top_ids"], (T, len(layers), 3))
    # show the last few positions across the stack
    print(f"  {DIM}position          " + "".join(f"L{l:<8}" for l in layers) + f"→ output{R}")
    for pos in range(max(0, T - 6), T):
        piece = vocab[top[pos, -1, 0]].replace("\n", "\\n")[:12]
        row = f"  {vocab[sl['tokens'][pos]][:14].replace(chr(10),'/'):14}  "
        for li in range(len(layers)):
            t = vocab[int(top[pos, li, 0])][:7].replace("\n", "\\n")
            same = int(top[pos, li, 0]) == int(top[pos, -1, 0])
            row += f"{(GREEN if same else DIM)}{t:<9}{R}"
        print(row + f"  {BOLD}{GREEN}{piece}{R}")
        time.sleep(0.25)
    slow(f"\n  {DIM}the last row is the answer position — watch it resolve to{R} {BOLD}{GREEN} the{R}"
         f"{DIM}, then the currency.{R}")

    hdr("2. The runner-up hypotheses  —  contenders in the workspace band")
    ids, pieces = {}, {}
    for q in (" Euro", " euro", " lira", " yen"):
        res = c.get(B + "/lens/search_tokens", params={"q": q.strip(), "limit": 20}).json()["results"]
        ex = [r for r in res if r["piece"] == q]
        if ex:
            ids[q] = ex[0]["token"]
    rr = c.post(B + "/lens/ranks", json={"ctx_id": sl["ctx_id"], "token_ids": list(ids.values())}).json()
    R3 = _decode(rr["ranks"], rr["shape"])
    last = T - 1
    print(f"  {DIM}rank at the answer position, across the stack (lower = stronger):{R}\n")
    for j, q in enumerate(ids):
        ranks = [int(R3[last, li, j]) for li in range(len(layers))]
        spark = "".join("▁▂▃▄▅▆▇█"[max(0, min(7, int((5 - np.log10(max(1, r + 1))) / 5 * 7)))] for r in ranks)
        col = GREEN if ranks[-1] < 5 else GOLD if min(ranks) < 300 else DIM
        print(f"  {col}{q:7}{R}  {col}{spark}{R}  {DIM}final rank {ranks[-1]}{R}")
        time.sleep(0.3)
    slow(f"\n  {DIM}' euro' and ' lira' climb in the middle layers — the model considers Italy's"
         f" options{R}\n  {DIM}before committing. That's the J-space workspace, made visible.{R}")

    hdr("3. STEER it  —  summon a concept, same prompt, watch the answer change")
    slow(f"{DIM}$ lo lens gen \"…boot is\" --steer ' yen' --alpha 3 --layers 40,60{R}")
    time.sleep(0.4)
    out = c.post(B + "/lens/generate", json={
        "prompt": PROMPT, "n_predict": 12,
        "interventions": [{"type": "steer", "token_id": ids[" yen"], "alpha": 3.0,
                           "layers": [40, 60]}],
        "compare": True}).json()
    time.sleep(0.3)
    print(f"\n  {DIM}baseline {R}{DIM}(untouched):{R}  {BOLD}{out['baseline']['text'].strip()}{R}")
    time.sleep(0.9)
    print(f"  {GREEN}steered{R}  {DIM}(+' yen'):{R}    {BOLD}{GREEN}{out['steered']['text'].strip()}{R}")
    slow(f"\n  {DIM}we added the ' yen' direction to the residual stream — the model reaches"
         f" for it{R}\n  {DIM}and keeps writing fluently, live, on a quantized GGUF.{R}")

    hdr("… or take the concept AWAY  —  ablate ' Euro' from the stream")
    slow(f"{DIM}$ lo lens gen \"…boot is\" --ablate ' Euro' --layers 40,60{R}")
    ab = c.post(B + "/lens/generate", json={
        "prompt": PROMPT, "n_predict": 10,
        "interventions": [{"type": "ablate", "token_id": ids[" Euro"], "layers": [40, 60]},
                          {"type": "ablate", "token_id": ids.get(" euro", ids[" Euro"]),
                           "layers": [40, 60]}],
        "compare": False}).json()
    time.sleep(0.3)
    print(f"\n  {RED}ablated{R}  {DIM}(no ' Euro'):{R} {BOLD}{ab['steered']['text'].strip()}{R}")
    slow(f"\n{BOLD}{GOLD}  frontier APIs will never expose this. lo reads and steers it over HTTP.{R}\n")


if __name__ == "__main__":
    main()
