"""Pure render functions for the Jacobian-lens tab (Rung 6).

No Textual imports — Rich renderables only, colored via the active palette
(read by name at call time so /theme recolors them). The LensScreen fetches
data from the lens service and hands the decoded arrays here.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import render as R

# rank → shade: how far a token is from the top of the distribution. Low rank
# (near the model's actual belief) = bright/solid; high rank = dim.
_BANDS = [
    (0, "█", "bold "),      # top-1
    (3, "▓", "bold "),      # top few
    (20, "▒", ""),          # workspace band
    (200, "░", "dim "),     # runner-up hypotheses
]


def _shade(rank: int) -> tuple[str, str]:
    for thresh, ch, style in _BANDS:
        if rank <= thresh:
            return ch, style
    return "·", "dim "


def _piece(vocab, tid: int) -> str:
    p = vocab[tid] if 0 <= tid < len(vocab) else f"[{tid}]"
    return p.replace("\n", "\\n").replace("\t", "\\t")


def lens_grid(
    *,
    pieces: list[str],
    layers: list[int],
    top_ids,          # np [T, L, K] int
    final_ids,        # np [T] int — the model's own top-1 (last layer col)
    changed=None,     # optional set[(pos)] where intervention flipped top-1
    cursor: tuple[int, int] = (0, 0),   # (pos_index, layer_index)
    vocab=None,
    max_rows: int = 40,
    boundary: int | None = None,   # index where prompt ends / generation begins
) -> RenderableType:
    """The position×layer heatmap: each cell = lens top-1 token colored by rank.

    Cells whose top-1 differs from the model's final output are dim; the
    cursor cell is highlighted; ``changed`` positions get an orange ⚠.
    """
    T = len(pieces)
    lo = max(0, min(cursor[0] - max_rows // 2, T - max_rows)) if T > max_rows else 0
    hi = min(T, lo + max_rows)

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right")   # position + token
    for _ in layers:
        table.add_column(justify="left")
    # header row
    hdr = [Text("pos", style=R.C_DIM)]
    for li, l in enumerate(layers):
        style = "bold " + R.C_GOLD if li == cursor[1] else R.C_DIM
        hdr.append(Text(f"L{l}", style=style))
    table.add_row(*hdr)

    changed = changed or set()
    for pos in range(lo, hi):
        tok = _piece(vocab, top_ids[pos, -1, 0]) if vocab is not None else ""
        label = Text()
        if boundary is not None and pos == boundary:
            label.append("┈ ", style=R.C_DIM)
        mark = "⚠" if pos in changed else ("›" if pos == cursor[0] else " ")
        label.append(f"{mark}{pos:>3} ", style=(R.C_TOOL if pos in changed else R.C_DIM))
        label.append((tok[:10]).ljust(10),
                     style=("bold " + R.C_OK if pos == cursor[0] else R.C_DIM))
        cells = [label]
        for li, _ in enumerate(layers):
            tid = int(top_ids[pos, li, 0])
            piece = _piece(vocab, tid)[:8] if vocab is not None else str(tid)
            # rank of this cell's top-1 within the FINAL distribution isn't
            # available here; we shade by column depth as a cheap proxy unless
            # a rank array is supplied. Use solidity by whether it matches final.
            same = vocab is not None and tid == int(final_ids[pos])
            ch, base = _shade(0 if same else 30)
            cur = pos == cursor[0] and li == cursor[1]
            style = ("bold " + R.C_GOLD) if cur else (base + (R.C_OK if same else R.C_DIM))
            cells.append(Text(f"{ch} {piece}", style=style))
        table.add_row(*cells)

    title = Text("J-Lens · ", style="bold " + R.C_GOLD)
    title.append("position × layer — cell = top-1 token; solid = matches output",
                 style=R.C_DIM)
    return Panel(table, title=title, border_style=R.B_JADE if hasattr(R, "B_JADE") else R.C_OK)


def cell_readout_panel(readout: dict, vocab=None) -> RenderableType:
    """Top-k tokens with probabilities at the selected (pos, layer) cell."""
    t = Text()
    t.append(f"cell L{readout['layer']}·p{readout['pos']}  ", style="bold " + R.C_GOLD)
    for tok in readout["tokens"][:8]:
        piece = tok["piece"].replace("\n", "\\n")
        t.append(f"{piece!r}", style=R.C_OK)
        t.append(f" {tok['prob']:.2f}  ", style=R.C_DIM)
    return t


def pins_panel(pins: list[dict]) -> RenderableType:
    """Rank trajectories (rank vs layer) for pinned tokens — a sparkline each."""
    if not pins:
        return Text("pins: (none — press p on a cell to pin its token)", style=R.C_DIM)
    spark = "▁▂▃▄▅▆▇█"
    t = Text("pins:\n", style="bold " + R.C_GOLD)
    for pin in pins:
        ranks = pin["ranks"]  # list[int], per readout layer at the final pos
        # map rank (0=best) to sparkline height (best=tall) on a log scale
        import math
        heights = []
        for r in ranks:
            lg = math.log10(max(1, r + 1))
            idx = max(0, min(len(spark) - 1, int((5 - lg) / 5 * (len(spark) - 1))))
            heights.append(spark[idx])
        t.append(f"  {pin['piece']!r:14}", style=R.C_SAKURA)
        t.append("".join(heights), style=R.C_OK)
        t.append(f"  r{ranks[-1]}\n", style=R.C_DIM)
    return t


def decompose_panel(items: list[dict], explained: float | None = None) -> RenderableType:
    """The sparse J-space decomposition: which token-directions make up this cell."""
    if not items:
        return Text("decompose: (press d on a cell)", style=R.C_DIM)
    t = Text("decompose: ", style="bold " + R.C_GOLD)
    parts = []
    for it in items[:6]:
        parts.append(f"{it['piece']!r} {it['coeff']:.1f}")
    t.append(" + ".join(parts), style=R.C_OK)
    if explained is not None:
        t.append(f"   ({explained*100:.0f}% var)", style=R.C_DIM)
    return t


def interventions_panel(specs: list[dict], mode: str, vocab=None) -> RenderableType:
    """The live intervention set + INSPECT/STEER status."""
    t = Text()
    chip = "STEER" if mode == "steer" else "INSPECT"
    t.append(f" {chip} ", style=f"bold {R.INK} on {R.C_GOLD if mode == 'steer' else R.C_DIM}")
    t.append("  interventions:\n", style=R.C_DIM)
    if not specs:
        t.append("  (none — [s]teer [a]blate [w]swap on a cell)\n", style=R.C_DIM)
        return t
    glyph = {"steer": "↑", "ablate": "✂", "swap": "⇄"}
    for sp in specs:
        g = glyph.get(sp["type"], "•")
        lr = sp.get("layers") or ["all"]
        if sp["type"] == "swap":
            desc = f"{sp.get('piece_a', sp.get('token_a'))!r}⇄{sp.get('piece_b', sp.get('token_b'))!r}"
        else:
            desc = f"{sp.get('piece', sp.get('token_id'))!r}"
            if sp["type"] == "steer":
                desc += f" α={sp.get('alpha', 2.0)}"
        t.append(f"  {g} {sp['type']:6} {desc}  L{lr[0]}–{lr[-1]}\n", style=R.C_OK)
    return t


_HELP_ROWS = [
    ("what am I looking at?", None),
    ("", "Each column is a layer, each row a token of the turn. A cell shows what "
         "the model's residual stream 'means' at that layer — the tokens it would "
         "decode to if you read it out there. Watch a concept sharpen left→right."),
    ("", "A dim divider separates your prompt from what the model generated."),
    ("moving around", None),
    ("↑ ↓ ← →", "move the cursor; the readout below follows the selected cell"),
    ("d", "decompose the selected cell into contributing directions"),
    ("p", "pin the cell's top token — pinned tokens get a rank-per-layer strip"),
    ("L", "toggle lens ⇄ logit-lens (raw unembedding readout)"),
    ("reading a different turn", None),
    ("e", "edit the text being analyzed (type your own)"),
    ("r", "re-run on the latest turn from the chat"),
    ("n", "generate a continuation through the lens, so you watch it token by token"),
    ("steering (needs an intervention-capable endpoint)", None),
    ("s", "steer toward the cell's token (adds it, then re-slices)"),
    ("a", "ablate the cell's token"),
    ("w", "swap the last pinned token with the cell's token"),
    ("g", "A/B generate: baseline vs the current intervention set"),
    ("V", "push these interventions to the chat — steers your NEXT real turn"),
    ("c", "clear interventions (also clears any pushed to the chat)"),
    ("", None),
    ("? / esc", "toggle this help / close the lens"),
]


def help_panel() -> RenderableType:
    """The key legend. The tab binds 14 keys; the hint bar can only show a few."""
    t = Text()
    for key, desc in _HELP_ROWS:
        if desc is None:
            t.append(f"\n{key}\n" if key else "\n", style="bold " + R.C_GOLD)
            continue
        if key:
            t.append(f"  {key:<10}", style="bold " + R.C_ANSWER)
            t.append(f"{desc}\n", style=R.C_DIM)
        else:
            for line in _wrap(desc, 76):
                t.append(f"  {line}\n", style=R.C_DIM)
    return Panel(t, title="lens — keys", border_style=R.C_GOLD, expand=False)


def _wrap(s: str, width: int) -> list[str]:
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def empty_state(reason: str) -> RenderableType:
    """Shown when there is no turn to read — never analyze text the user didn't pick."""
    t = Text()
    t.append("Nothing to read yet\n\n", style="bold " + R.C_GOLD)
    t.append(f"{reason}\n\n", style=R.C_DIM)
    t.append("  e", style="bold " + R.C_ANSWER)
    t.append("  type the text you want to analyze\n", style=R.C_DIM)
    t.append("  r", style="bold " + R.C_ANSWER)
    t.append("  re-run on the latest turn from the chat\n", style=R.C_DIM)
    t.append("  ?", style="bold " + R.C_ANSWER)
    t.append("  what this tab shows and every key\n", style=R.C_DIM)
    return Panel(t, border_style=R.C_GOLD, expand=False)


def lens_screen(*, header: str, grid, readout=None, pins=None, decompose=None,
                interventions=None, hint: str = "") -> RenderableType:
    """Compose the full lens tab."""
    blocks: list[RenderableType] = [Text(header, style="bold " + R.C_GOLD), grid]
    if readout is not None:
        blocks.append(readout)
    if pins is not None:
        blocks.append(pins)
    if decompose is not None:
        blocks.append(decompose)
    if interventions is not None:
        blocks.append(interventions)
    if hint:
        blocks.append(Text(hint, style=R.C_DIM))
    return Group(*blocks)


def live_strip(concepts: list[dict], alerts: list[dict]) -> Text:
    """One-line J-space readout under the live pane while streaming.

    concepts: [{piece, rank}] top workspace tokens for the newest position.
    alerts:   [{piece, rank, prev}] tracked tokens whose rank surged.
    """
    t = Text()
    t.append("⊹ jspace ", style="bold " + R.C_GOLD)
    for c in concepts[:4]:
        t.append(f"{c['piece']!r}", style=R.C_OK)
        t.append(f"·r{c['rank']} ", style=R.C_DIM)
    for a in alerts:
        t.append(f" ⚠ {a['piece']!r} r{a.get('prev','?')}→{a['rank']}", style=R.C_TOOL)
    t.append("   ^j lens", style=R.C_DIM)
    return t
