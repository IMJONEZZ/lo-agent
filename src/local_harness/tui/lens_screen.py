"""The Jacobian-lens tab: a ModalScreen driving the lens service (Rung 6).

ctrl+j opens it. Fetches a slice for the current turn's tokens, renders the
position×layer heatmap, and lets you navigate cells, pin tokens, decompose,
and build steer/ablate/swap interventions (applied via the service). Read-only
"INSPECT" unless the endpoint reports intervention capability.
"""

from __future__ import annotations

import base64

import httpx
import numpy as np
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from . import lens_render as LR


class LensClient:
    """Async httpx wrapper over the lens service's compact API."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    async def _post(self, path, body):
        async with httpx.AsyncClient(timeout=600) as c:
            r = await c.post(self.base + path, json=body)
            r.raise_for_status()
            return r.json()

    async def _get(self, path, params=None):
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(self.base + path, params=params)
            r.raise_for_status()
            return r.json()

    async def props(self):
        return await self._get("/lens/props")

    async def vocab(self):
        return (await self._get("/lens/vocab"))["pieces"]

    async def slice(self, **body):
        return await self._post("/lens/slice", body)

    async def ranks(self, ctx_id, token_ids):
        return await self._post("/lens/ranks", {"ctx_id": ctx_id, "token_ids": token_ids})

    async def readout(self, ctx_id, pos, layer, top_n=8):
        return await self._post("/lens/readout",
                                {"ctx_id": ctx_id, "pos": pos, "layer": layer, "top_n": top_n})

    async def decompose(self, ctx_id, pos, layer, k=8):
        return await self._post("/lens/decompose",
                                {"ctx_id": ctx_id, "pos": pos, "layer": layer, "k": k})

    async def generate(self, **body):
        return await self._post("/lens/generate", body)

    async def search_tokens(self, q, limit=20):
        return (await self._get("/lens/search_tokens", {"q": q, "limit": limit}))["results"]


def _decode(b64, shape, dtype="<i4"):
    return np.frombuffer(base64.b64decode(b64), dtype).reshape(shape)


class LensScreen(ModalScreen[None]):
    """The lens tab. Data flows: slice → grid; cell nav → readout/decompose."""

    BINDINGS = [
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
        ("up", "cursor(0,-1)", ""),
        ("down", "cursor(0,1)", ""),
        ("left", "cursor(-1,0)", ""),
        ("right", "cursor(1,0)", ""),
        ("p", "pin", "Pin"),
        ("d", "decompose", "Decompose"),
        ("L", "toggle_lens", "Logit-lens"),
        ("g", "generate", "A/B gen"),
        ("s", "steer", "Steer"),
        ("a", "ablate", "Ablate"),
        ("w", "swap", "Swap"),
        ("c", "clear_interventions", "Clear ivs"),
    ]

    def __init__(self, lens_url: str, *, prompt: str | None = None,
                 tokens: list[int] | None = None, can_steer: bool = False):
        super().__init__()
        self.client = LensClient(lens_url)
        self._prompt = prompt
        self._tokens = tokens
        self.can_steer = can_steer
        self.vocab: list[str] = []
        self.slice = None
        self.top_ids = None
        self._layers: list[int] = []
        self.cursor = (0, 0)
        self.pins: list[int] = []
        self.use_lens = True
        self._readout = None
        self._decompose = None
        self._interventions: list[dict] = []
        self._status = "loading…"

    def compose(self) -> ComposeResult:
        with Vertical(id="lensbox"):
            with VerticalScroll(id="lensbody"):
                yield Static(id="lenscontent")
            yield Label("↑↓←→ move · p pin · d decompose · g A/B · L logit-lens · esc close",
                        id="lenshint")

    async def on_mount(self) -> None:
        self.query_one("#lensbody").focus()
        try:
            self.vocab = await self.client.vocab()
            await self._run_slice()
        except Exception as e:  # noqa: BLE001
            self._status = f"lens error: {e}"
            self._repaint()

    async def _run_slice(self):
        self._status = "computing slice…"
        self._repaint()
        body = {"stride": 4, "top_n": 5, "use_lens": self.use_lens}
        if self._tokens:
            body["tokens"] = self._tokens
        elif self._prompt:
            body["prompt"] = self._prompt
        else:
            body["prompt"] = "The quick brown fox jumps over the lazy dog."
        if self._interventions:
            body["interventions"] = self._interventions
        self.slice = await self.client.slice(**body)
        T = len(self.slice["tokens"])
        self._layers = self.slice["layers"]
        self.top_ids = _decode(self.slice["top_ids"], (T, len(self._layers), self.slice["top_n"]))
        self.cursor = (min(self.cursor[0], T - 1), min(self.cursor[1], len(self._layers) - 1))
        self._status = ""
        await self._refresh_cell()

    async def _refresh_cell(self):
        if self.slice is None:
            return
        pos, li = self.cursor
        layer = self._layers[li]
        try:
            self._readout = await self.client.readout(self.slice["ctx_id"], pos, layer)
        except Exception:
            self._readout = None
        self._repaint()

    def _final_ids(self):
        # last column (final layer) top-1 per position
        return self.top_ids[:, -1, 0]

    async def _pin_ranks(self):
        if not self.pins or self.slice is None:
            return []
        r = await self.client.ranks(self.slice["ctx_id"], self.pins)
        shape = r["shape"]
        arr = _decode(r["ranks"], shape)
        out = []
        for j, tid in enumerate(self.pins):
            out.append({"piece": self.vocab[tid], "ranks": [int(arr[-1, li, j]) for li in range(shape[1])]})
        return out

    def _repaint(self):
        if self.slice is None:
            self.query_one("#lenscontent", Static).update(self._status or "…")
            return
        pieces = self.slice["pieces"]
        grid = LR.lens_grid(
            pieces=pieces, layers=self._layers, top_ids=self.top_ids,
            final_ids=self._final_ids(), cursor=self.cursor, vocab=self.vocab,
            boundary=self.slice.get("n_prompt"),
        )
        header = (f"{self.slice.get('generated_text','') and 'turn' or 'prompt'}  "
                  f"{'logit-lens' if not self.use_lens else 'lens'} · "
                  f"{len(self.slice['tokens'])} tok × {len(self._layers)} layers"
                  + (f"  [{self._status}]" if self._status else ""))
        readout = LR.cell_readout_panel(self._readout, self.vocab) if self._readout else None
        pins = LR.pins_panel(self._pins_cache) if getattr(self, "_pins_cache", None) else None
        dec = self._decompose
        iv = LR.interventions_panel(self._interventions, "steer" if self.can_steer else "inspect",
                                    self.vocab)
        content = LR.lens_screen(header=header, grid=grid, readout=readout,
                                 pins=pins, decompose=dec, interventions=iv)
        self.query_one("#lenscontent", Static).update(content)

    # --- actions ---

    async def action_cursor(self, dx: int, dy: int) -> None:
        if self.slice is None:
            return
        T = len(self.slice["tokens"])
        px = max(0, min(T - 1, self.cursor[0] + dx))
        py = max(0, min(len(self._layers) - 1, self.cursor[1] + dy))
        self.cursor = (px, py)
        await self._refresh_cell()

    async def action_pin(self) -> None:
        if self._readout and self._readout["tokens"]:
            tid = self._readout["tokens"][0]["token"]
            if tid not in self.pins:
                self.pins.append(tid)
            self._pins_cache = await self._pin_ranks()
            self._repaint()

    async def action_decompose(self) -> None:
        if self.slice is None:
            return
        pos, li = self.cursor
        try:
            d = await self.client.decompose(self.slice["ctx_id"], pos, self._layers[li])
            self._decompose = LR.decompose_panel(
                d["items"], d["items"][-1].get("explained") if d["items"] else None)
        except Exception as e:  # noqa: BLE001
            self._decompose = LR.decompose_panel([], None)
            self.notify(f"decompose: {e}", severity="warning")
        self._repaint()

    async def action_toggle_lens(self) -> None:
        self.use_lens = not self.use_lens
        await self._run_slice()

    async def action_generate(self) -> None:
        """A/B: continue the prompt with the current intervention set vs baseline."""
        if self.slice is None:
            return
        self._status = "generating A/B…"
        self._repaint()
        try:
            out = await self.client.generate(
                tokens=self.slice["tokens"][: self.slice["n_prompt"]],
                n_predict=24, interventions=self._interventions, compare=True)
            base = out.get("baseline", {}).get("text", "")
            steer = out["steered"]["text"]
            msg = f"baseline: {base!r}\nsteered:  {steer!r}" if base else f"gen: {steer!r}"
            self.notify(msg, timeout=12)
        except Exception as e:  # noqa: BLE001
            self.notify(f"generate: {e}", severity="error")
        self._status = ""
        self._repaint()

    def _cell_token(self):
        """The (id, piece) of the selected cell's top-1 token."""
        pos, li = self.cursor
        tid = int(self.top_ids[pos, li, 0])
        return tid, self.vocab[tid] if 0 <= tid < len(self.vocab) else str(tid)

    async def _add_and_reslice(self, spec: dict) -> None:
        if not self.can_steer:
            self.notify("this endpoint is INSPECT-only (no intervention capability)",
                        severity="warning")
            return
        self._interventions.append(spec)
        self.notify(f"added {spec['type']} — re-slicing with it active")
        await self._run_slice()

    async def action_steer(self) -> None:
        tid, piece = self._cell_token()
        # doctrine: steer defaults to prompt-only (service applies it); α from cell
        await self._add_and_reslice(
            {"type": "steer", "token_id": tid, "piece": piece, "alpha": 2.0})

    async def action_ablate(self) -> None:
        tid, piece = self._cell_token()
        await self._add_and_reslice({"type": "ablate", "token_id": tid, "piece": piece})

    async def action_swap(self) -> None:
        """Swap the two most recently pinned tokens (or last pin ⇄ cell top-1)."""
        tid, piece = self._cell_token()
        if not self.pins:
            self.notify("pin a token first (p), then w swaps it with the current cell")
            return
        a = self.pins[-1]
        await self._add_and_reslice(
            {"type": "swap", "token_a": a, "token_b": tid,
             "piece_a": self.vocab[a], "piece_b": piece})

    async def action_clear_interventions(self) -> None:
        if self._interventions:
            self._interventions = []
            self.notify("cleared interventions")
            await self._run_slice()

    def action_close(self) -> None:
        self.dismiss(None)
