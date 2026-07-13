"""Phase-2 tests: lens render functions + LensScreen against the mock service."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("numpy")
pytest.importorskip("gguf")
pytest.importorskip("textual")

import numpy as np  # noqa: E402

from rich.console import Console  # noqa: E402

from local_harness.inference.capabilities import Capabilities  # noqa: E402
from local_harness.jlens.model_reader import ReadoutWeights  # noqa: E402
from local_harness.jlens.service import LensService, create_app  # noqa: E402
from local_harness.tui import lens_render as LR  # noqa: E402
from local_harness.tui.render import ability_glyphs  # noqa: E402
from jlens_support import MockModel, MockSidecar  # noqa: E402


def _render(r) -> str:
    con = Console(width=100, record=True, file=open("/dev/null", "w"))
    con.print(r)
    return con.export_text()


def test_lens_grid_marks_cursor_and_output_match():
    T, L, K = 5, 3, 4
    top_ids = np.zeros((T, L, K), dtype=np.int32)
    # make the final column the "output"; column 0 disagrees
    top_ids[:, -1, 0] = [10, 11, 12, 13, 14]
    top_ids[:, 0, 0] = [99, 11, 99, 13, 99]  # some match, some not
    vocab = [f"w{i}" for i in range(120)]
    out = _render(LR.lens_grid(
        pieces=["a", "b", "c", "d", "e"], layers=[0, 5, 11],
        top_ids=top_ids, final_ids=top_ids[:, -1, 0], cursor=(2, 1), vocab=vocab))
    assert "L0" in out and "L11" in out
    assert "›  2" in out or "› 2" in out  # cursor marker on row 2


def test_live_strip_and_glyphs():
    strip = _render(LR.live_strip(
        [{"piece": " yen", "rank": 5}, {"piece": " euro", "rank": 61}],
        [{"piece": " euro", "rank": 38, "prev": 61}]))
    assert "jspace" in strip and "yen" in strip and "⚠" in strip

    caps = Capabilities(activations=True, interventions=True)
    assert "⊹steer" in ability_glyphs(caps)
    caps2 = Capabilities(activations=True, interventions=False)
    assert "⊹lens" in ability_glyphs(caps2)


def test_pins_and_decompose_panels():
    pins = _render(LR.pins_panel([{"piece": " yen", "ranks": [8000, 900, 50, 1]}]))
    assert "yen" in pins and "r1" in pins
    dec = _render(LR.decompose_panel(
        [{"piece": " yen", "coeff": 0.4}, {"piece": " currency", "coeff": 0.2}], explained=0.72))
    assert "yen" in dec and "72% var" in dec


def _mock_service(monkeypatch, sc):
    m = sc.model
    w = ReadoutWeights(arch="mock", n_layers=m.n_layers, d_model=m.d, n_vocab=m.vocab,
                       w_unembed=m.w_unembed, norm_weight=m.norm_weight, norm_bias=None,
                       norm_type="rms", eps=1e-5, n_nextn=0, model_name="mock")
    monkeypatch.setattr(ReadoutWeights, "from_gguf", staticmethod(lambda p: w))
    return LensService(model_path="/mock/toy.gguf", native_url=sc.url, lens_path=None)


@pytest.mark.asyncio
async def test_lens_screen_renders_grid(monkeypatch):
    """Mount LensScreen against a live mock lens service; the grid appears."""
    import uvicorn
    from textual.app import App
    from textual.widgets import Static

    from local_harness.tui.lens_screen import LensScreen

    with MockSidecar() as sc:
        app_svc = create_app(_mock_service(monkeypatch, sc))
        cfg = uvicorn.Config(app_svc, host="127.0.0.1", port=0, log_level="error")
        server = uvicorn.Server(cfg)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        for _ in range(100):
            if server.started and server.servers:
                break
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]

        class Host(App):
            def on_mount(self):
                self.push_screen(LensScreen(f"http://127.0.0.1:{port}",
                                            prompt="a b c d e", can_steer=True))

        async with Host().run_test() as pilot:
            # let the screen's on_mount fetch vocab + slice
            for _ in range(40):
                await pilot.pause(0.1)
                screens = [s for s in pilot.app.screen_stack if isinstance(s, LensScreen)]
                if screens and screens[0].slice is not None:
                    break
            ls = [s for s in pilot.app.screen_stack if isinstance(s, LensScreen)][0]
            assert ls.slice is not None
            assert ls.top_ids is not None and ls._layers
            # the grid renders from real slice data
            text = _render(LR.lens_grid(
                pieces=ls.slice["pieces"], layers=ls._layers, top_ids=ls.top_ids,
                final_ids=ls._final_ids(), cursor=ls.cursor, vocab=ls.vocab))
            assert "J-Lens" in text and "L0" in text
            # cursor navigation works
            await ls.action_cursor(1, 0)
            assert ls.cursor[0] == 1
        server.should_exit = True
