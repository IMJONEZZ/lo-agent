"""The web client is the phone surface — lock the mobile contract.

These assertions look pedantic, but each one is a bug you only see on a real
phone: a sub-16px input silently zooms the whole page on iOS, a `100vh` app
puts the composer under the on-screen keyboard, a missing safe-area inset hides
the send button behind the home indicator, and an unowned EventSource retry
replays the whole transcript on top of itself every time the tab backgrounds.
"""

import json
import re
import shutil
import subprocess

import httpx
import pytest

from local_harness.server.app import create_server_app
from local_harness.server.webui import icon_svg, index_html, manifest_json

from test_server import make_manager

HTML = index_html()


def _css() -> str:
    return re.search(r"<style>(.*?)</style>", HTML, re.S).group(1)


def _script() -> str:
    return re.search(r"<script>(.*?)</script>", HTML, re.S).group(1)


def _rule(selector: str) -> str:
    """The declaration block for a selector, e.g. 'input, textarea'."""
    m = re.search(r"(?m)^\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", _css())
    assert m, f"no CSS rule for {selector!r}"
    return m.group(1)


def test_viewport_opts_into_the_notch():
    meta = re.search(r'<meta name="viewport" content="([^"]+)"', HTML).group(1)
    assert "width=device-width" in meta
    # without viewport-fit=cover the env(safe-area-inset-*) values are all 0
    assert "viewport-fit=cover" in meta


def test_text_inputs_are_16px_so_ios_does_not_zoom():
    decls = _rule("input, textarea")
    size = re.search(r"font-size:\s*(\d+)px", decls)
    assert size and int(size.group(1)) >= 16, decls


def test_layout_survives_the_on_screen_keyboard():
    css = _css()
    # the app box is sized from the *visual* viewport, not the layout viewport
    assert "--app-h:100dvh" in css.replace(" ", "")
    assert "height:var(--app-h)" in css.replace(" ", "")
    js = _script()
    assert "visualViewport" in js and '--app-h' in js


def test_safe_areas_are_padded_top_and_bottom():
    css = _css()
    assert "env(safe-area-inset-bottom" in css and "env(safe-area-inset-top" in css
    # the composer is the bottom-most element, so it must carry the inset
    assert "--safe-b" in _rule("#bottom")


def test_touch_targets_are_at_least_44px():
    assert "--tap:44px" in _css().replace(" ", "")
    for sel in ("button", ".icon", ".sess"):
        assert "var(--tap)" in _rule(sel), sel


def test_sessions_collapse_into_a_drawer_on_phones():
    css = _css()
    assert 'id="menu"' in HTML                      # the ☰ toggle
    assert "translateX(-101%)" in _rule("#side")    # off-canvas by default
    assert "#side.open" in css
    # ...and expands back into the two-pane desktop layout
    wide = re.search(r"@media \(min-width:760px\) \{(.*?)\n  \}", css, re.S).group(1)
    assert "position:static" in wide and "#menu { display:none" in wide
    # the old fixed desktop grid is gone
    assert "grid-template-columns:260px" not in css


def test_the_stream_owns_its_reconnect():
    js = _script()
    onerror = re.search(r"es\.onerror=\(\)=>\{(.*?)\};", js, re.S).group(1)
    # close first: the browser's own retry would replay the log into a
    # transcript that is already rendered
    assert "es.close()" in onerror and "setTimeout" in onerror
    assert "backoff" in onerror
    # returning to a backgrounded tab reattaches
    assert 'addEventListener("visibilitychange"' in js


def test_scrollback_is_not_yanked_to_the_bottom():
    js = _script()
    assert 'id="jump"' in HTML          # jump-to-latest pill
    assert "function near()" in js or "near=()=>" in js
    assert "settle(was)" in js


def test_a_reload_lands_back_in_the_last_session():
    js = _script()
    # phones evict tabs; reopening should resume, not dump you on a blank picker
    assert 'localStorage.setItem("lo.active"' in js
    assert 'localStorage.getItem("lo.active")' in js
    assert "openDrawer(true)" in js      # ...and only then show the picker


def test_the_script_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node available to syntax-check the client")
    r = subprocess.run([node, "--input-type=module", "--check"],
                       input=_script(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_manifest_is_installable():
    m = json.loads(manifest_json())
    assert m["display"] == "standalone"          # no address bar eating the screen
    assert m["start_url"] == "./" and m["scope"] == "./"
    assert m["icons"][0]["src"] == "./icon.svg"
    assert "maskable" in m["icons"][0]["purpose"]
    assert '<link rel="manifest" href="./manifest.webmanifest">' in HTML
    assert 'apple-mobile-web-app-capable' in HTML


async def test_home_screen_assets_are_served(tmp_path):
    mgr, _ = make_manager(tmp_path)
    app = create_server_app(mgr)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://srv"
    ) as c:
        r = await c.get("/")
        assert r.status_code == 200 and "viewport-fit=cover" in r.text

        m = await c.get("/manifest.webmanifest")
        assert m.status_code == 200
        assert m.headers["content-type"].startswith("application/manifest+json")
        assert m.json()["short_name"] == "lo"

        i = await c.get("/icon.svg")
        assert i.status_code == 200
        assert i.headers["content-type"].startswith("image/svg+xml")
        assert i.text.startswith("<svg") and icon_svg() == i.text


# --- getting to it from a phone in the first place ------------------------

def test_serve_banner_hands_over_a_lan_url_when_bound_wide():
    from local_harness.cli.main import _serve_banner

    out = "\n".join(_serve_banner("0.0.0.0", 8099))
    assert "http://localhost:8099" in out
    assert "on your phone" in out and ":8099" in out.split("on your phone")[1]


def test_serve_banner_says_how_when_bound_to_loopback():
    from local_harness.cli.main import _serve_banner

    out = "\n".join(_serve_banner("127.0.0.1", 8099))
    assert "--host 0.0.0.0" in out       # a phone can't reach loopback
