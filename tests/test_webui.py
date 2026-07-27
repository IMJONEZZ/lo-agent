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


def _scripts() -> list[str]:
    return re.findall(r"<script>(.*?)</script>", HTML, re.S)


def _script() -> str:
    """The app script. The first block is the pure markdown renderer."""
    return _scripts()[1]


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
    for i, block in enumerate(_scripts()):
        r = subprocess.run([node, "--input-type=module", "--check"],
                           input=block, capture_output=True, text=True)
        assert r.returncode == 0, f"script block {i}: {r.stderr}"


# --- the markdown renderer ------------------------------------------------
#
# It's pure (no DOM, no app globals) precisely so it can be executed here: the
# transcript renders model and tool output as markup, so escaping is a security
# property, not a formatting preference.

def _md(*sources: str) -> list[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node available to run the markdown renderer")
    src = _scripts()[0] + "\nprocess.stdout.write(JSON.stringify(%s.map(md)))" % (
        json.dumps(list(sources)))
    r = subprocess.run([node, "-e", src], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_fenced_code_becomes_a_scrollable_block():
    out, = _md("here:\n```python\nfor x in xs:\n    print(x)\n```")
    assert '<pre class="code">' in out
    assert "<code>for x in xs:\n    print(x)</code>" in out
    assert "<p>here:</p>" in out
    # the language labels the block; the button says what it does. (They were
    # the same element once, so the chip reading "python" copied when tapped.)
    assert '<span class="lang">python</span>' in out
    assert '<button class="copy" type="button">copy</button>' in out


def test_an_unterminated_fence_still_renders():
    # the common case mid-stream: the closing ``` hasn't arrived yet
    out, = _md("```js\nlet x = 1")
    assert '<pre class="code">' in out and "let x = 1" in out


def test_markup_in_model_output_is_escaped_everywhere():
    para, code, item, quote = _md(
        "<img src=x onerror=alert(1)>",
        "```\n<script>alert(1)</script>\n```",
        "- <b onclick=x>hi</b>",
        "> <iframe src=evil>",
    )
    for out in (para, code, item, quote):
        assert "&lt;" in out
    assert "<img" not in para and "<script>" not in code
    assert "<b onclick" not in item and "<iframe" not in quote


def test_inline_code_bold_and_italic():
    out, = _md("call `lens_grid()` for **bold** and *em*")
    assert "<code>lens_grid()</code>" in out
    assert "<b>bold</b>" in out and "<i>em</i>" in out


def test_snake_case_is_not_italicised():
    # why _italic_ is unsupported: identifiers are far more common than emphasis
    out, = _md("see lens_render.py and _private_name in jlens_service")
    assert "<i>" not in out and "lens_render.py" in out


def test_only_http_links_become_anchors():
    ok, bad, quoted = _md(
        "[docs](https://example.com/a?b=1)",
        "[x](javascript:alert(1))",
        '[a" onmouseover=alert(1)](https://example.com)',
    )
    assert '<a href="https://example.com/a?b=1"' in ok
    assert 'rel="noreferrer noopener"' in ok
    assert "<a" not in bad                  # no javascript: hrefs, ever
    assert 'onmouseover' not in quoted.split("</a>")[0].split(">")[0]
    assert "&quot;" in quoted or '"' not in quoted.split('href="')[1].split('"')[0]


def test_headings_lists_quotes_and_rules():
    out, = _md("## Plan\n- one\n2) two\n\n> a note\n\n---\ntail")
    assert '<div class="h">Plan</div>' in out
    assert "<ul><li>one</li><li>two</li></ul>" in out
    assert "<blockquote>a note</blockquote>" in out
    assert "<hr>" in out and "<p>tail</p>" in out


def test_code_blocks_scroll_sideways_instead_of_wrapping():
    decls = _rule("pre.code")
    assert "overflow-x:auto" in decls and "white-space:pre" in decls


def test_the_streamed_answer_is_upgraded_not_reprinted():
    js = _script()
    # token deltas and the persisted model_call carry the same text; the row is
    # replaced in place, so the answer isn't rendered twice
    assert "liveEl||add(" in js and "row.innerHTML=md(c)" in js


def test_long_tool_output_can_be_expanded():
    js = _script()
    assert "show all (" in js and "classList.toggle(\"open\")" in js
    assert "const CUT=200" in js


def test_copy_works_off_https():
    # a phone hits this over http://192.168.x.x — navigator.clipboard is absent
    js = _script()
    assert "isSecureContext" in js and "execCommand" in js


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
