import httpx
import pytest

from local_harness.agent.tools import (
    Tool, ToolRegistry, apply_patch, bash, builtin_tools, calculator, edit_file,
    glob, grep, read_file, web_search, webfetch, write_file,
)

_PATCH = (
    "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n"
    " def add(a, b):\n-    return a + b\n+    return a + b + 0\n     pass\n"
)


def test_calculator_arithmetic():
    assert calculator("2 + 3 * 4") == "14"
    assert calculator("(2 + 3) * -4") == "-20"
    assert calculator("2 ** 10 % 7") == "2"


def test_calculator_rejects_code():
    with pytest.raises(Exception):
        calculator("__import__('os').system('id')")
    with pytest.raises(Exception):
        calculator("open('/etc/passwd')")


async def test_registry_executes_and_reports_errors():
    reg = ToolRegistry(builtin_tools())
    assert await reg.execute("calculator", '{"expression": "5*7"}') == "35"
    assert (await reg.execute("nope", "{}")).startswith("error: unknown tool")
    assert (await reg.execute("calculator", "{not json")).startswith("error: invalid JSON")
    # tool exceptions come back as strings the model can read
    assert (await reg.execute("read_file", '{"path": "/does/not/exist"}')).startswith("error:")


async def test_registry_runs_async_tools():
    # an async tool fn is awaited transparently (the MCP/UTCP/webfetch case)
    async def aecho(text: str) -> str:
        return f"async:{text}"
    reg = ToolRegistry([Tool("aecho", "async echo",
                             {"type": "object", "properties": {"text": {"type": "string"}}}, aecho)])
    assert await reg.execute("aecho", '{"text": "hi"}') == "async:hi"


def test_bash_returns_stdout():
    assert bash("echo hello world").strip() == "hello world"


def test_bash_surfaces_nonzero_exit_and_stderr():
    out = bash("echo boom >&2; exit 3")
    assert "exit 3" in out and "boom" in out  # the model can see what failed


def test_bash_times_out_instead_of_hanging():
    out = bash("sleep 30", timeout=1)
    assert "timed out" in out.lower()


def test_bash_output_is_bounded():
    out = bash("yes longline | head -n 100000")  # ~900KB unbounded
    assert len(out) <= 20000  # truncated so it never floods context


async def test_bash_is_a_builtin_tool():
    reg = ToolRegistry(builtin_tools())
    assert await reg.execute("bash", '{"command": "printf 42"}') == "42"


async def test_webfetch_strips_html_to_text():
    def handler(request):
        return httpx.Response(200, text="<html><body><h1>Hi</h1><script>x=1</script><p>world</p></body></html>",
                              headers={"content-type": "text/html"})
    out = await webfetch("http://example.com", transport=httpx.MockTransport(handler))
    assert "Hi world" in out and "<" not in out and "x=1" not in out  # tags + scripts gone


async def test_web_search_defaults_to_keyless_wikipedia(monkeypatch):
    monkeypatch.delenv("LO_SEARCH_URL", raising=False)

    def handler(request):
        assert "wikipedia.org" in request.url.host
        assert dict(request.url.params).get("q") == "rivers"
        return httpx.Response(200, json={"pages": [
            {"title": "River", "key": "River",
             "excerpt": "A <span class=\"searchmatch\">river</span> is a stream."}]})

    out = await web_search("rivers", transport=httpx.MockTransport(handler))
    assert "River" in out and "en.wikipedia.org/wiki/River" in out
    assert "<span" not in out  # excerpt HTML stripped


async def test_web_search_uses_configured_provider(monkeypatch):
    seen = {}

    def handler(request):
        seen["q"] = dict(request.url.params).get("q")
        return httpx.Response(200, text="result one; result two")
    monkeypatch.setenv("LO_SEARCH_URL", "http://search.test/api")
    out = await web_search("rivers", transport=httpx.MockTransport(handler))
    assert seen["q"] == "rivers" and "result one" in out


def test_read_file_whole_and_line_range(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("l1\nl2\nl3\nl4\nl5\n")
    assert read_file(str(p)) == "l1\nl2\nl3\nl4\nl5\n"          # unchanged default
    assert read_file(str(p), start_line=2, end_line=4) == "l2\nl3\nl4\n"
    assert read_file(str(p), start_line=3) == "l3\nl4\nl5\n"     # open end
    assert read_file(str(p), end_line=2) == "l1\nl2\n"           # open start
    assert read_file(str(p), start_line=99) == ""                # past EOF


async def test_read_file_line_range_via_registry(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("a\nb\nc\nd\n")
    reg = ToolRegistry(builtin_tools())
    out = await reg.execute("read_file", f'{{"path": "{p}", "start_line": 2, "end_line": 3}}')
    assert out == "b\nc\n"


def test_write_file_creates_and_writes(tmp_path):
    p = tmp_path / "sub" / "note.txt"
    out = write_file(str(p), "hello world")
    assert p.read_text() == "hello world" and "note.txt" in out


def test_edit_file_replaces_unique_string(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("x = 1\ny = 2\n")
    assert "edited" in edit_file(str(p), "y = 2", "y = 3")
    assert p.read_text() == "x = 1\ny = 3\n"


def test_edit_file_errors_on_missing_or_ambiguous(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("a\na\nb\n")
    assert edit_file(str(p), "nope", "x").startswith("error:")          # not found
    assert "not unique" in edit_file(str(p), "a", "c")                  # 2 matches
    assert edit_file(str(tmp_path / "missing"), "a", "b").startswith("error:")


def test_grep_finds_matches(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("x = 2\n")
    out = grep("def foo", str(tmp_path))
    assert "a.py:1:" in out and "def foo" in out


def test_glob_finds_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "c.py").write_text("")
    out = glob("**/*.py", str(tmp_path))
    assert "a.py" in out and "c.py" in out and "b.txt" not in out


def test_grep_and_glob_honor_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n/dist\n!keep.log\n")
    (tmp_path / "src.py").write_text("TODO real")
    (tmp_path / "a.log").write_text("TODO log")           # ignored (*.log)
    (tmp_path / "keep.log").write_text("TODO kept")        # un-ignored (!keep.log)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "o.py").write_text("TODO build")  # ignored (build/)
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "d.py").write_text("TODO dist")    # ignored (/dist, root)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "dist").mkdir()
    (tmp_path / "sub" / "dist" / "s.py").write_text("TODO subdist")  # kept (/dist anchored)

    g = grep("TODO", str(tmp_path))
    assert "src.py:1:" in g and "keep.log:1:" in g and "sub/dist/s.py" in g
    assert "a.log" not in g and "build/o.py" not in g
    assert "/dist/d.py" not in g  # root dist ignored, but sub/dist kept above

    files = glob("**/*.py", str(tmp_path))
    assert "src.py" in files and "s.py" in files
    assert "build/o.py".replace("/", "/") not in files and "dist/d.py" not in files


def test_glob_without_gitignore_still_prunes_default_dirs(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc.py").write_text("")
    out = glob("**/*.py", str(tmp_path))
    assert "a.py" in out and "__pycache__" not in out


def test_apply_patch_applies_hunk(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("def add(a, b):\n    return a + b\n    pass\n")
    assert apply_patch(_PATCH, str(p)) == f"patched {p}"
    assert p.read_text() == "def add(a, b):\n    return a + b + 0\n    pass\n"


def test_apply_patch_tolerates_line_drift(tmp_path):
    # same hunk, but the target has extra leading lines: content-matching still lands
    p = tmp_path / "f.py"
    p.write_text("# a\n# b\ndef add(a, b):\n    return a + b\n    pass\n")
    assert apply_patch(_PATCH, str(p)).startswith("patched")
    assert "return a + b + 0" in p.read_text()


def test_apply_patch_uses_header_path(tmp_path, monkeypatch):
    (tmp_path / "f.py").write_text("def add(a, b):\n    return a + b\n    pass\n")
    monkeypatch.chdir(tmp_path)
    assert apply_patch(_PATCH) == "patched f.py"  # path from +++ b/f.py header


def test_apply_patch_fails_cleanly_without_writing(tmp_path):
    p = tmp_path / "f.py"
    original = "def sub(a, b):\n    return a - b\n"
    p.write_text(original)
    out = apply_patch(_PATCH, str(p))
    assert out.startswith("error:") and "context not found" in out
    assert p.read_text() == original  # nothing written


async def test_apply_patch_is_a_builtin(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("def add(a, b):\n    return a + b\n    pass\n")
    reg = ToolRegistry(builtin_tools())
    import json
    out = await reg.execute("apply_patch", json.dumps({"patch": _PATCH, "path": str(p)}))
    assert out.startswith("patched")


def test_coding_tools_are_builtins():
    names = {s["function"]["name"] for s in ToolRegistry(builtin_tools()).schemas()}
    assert {"write_file", "edit_file", "grep", "glob", "apply_patch"} <= names


def test_schemas_are_openai_shaped():
    reg = ToolRegistry(builtin_tools())
    for schema in reg.schemas():
        assert schema["type"] == "function"
        assert {"name", "description", "parameters"} <= schema["function"].keys()
