"""Sandbox seam: host backend behavior, fail-closed factory, and bash routing.

The microVM backend needs KVM + msb installed, so it's validated live (see the
turn log), not in CI; here we cover the host backend and the routing contract.
"""

import pytest

from local_harness.sandbox import (
    HostSandbox, Sandbox, SandboxUnavailable, make_sandbox,
)
from local_harness.agent.tools import ToolRegistry, builtin_tools


async def test_host_sandbox_runs_and_reports_exit_code(tmp_path):
    sb = make_sandbox("host", tmp_path)
    assert isinstance(sb, HostSandbox) and sb.kind == "host"
    out, code = await sb.exec("echo hello && pwd")
    assert "hello" in out and str(tmp_path) in out and code == 0
    out, code = await sb.exec("exit 3")
    assert code == 3
    await sb.aclose()


def test_factory_is_fail_closed_for_unavailable_microvm(tmp_path, monkeypatch):
    # Force "unavailable" and assert we RAISE rather than silently use the host.
    monkeypatch.setattr("local_harness.sandbox.microvm_ready", lambda: (False, "no msb"))
    with pytest.raises(SandboxUnavailable):
        make_sandbox("microvm", tmp_path)
    # fail_closed=False downgrades to host (with a warning) instead
    sb = make_sandbox("microvm", tmp_path, fail_closed=False)
    assert sb.kind == "host"


def test_unknown_sandbox_kind_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_sandbox("nope", tmp_path)


async def test_host_sandbox_file_ops_confined_to_workdir(tmp_path):
    sb = make_sandbox("host", tmp_path)
    assert "wrote" in await sb.write_file("a/b.txt", "hi")
    assert await sb.read_file("a/b.txt") == "hi"
    assert "b.txt" in await sb.list_dir("a")
    # cannot escape the workdir root
    import pytest as _pytest
    with _pytest.raises(ValueError):
        await sb.read_file("../../etc/passwd")


class _FakeBox(Sandbox):
    """A non-host sandbox that records routing (no real VM)."""
    kind = "microvm"

    def __init__(self):
        super().__init__(".")
        self.calls = []
        self.files = {}

    async def exec(self, command, timeout=30):
        self.calls.append(command)
        return f"ran-in-vm: {command}", 0

    async def read_file(self, path, max_bytes=65536):
        return self.files[path]

    async def write_file(self, path, content):
        self.files[path] = content
        return f"wrote {len(content)} chars to {path}"

    async def list_dir(self, path="."):
        return "\n".join(self.files)


async def test_bash_tool_routes_through_a_non_host_sandbox():
    box = _FakeBox()
    reg = ToolRegistry(builtin_tools(sandbox=box))
    result = await reg.execute("bash", '{"command": "ls -la"}')
    assert "ran-in-vm: ls -la" in result
    assert box.calls == ["ls -la"]  # the command went to the sandbox, not the host


async def test_file_tools_route_through_a_non_host_sandbox():
    box = _FakeBox()
    reg = ToolRegistry(builtin_tools(sandbox=box))
    import json
    await reg.execute("write_file", json.dumps({"path": "x.py", "content": "v = 1\n"}))
    assert box.files["x.py"] == "v = 1\n"                       # write went to the sandbox
    assert "v = 1" in await reg.execute("read_file", json.dumps({"path": "x.py"}))
    await reg.execute("edit_file", json.dumps(
        {"path": "x.py", "old_string": "v = 1", "new_string": "v = 99"}))
    assert box.files["x.py"] == "v = 99\n"                      # edit routed read+write
    assert "ran-in-vm: grep" in (await reg.execute("grep", json.dumps({"pattern": "v"})))


async def test_tools_stay_on_host_when_no_sandbox():
    reg = ToolRegistry(builtin_tools())  # default
    assert "on-host" in await reg.execute("bash", '{"command": "echo on-host"}')
