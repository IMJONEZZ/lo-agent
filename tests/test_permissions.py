"""Tiered tool permissions: allow/ask/deny ordering, the approver, and that the
registry actually blocks a denied/unapproved tool before running it."""

from local_harness.agent.permissions import Permissions, default_permissions
from local_harness.agent.tools import ToolRegistry, builtin_tools


def test_decide_order_deny_then_ask_then_allow():
    p = Permissions(allow=["bash"], ask=["bash"], deny=["bash"])
    assert p.decide("bash") == "deny"          # deny wins over ask and allow
    p2 = Permissions(allow=["bash"], ask=["bash"])
    assert p2.decide("bash") == "ask"          # ask wins over allow
    p3 = Permissions(allow=["read_*"])
    assert p3.decide("read_file") == "allow" and p3.decide("bash") == "ask"  # default ask


async def test_allow_runs_deny_blocks():
    allow = Permissions(allow=["calculator"], deny=["bash"])
    reg = ToolRegistry(builtin_tools(), permissions=allow)
    assert await reg.execute("calculator", '{"expression": "6*7"}') == "42"
    out = await reg.execute("bash", '{"command": "echo hi"}')
    assert out.startswith("error:") and "denied" in out   # blocked before running


async def test_ask_approver_gates_and_remembers():
    calls = []

    def approver(tool, args):
        calls.append(tool)
        return tool == "write_file"            # approve writes, reject the rest

    p = Permissions(default="ask", approver=approver)
    reg = ToolRegistry(builtin_tools(), permissions=p)

    bad = await reg.execute("bash", '{"command": "echo no"}')
    assert "not approved" in bad
    # approve write_file once, then a second call is remembered (no re-ask)
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "f.txt")
    assert "wrote" in await reg.execute("write_file", f'{{"path": "{path}", "content": "a"}}')
    await reg.execute("write_file", f'{{"path": "{path}", "content": "b"}}')
    assert calls.count("write_file") == 1      # asked once, then session-allowed


async def test_no_approver_means_ask_is_blocked():
    reg = ToolRegistry(builtin_tools(), permissions=Permissions(default="ask"))
    out = await reg.execute("bash", '{"command": "echo hi"}')
    assert "no approver" in out


async def test_confidence_arg_is_ignored_permissions_are_policy_only():
    # logprob-gated permissions were removed (token-logprob ≠ correctness/safety).
    # `confidence` is still accepted for call-compatibility but must not change the
    # decision: an allowed tool runs regardless of confidence, never escalates to ask.
    asked = []

    def approver(tool, args):
        asked.append(tool)
        return True

    p = Permissions(allow=["calculator"], approver=approver)
    reg = ToolRegistry(builtin_tools(), permissions=p)
    assert await reg.execute("calculator", '{"expression": "6*7"}', confidence=-0.1) == "42"
    assert await reg.execute("calculator", '{"expression": "6*7"}', confidence=-9.9) == "42"
    assert asked == []


def test_deny_always_blocks():
    p = Permissions(deny=["bash"])
    assert p.decide("bash") == "deny"


def test_default_permissions_readonly_allow_writes_ask():
    p = default_permissions()
    assert p.decide("read_file") == "allow" and p.decide("grep") == "allow"
    assert p.decide("bash") == "ask" and p.decide("write_file") == "ask"
    assert p.decide("web_search") == "ask"
