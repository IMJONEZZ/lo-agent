"""Guardrails layer: rescue parsing, validation, step enforcement, error
budgets, compaction, and full agent-loop integration."""

import json

from local_harness.agent.compaction import compact, estimate_tokens
from local_harness.agent.loop import Agent
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.log import GUARDRAIL, MODEL_CALL, EventLog
from local_harness.guardrails.guardrails import Guardrails
from local_harness.guardrails.rescue import rescue_tool_calls
from local_harness.guardrails.steps import StepEnforcer
from local_harness.guardrails.validator import ResponseValidator
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.inference.types import Message, ToolCallRequest

from mocks import MockLlamaCpp, chat_response

TOOLS = ["calculator", "read_file", "list_dir"]
CAPS = Capabilities(server="llama.cpp", seed=True, logprobs=True)


# --- rescue ------------------------------------------------------------


def test_rescue_fenced_and_embedded():
    fenced = 'Sure!\n```json\n{"tool": "calculator", "args": {"expression": "1+1"}}\n```'
    calls = rescue_tool_calls(fenced, TOOLS)
    assert len(calls) == 1 and calls[0].name == "calculator"
    assert json.loads(calls[0].arguments) == {"expression": "1+1"}

    embedded = 'I will call {"name": "list_dir", "arguments": {"path": "."}} now.'
    calls = rescue_tool_calls(embedded, TOOLS)
    assert len(calls) == 1 and calls[0].name == "list_dir"

    multiple = ('{"tool": "calculator", "args": {"expression": "1"}} and then '
                '{"tool": "read_file", "args": {"path": "x"}}')
    assert [c.name for c in rescue_tool_calls(multiple, TOOLS)] == ["calculator", "read_file"]


def test_rescue_rejects_junk():
    assert rescue_tool_calls("just prose, no calls", TOOLS) == []
    assert rescue_tool_calls('{"tool": "nuke", "args": {}}', TOOLS) == []      # unknown tool
    assert rescue_tool_calls('{"tool": "calculator", "args": [1]}', TOOLS) == []  # non-dict args
    assert rescue_tool_calls('{"key": "value"}', TOOLS) == []                  # not a call
    # nested braces inside JSON strings don't break the scanner
    tricky = '{"tool": "calculator", "args": {"expression": "len({1,2})"}}'
    assert len(rescue_tool_calls(tricky, TOOLS)) == 1


def test_rescue_string_arguments_form():
    text = '{"name": "calculator", "arguments": "{\\"expression\\": \\"2*2\\"}"}'
    calls = rescue_tool_calls(text, TOOLS)
    assert len(calls) == 1 and json.loads(calls[0].arguments) == {"expression": "2*2"}


# --- validator ----------------------------------------------------------


def test_validator_paths():
    v = ResponseValidator(TOOLS)
    # plain text = final answer (harness default)
    assert v.validate(Message(role="assistant", content="The answer is 5.")).final
    # text containing a call = rescued
    r = v.validate(Message(role="assistant", content='{"tool": "calculator", "args": {}}'))
    assert r.rescued and r.tool_calls[0].name == "calculator"
    # unknown tool -> tool-channel nudge
    r = v.validate(Message(role="assistant", tool_calls=[ToolCallRequest("1", "nuke", "{}")]))
    assert r.nudge.role == "tool" and r.nudge.kind == "unknown_tool"
    # malformed args -> tool-channel nudge
    r = v.validate(Message(role="assistant", tool_calls=[ToolCallRequest("1", "calculator", "[1,2]")]))
    assert r.nudge.kind == "bad_args"
    # empty text, nothing rescued -> retry nudge on user channel
    r = ResponseValidator(TOOLS, text_is_final=False).validate(
        Message(role="assistant", content="I think I should use the calculator."))
    assert r.nudge.role == "user" and r.nudge.kind == "retry"


# --- step enforcement ----------------------------------------------------


def test_step_enforcer_escalates_and_releases():
    e = StepEnforcer(required_steps=["read_file"], terminal_tools={"submit"})
    first = e.check_tools([ToolCallRequest("1", "submit", "{}")])
    second = e.check_tools([ToolCallRequest("2", "submit", "{}")])
    assert first.kind == "step" and "cannot finish" in first.content
    assert "Pick one" in second.content                      # tier 2 escalation
    assert e.check_tools([ToolCallRequest("3", "read_file", "{}")]) is None
    e.record(["read_file"])
    assert e.check_tools([ToolCallRequest("4", "submit", "{}")]) is None  # released
    assert e.check_finish() is None


def test_step_enforcer_prerequisites():
    e = StepEnforcer(prerequisites={"calculator": ["read_file"]})
    nudge = e.check_tools([ToolCallRequest("1", "calculator", "{}")])
    assert nudge.kind == "prerequisite" and nudge.role == "tool"
    e.record(["read_file"])
    assert e.check_tools([ToolCallRequest("2", "calculator", "{}")]) is None


def test_guardrails_fatal_on_retry_budget():
    g = Guardrails(TOOLS, max_retries=1)
    bad = Message(role="assistant", tool_calls=[ToolCallRequest("1", "nuke", "{}")])
    assert g.check(bad).action == "nudge"
    result = g.check(bad)  # second consecutive failure > budget of 1
    assert result.action == "fatal" and "retry budget" in result.reason


def test_guardrails_tool_error_budget():
    g = Guardrails(TOOLS, max_tool_errors=1)
    assert g.record(["calculator"], had_errors=True) is None
    assert "tool error budget" in g.record(["calculator"], had_errors=True)
    g2 = Guardrails(TOOLS, max_tool_errors=1)
    g2.record(["calculator"], had_errors=True)
    assert g2.record(["calculator"], had_errors=False) is None  # clean batch resets
    assert g2.record(["calculator"], had_errors=True) is None


# --- doom-loop detection --------------------------------------------------


def test_loop_detector_escalates_then_fatal():
    from local_harness.guardrails.loops import LoopDetector

    d = LoopDetector(max_repeats=3, hard_cap=5)
    tc = ToolCallRequest("1", "read_file", '{"path": "x"}')
    assert d.inspect([tc]) is None       # 1
    assert d.inspect([tc]) is None       # 2
    assert d.inspect([tc])[0] == "nudge"  # 3 → nudge once
    assert d.inspect([tc]) is None       # 4 → already nudged, tolerated
    assert d.inspect([tc])[0] == "fatal"  # 5 → hard cap


def test_loop_detector_distinguishes_args():
    from local_harness.guardrails.loops import LoopDetector

    d = LoopDetector(max_repeats=2, hard_cap=4)
    a = ToolCallRequest("1", "read_file", '{"path": "a"}')
    b = ToolCallRequest("2", "read_file", '{"path": "b"}')
    assert d.inspect([a]) is None
    assert d.inspect([b]) is None            # different args — independent count
    assert d.inspect([a])[0] == "nudge"      # a's 2nd → trips; b untouched


def test_guardrails_doom_loop_nudge_then_fatal():
    g = Guardrails(TOOLS, max_repeats=2, max_loop=3)
    msg = Message(role="assistant",
                  tool_calls=[ToolCallRequest("1", "read_file", '{"path": "x"}')])
    assert g.check(msg).action == "execute"      # 1
    r2 = g.check(msg)
    assert r2.action == "nudge" and r2.nudge.kind == "loop"  # 2 → nudge
    assert g.check(msg).action == "fatal"        # 3 → doom loop fatal


def test_guardrails_doom_loop_ignores_varied_calls():
    g = Guardrails(TOOLS, max_repeats=2, max_loop=3)
    for path in ("a", "b", "c", "d"):
        msg = Message(role="assistant",
                      tool_calls=[ToolCallRequest("1", "read_file",
                                                  f'{{"path": "{path}"}}')])
        assert g.check(msg).action == "execute"  # distinct args never trip


# --- agent integration ----------------------------------------------------


def _agent(script, log, **gr_kwargs):
    tools = ToolRegistry(builtin_tools())
    names = [s["function"]["name"] for s in tools.schemas()]
    client = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script=script).transport())
    return Agent(client, tools, log, capabilities=CAPS, base_seed=1,
                 guardrails_factory=lambda: Guardrails(names, **gr_kwargs))


async def test_agent_rescues_text_tool_call(tmp_path):
    """Model emits the call as fenced JSON text; harness rescues and executes."""
    script = {
        1: chat_response(content='```json\n{"tool": "calculator", "args": {"expression": "6*7"}}\n```'),
        2: chat_response(content="It is 42."),
    }
    log = EventLog(tmp_path / "e.db")
    result = await _agent(script, log).run("six times seven?")
    assert result.answer == "It is 42."
    events = log.events(result.run_id, type=GUARDRAIL)
    assert events[0].payload["action"] == "execute" and events[0].payload["rescued"]
    # the rescued call actually ran
    tool_events = [e for e in log.events(result.run_id) if e.type == "tool_call"]
    assert tool_events[0].payload["result"] == "42"


async def test_agent_nudges_unknown_tool_then_recovers(tmp_path):
    script = {
        1: chat_response(tool_calls=[("x1", "wolfram", '{"q": "6*7"}')]),
        2: chat_response(tool_calls=[("x2", "calculator", '{"expression": "6*7"}')]),
        3: chat_response(content="42."),
    }
    log = EventLog(tmp_path / "e.db")
    result = await _agent(script, log).run("six times seven?")
    assert result.answer == "42."
    kinds = [e.payload["kind"] for e in log.events(result.run_id, type=GUARDRAIL)]
    assert kinds[0] == "unknown_tool"
    assert len(log.events(result.run_id, type=MODEL_CALL)) == 3


async def test_agent_enforces_required_steps(tmp_path):
    """Premature text answer gets a step nudge; agent finishes after the
    required tool runs."""
    script = {
        1: chat_response(content="The answer is probably 42."),  # premature finish
        2: chat_response(tool_calls=[("c1", "calculator", '{"expression": "6*7"}')]),
        3: chat_response(content="Verified: 42."),
    }
    log = EventLog(tmp_path / "e.db")
    result = await _agent(script, log, required_steps=["calculator"]).run("six times seven?")
    assert result.answer == "Verified: 42."
    guardrail_events = log.events(result.run_id, type=GUARDRAIL)
    assert guardrail_events[0].payload["kind"] == "step"
    assert guardrail_events[-1].payload["action"] == "final"


async def test_agent_fatal_on_garbage_loop(tmp_path):
    """Model produces unusable output forever; error budget stops the run."""
    script = {seed: chat_response(content="") for seed in range(1, 12)}
    log = EventLog(tmp_path / "e.db")
    result = await _agent(script, log, max_retries=2).run("do something")
    assert result.status == "fatal"
    run = log.runs()[0]
    assert log.run(run.run_id).status == "failed"


async def test_agent_fatal_on_repeated_identical_call(tmp_path):
    """Model repeats one SUCCESSFUL tool call forever; the doom-loop guard stops
    it. The error budget can't — each call succeeds, so had_errors stays False."""
    script = {seed: chat_response(
        tool_calls=[(f"c{seed}", "calculator", '{"expression": "2+2"}')])
        for seed in range(1, 20)}
    log = EventLog(tmp_path / "e.db")
    result = await _agent(script, log, max_repeats=2, max_loop=4).run("loop forever")
    assert result.status == "fatal"
    kinds = [e.payload.get("kind") for e in log.events(result.run_id, type=GUARDRAIL)]
    assert "loop" in kinds  # a doom-loop nudge fired before the fatal stop
    assert log.run(result.run_id).status == "failed"


async def test_agent_resume_rebuilds_step_state(tmp_path):
    """Crash after the required tool ran: resume must not re-demand it."""
    script = {
        1: chat_response(tool_calls=[("c1", "calculator", '{"expression": "6*7"}')]),
        2: chat_response(content="42 it is."),
    }
    log = EventLog(tmp_path / "e.db")
    crashing = _agent(script, log, required_steps=["calculator"])
    crashing.client = OpenAICompatClient(
        "http://t", "m", transport=MockLlamaCpp(script=script, fail_after=1).transport())
    import httpx
    import pytest

    with pytest.raises(httpx.ConnectError):
        await crashing.run("six times seven?")
    run_id = log.runs()[0].run_id

    resumed = await _agent(script, log, required_steps=["calculator"]).resume(run_id)
    assert resumed.answer == "42 it is."


# --- compaction ------------------------------------------------------------


def _msgs():
    out = [Message(role="system", content="sys"), Message(role="user", content="task")]
    for i in range(8):
        out.append(Message(role="assistant", content=None,
                           tool_calls=[ToolCallRequest(f"t{i}", "read_file", '{"path": "x"}')]))
        out.append(Message(role="tool", content=f"line one of result {i}\n" + "data " * 200,
                           tool_call_id=f"t{i}"))
        out.append(Message(role="user", content="nudge text here", name="nudge"))
    out.append(Message(role="assistant", content="recent thinking"))
    return out


def test_compaction_phases():
    messages = _msgs()
    full = estimate_tokens(messages)

    untouched, phase = compact(messages, budget_tokens=full + 100)
    assert phase == 0 and untouched is messages

    dropped, phase = compact(messages, budget_tokens=full - 50)
    assert phase >= 1
    old = dropped[2:-6]
    assert all(m.name != "nudge" for m in old)            # nudges culled first

    squeezed, phase = compact(messages, budget_tokens=full // 4)
    assert phase >= 2
    assert estimate_tokens(squeezed) < full // 2
    assert squeezed[0].content == "sys" and squeezed[1].content == "task"  # sacred
    assert squeezed[-1].content == "recent thinking"                        # recent intact

    collapsed, phase = compact(messages, budget_tokens=60)
    assert phase == 3
    assert any(m.name == "recap" for m in collapsed)
