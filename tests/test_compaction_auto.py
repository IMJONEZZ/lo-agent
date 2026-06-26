"""Auto-compaction: context-window detection, the 85% trigger, and the
Claude-Code-style summarizing compaction with its progress callback."""

from __future__ import annotations

import pytest

from local_harness.agent.compaction import (
    SUMMARY_NAME,
    estimate_tokens,
    recent_window,
    summarize_and_compact,
)
from local_harness.agent.loop import Agent
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.log import CONTEXT_COMPACTED, EventLog
from local_harness.inference.adapters.base import Fingerprint
from local_harness.inference.capabilities import Capabilities, detect_context_window, probe
from local_harness.inference.client import OpenAICompatClient
from local_harness.inference.types import Message

from mocks import MockLlamaCpp, chat_response


# --- context-window detection ---------------------------------------------

def test_detect_context_window_llamacpp_props():
    fp = Fingerprint(props={"default_generation_settings": {"n_ctx": 8192}})
    assert detect_context_window(fp) == 8192


def test_detect_context_window_llamacpp_toplevel():
    fp = Fingerprint(props={"n_ctx": 4096})
    assert detect_context_window(fp) == 4096


def test_detect_context_window_vllm_model_card():
    fp = Fingerprint(model_card={"id": "m", "max_model_len": 32768})
    assert detect_context_window(fp) == 32768


def test_detect_context_window_generic_context_length():
    fp = Fingerprint(model_card={"id": "m", "context_length": 16384})
    assert detect_context_window(fp) == 16384


def test_detect_context_window_unknown():
    assert detect_context_window(Fingerprint()) is None
    # ignores junk / non-positive values
    assert detect_context_window(Fingerprint(model_card={"max_model_len": 0})) is None


@pytest.mark.asyncio
async def test_probe_reads_context_window():
    # MockLlamaCpp's /props carries n_ctx here, so probe surfaces it on caps.
    mock = MockLlamaCpp()
    orig = mock.handler

    def handler(request):
        if request.url.path == "/props":
            import httpx
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}})
        return orig(request)

    import httpx
    client = OpenAICompatClient("http://x", "test-model",
                                transport=httpx.MockTransport(handler))
    caps = await probe(client)
    assert caps.context_window == 8192
    assert "auto-compact at 6,963" in caps.summary()  # 0.85 * 8192


# --- the 85% auto-budget on the agent --------------------------------------

def test_agent_auto_budget_from_context_window():
    caps = Capabilities(context_window=10000)
    log = EventLog(":memory:")
    agent = Agent(client=None, tools=ToolRegistry([]), log=log, capabilities=caps)
    assert agent.auto_budget is True
    assert agent.context_budget == 8500  # 0.85 * 10000


def test_explicit_budget_overrides_auto():
    caps = Capabilities(context_window=10000)
    log = EventLog(":memory:")
    agent = Agent(client=None, tools=ToolRegistry([]), log=log, capabilities=caps,
                  context_budget=1234)
    assert agent.auto_budget is False
    assert agent.context_budget == 1234


def test_custom_compact_fraction():
    caps = Capabilities(context_window=10000)
    log = EventLog(":memory:")
    agent = Agent(client=None, tools=ToolRegistry([]), log=log, capabilities=caps,
                  compact_fraction=0.5)
    assert agent.context_budget == 5000


# --- orphan-tool-safe recent window ----------------------------------------

def test_recent_window_drops_leading_orphan_tool():
    tail = [
        Message(role="assistant", content="a"),
        Message(role="tool", content="t1", tool_call_id="1", name="x"),
        Message(role="assistant", content="b"),
    ]
    # keep_recent=2 would start on the tool message; it must be dropped.
    win = recent_window(tail, keep_recent=2)
    assert win[0].role != "tool"
    assert [m.role for m in win] == ["assistant"]


# --- summarizing compaction (Claude-Code parity) ---------------------------

def _summary_client(summary_text="STRUCTURED SUMMARY"):
    # seed 0 is what summarize_and_compact uses; script it to return the summary.
    mock = MockLlamaCpp(script={0: chat_response(content=summary_text, with_logprobs=False)})
    return OpenAICompatClient("http://x", "test-model", transport=mock.transport())


def _long_convo():
    head = [Message(role="system", content="sys"), Message(role="user", content="task")]
    old = []
    for i in range(8):
        old.append(Message(role="assistant", content="",
                           tool_calls=[__import__("local_harness.inference.types", fromlist=["ToolCallRequest"]).ToolCallRequest(id=str(i), name="search", arguments='{"q":"x"}')]))
        old.append(Message(role="tool", content="result " * 50, tool_call_id=str(i), name="search"))
    recent = [Message(role="assistant", content="recent thought"),
              Message(role="user", content="recent follow-up")]
    return head + old + recent


@pytest.mark.asyncio
async def test_summarize_and_compact_replaces_old_with_summary():
    client = _summary_client("STRUCTURED SUMMARY")
    messages = _long_convo()
    before = estimate_tokens(messages)

    events = []
    new_messages, info = await summarize_and_compact(
        client, messages, trigger_tokens=before // 2, context_window=8192,
        on_compact=lambda phase, d: events.append((phase, d.get("frac"))),
    )

    # system + task are preserved verbatim
    assert new_messages[0].content == "sys"
    assert new_messages[1].content == "task"
    # the summary message carries the model's structured summary
    summary_msgs = [m for m in new_messages if m.name == SUMMARY_NAME]
    assert len(summary_msgs) == 1
    assert "STRUCTURED SUMMARY" in summary_msgs[0].content
    # recent turns survive
    assert new_messages[-1].content == "recent follow-up"
    # it actually shrank the context
    assert info["method"] == "summarize"
    assert info["after_tokens"] < info["before_tokens"]
    assert info["summary"] == "STRUCTURED SUMMARY"
    # progress fired start … done
    assert events[0][0] == "start"
    assert events[-1] == ("done", 1.0)


@pytest.mark.asyncio
async def test_summarize_falls_back_to_mechanical_on_error():
    # A client whose summary call errors must NOT crash compaction.
    mock = MockLlamaCpp(fail_after=0)  # any chat call raises
    client = OpenAICompatClient("http://x", "test-model", transport=mock.transport())
    messages = _long_convo()
    new_messages, info = await summarize_and_compact(
        client, messages, trigger_tokens=estimate_tokens(messages) // 3,
    )
    assert info["method"] == "mechanical"
    assert "fallback_reason" in info
    # still produced a valid, shorter transcript
    assert new_messages[0].content == "sys"
    assert estimate_tokens(new_messages) < estimate_tokens(messages)


@pytest.mark.asyncio
async def test_loop_logs_compaction_event_when_over_budget():
    # Mechanical strategy keeps the agent's own seeds clean; we only assert that
    # crossing the trigger emits a CONTEXT_COMPACTED event mid-run.
    from local_harness.inference.capabilities import PROBE_SEED
    script = {
        PROBE_SEED: chat_response(content="probe"),  # capability prober's determinism check
        1: chat_response(tool_calls=[("c1", "search", '{"q":"a"}')]),
        2: chat_response(tool_calls=[("c2", "search", '{"q":"b"}')]),
        3: chat_response(content="final answer"),
    }
    mock = MockLlamaCpp(script=script)
    client = OpenAICompatClient("http://x", "test-model", transport=mock.transport())
    caps = await probe(client)
    log = EventLog(":memory:")
    tools = ToolRegistry(builtin_tools())
    agent = Agent(client, tools, log, capabilities=caps, max_steps=6,
                  context_budget=30, compaction_strategy="mechanical")
    result = await agent.run("do a search task")
    assert result.status == "completed"
    compactions = log.events(result.run_id, type=CONTEXT_COMPACTED)
    assert len(compactions) >= 1
    assert compactions[0].payload["method"] == "mechanical"
