"""Agent loop: 3-step task, crash resume, pending-tool-call recovery, replay."""

import httpx
import pytest

from local_harness.agent.loop import Agent
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.log import MODEL_CALL, TOOL_CALL, EventLog
from local_harness.events.replay import replay_run
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient

from mocks import MockLlamaCpp, chat_response

TIER1_CAPS = Capabilities(server="llama.cpp", seed=True, logprobs=True)

# Scripted model keyed by seed (agent uses base_seed + call_index = 1, 2, 3):
# two calculator calls, then a final answer.
SCRIPT = {
    1: chat_response(tool_calls=[("c1", "calculator", '{"expression": "2+3"}')]),
    2: chat_response(tool_calls=[("c2", "calculator", '{"expression": "5*7"}')]),
    3: chat_response(content="The answer is 35."),
    424242: chat_response(content="probe"),  # capability probe traffic, if any
}


def make_agent(mock: MockLlamaCpp, log: EventLog) -> Agent:
    client = OpenAICompatClient("http://test", "test-model", transport=mock.transport())
    return Agent(client, ToolRegistry(builtin_tools()), log, capabilities=TIER1_CAPS, base_seed=1)


async def test_three_step_task(tmp_path):
    log = EventLog(tmp_path / "e.db")
    agent = make_agent(MockLlamaCpp(script=SCRIPT), log)
    result = await agent.run("compute (2+3) then multiply by 7")

    assert result.status == "completed"
    assert result.answer == "The answer is 35."
    assert result.model_calls == 3

    calls = log.events(result.run_id, type=MODEL_CALL)
    assert [c.payload["seed"] for c in calls] == [1, 2, 3]
    assert all(c.payload["logprob_summary"]["n_tokens"] == 2 for c in calls)
    tool_events = log.events(result.run_id, type=TOOL_CALL)
    assert [t.payload["result"] for t in tool_events] == ["5", "35"]
    assert log.run(result.run_id).status == "completed"


async def test_continue_a_finished_run(tmp_path):
    log = EventLog(tmp_path / "e.db")
    script = {1: chat_response(content="First answer."),
              2: chat_response(content="Second answer, following up."),
              424242: chat_response(content="probe")}
    r1 = await make_agent(MockLlamaCpp(script=script), log).run("first question")
    assert r1.status == "completed" and r1.answer == "First answer."

    # continue the SAME conversation with a follow-up
    r2 = await make_agent(MockLlamaCpp(script=script), log).continue_run(
        r1.run_id, "a follow-up question")
    assert r2.run_id == r1.run_id                       # same run, not a new one
    assert r2.answer == "Second answer, following up."

    from local_harness.events.log import USER_MESSAGE
    follow = log.events(r1.run_id, type=USER_MESSAGE)
    assert follow and follow[-1].payload["content"] == "a follow-up question"
    assert log.run(r1.run_id).status == "completed"
    # the rebuilt transcript carried the original answer before the follow-up
    calls = log.events(r1.run_id, type=MODEL_CALL)
    assert [c.payload["seed"] for c in calls] == [1, 2]


async def test_streaming_recovers_logprobs_on_llamacpp(tmp_path):
    """On an engine that can't stream logprobs with tools (llama.cpp), the agent
    streams for the typing feel, then recovers logprobs via a seeded re-pass — so
    confidence signals survive streaming."""
    from local_harness.agent.tools import builtin_tools
    from local_harness.signals.policies import StepPolicy
    log = EventLog(tmp_path / "e.db")
    caps = Capabilities(server="llama.cpp", seed=True, logprobs=True)  # stream_logprobs=False
    streamed = []
    client = OpenAICompatClient("http://t", "m",
                                transport=MockLlamaCpp(script={1: chat_response(content="hi")}).transport())
    # a policy is the consumer that triggers the seeded logprob re-pass
    agent = Agent(client, ToolRegistry(builtin_tools()), log, capabilities=caps, base_seed=1,
                  policy=StepPolicy(min_mean_logprob=-99.0),  # never resamples, just needs logprobs
                  on_token=lambda kind, text: streamed.append(kind))
    result = await agent.run("say hi")

    assert result.answer == "hi"
    assert "content" in streamed                      # tokens streamed live
    call = log.events(result.run_id, type=MODEL_CALL)[0]
    assert call.payload["logprob_summary"] is not None  # logprobs recovered despite streaming


async def test_crash_then_resume(tmp_path):
    log = EventLog(tmp_path / "e.db")
    crashing = MockLlamaCpp(script=SCRIPT, fail_after=2)
    agent = make_agent(crashing, log)
    with pytest.raises(httpx.ConnectError):
        await agent.run("compute (2+3) then multiply by 7")

    run_id = log.runs()[0].run_id
    assert log.run(run_id).status == "failed"
    assert len(log.events(run_id, type=MODEL_CALL)) == 2

    healthy = MockLlamaCpp(script=SCRIPT)
    resumed = make_agent(healthy, log)
    result = await resumed.resume(run_id)

    assert result.answer == "The answer is 35."
    assert result.status == "completed"
    # Only the third call was re-issued; completed steps were not redone.
    assert healthy.chat_calls == 1
    assert len(log.events(run_id, type=MODEL_CALL)) == 3
    assert log.run(run_id).status == "completed"


async def test_resume_executes_pending_tool_calls(tmp_path):
    """Crash between logging a model call and executing its tool calls."""
    log = EventLog(tmp_path / "e.db")
    run_id = log.create_run("compute (2+3) then multiply by 7")
    log.append(
        run_id,
        MODEL_CALL,
        {
            "call_index": 0,
            "seed": 1,
            "request_body": {},
            "response": SCRIPT[1],
            "timing_ms": 1.0,
            "logprob_summary": None,
        },
    )
    # no TOOL_CALL event: the process died before executing the calculator

    healthy = MockLlamaCpp(script=SCRIPT)
    agent = make_agent(healthy, log)
    result = await agent.resume(run_id)

    assert result.answer == "The answer is 35."
    tool_events = log.events(run_id, type=TOOL_CALL)
    assert [t.payload["result"] for t in tool_events] == ["5", "35"]


async def test_resume_completed_run_is_a_noop(tmp_path):
    log = EventLog(tmp_path / "e.db")
    agent = make_agent(MockLlamaCpp(script=SCRIPT), log)
    first = await agent.run("compute (2+3) then multiply by 7")

    healthy = MockLlamaCpp(script=SCRIPT)
    again = make_agent(healthy, log)
    result = await again.resume(first.run_id)
    assert result.answer == "The answer is 35."
    assert healthy.chat_calls == 0


async def test_replay_is_bit_identical(tmp_path):
    log = EventLog(tmp_path / "e.db")
    agent = make_agent(MockLlamaCpp(script=SCRIPT), log)
    result = await agent.run("compute (2+3) then multiply by 7")

    replay_client = OpenAICompatClient(
        "http://test", "test-model", transport=MockLlamaCpp(script=SCRIPT).transport()
    )
    report = await replay_run(log, result.run_id, replay_client)
    assert report.identical
    assert report.total == 3
    assert report.original_hash == report.replay_hash


async def test_replay_detects_divergence(tmp_path):
    log = EventLog(tmp_path / "e.db")
    agent = make_agent(MockLlamaCpp(script=SCRIPT), log)
    result = await agent.run("compute (2+3) then multiply by 7")

    diverged_script = dict(SCRIPT)
    diverged_script[3] = chat_response(content="Something else entirely.")
    replay_client = OpenAICompatClient(
        "http://test", "test-model", transport=MockLlamaCpp(script=diverged_script).transport()
    )
    report = await replay_run(log, result.run_id, replay_client)
    assert not report.identical
    assert report.matched == 2
    assert report.mismatches[0].seq == log.events(result.run_id, type=MODEL_CALL)[-1].seq


class _PickyTransport(httpx.AsyncBaseTransport):
    """A provider that 400s whenever logprobs/seed are present — like several
    OpenAI-compatible servers that other harnesses work with only because they
    never send those params."""

    def __init__(self) -> None:
        self.saw_logprobs = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(request.content) if request.content else {}
        if "logprobs" in body or "seed" in body:
            self.saw_logprobs = True
            return httpx.Response(400, json={"error": "logprobs unsupported"})
        if body.get("stream"):  # the agent always streams now
            sse = ('data: {"choices":[{"index":0,"delta":{"content":"done"}}]}\n\n'
                   'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                   'data: [DONE]\n\n')
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        msg = {"role": "assistant", "content": "done", "tool_calls": None}
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}})


async def test_agent_degrades_when_server_rejects_logprobs(tmp_path):
    transport = _PickyTransport()
    client = OpenAICompatClient("http://picky", "m", transport=transport)
    log = EventLog(tmp_path / "e.db")
    notices: list[str] = []
    # caps say logprobs/seed are supported (a false-positive probe), so the agent
    # sends them; the run must still complete by stripping them on the 400.
    agent = Agent(client, ToolRegistry(builtin_tools()), log,
                  capabilities=Capabilities(server="generic", seed=True, logprobs=True),
                  base_seed=1, max_steps=3, on_notice=notices.append)
    result = await agent.run("say hello")
    assert result.status == "completed" and result.answer == "done"
    assert transport.saw_logprobs            # it really did reject the first body
    assert notices and "logprobs" in notices[0]


class _BodyRecorder(httpx.AsyncBaseTransport):
    """Records every request body and streams a native tool_call then an answer,
    so we can assert the agent never sends the logprobs+tools+stream combo that
    real servers (vLLM, llama.cpp) reject."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.n = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(request.content)
        self.bodies.append(body)
        self.n += 1
        if body.get("stream"):
            if self.n == 1:
                chunks = [{"choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "c1", "function": {
                        "name": "calculator", "arguments": '{"expression":"6*7"}'}}]},
                    "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}]
            else:
                chunks = [{"choices": [{"index": 0, "delta": {"content": "42."},
                                        "finish_reason": None}]},
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
            text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        msg = {"role": "assistant", "content": "42.", "tool_calls": None}
        return httpx.Response(200, json={"choices": [
            {"index": 0, "message": msg, "finish_reason": "stop"}]})


async def test_never_streams_logprobs_with_tools(tmp_path):
    # even on a server that advertises stream_logprobs, a tool-calling turn must
    # not send logprobs+tools+stream together (the combo real servers 400 on).
    rec = _BodyRecorder()
    client = OpenAICompatClient("http://vllm", "m", transport=rec)
    caps = Capabilities(server="vllm", seed=True, logprobs=True, stream_logprobs=True)
    log = EventLog(tmp_path / "e.db")
    agent = Agent(client, ToolRegistry(builtin_tools()), log, capabilities=caps,
                  base_seed=1, max_steps=4, on_token=lambda k, t: None)
    await agent.run("What is 6*7? Use the calculator.")
    bad = [b for b in rec.bodies
           if b.get("stream") and b.get("logprobs") and b.get("tools")]
    assert not bad, "must never stream logprobs+tools together"


async def test_agent_requests_post_sampling_probs_when_supported(tmp_path):
    log = EventLog(tmp_path / "e.db")
    mock = MockLlamaCpp(script=SCRIPT, post_sampling=True)
    client = OpenAICompatClient("http://test", "test-model", transport=mock.transport())
    caps = Capabilities(
        server="llama.cpp", seed=True, logprobs=True, post_sampling_probs=True
    )
    agent = Agent(client, ToolRegistry(builtin_tools()), log, capabilities=caps, base_seed=1)
    result = await agent.run("compute (2+3) then multiply by 7")

    assert result.status == "completed"
    assert mock.post_sampling_requests == 3  # every model call asked for them
    calls = log.events(result.run_id, type=MODEL_CALL)
    # signals were computed from prob-shaped fields and carry provenance
    assert all(c.payload["logprob_summary"]["post_sampling"] is True for c in calls)
    assert all(
        abs(c.payload["logprob_summary"]["mean_logprob"] - (-0.275)) < 1e-9
        for c in calls
    )


async def test_agent_skips_post_sampling_probs_when_unsupported(tmp_path):
    log = EventLog(tmp_path / "e.db")
    mock = MockLlamaCpp(script=SCRIPT)
    agent = make_agent(mock, log)
    result = await agent.run("compute (2+3) then multiply by 7")
    assert result.status == "completed"
    assert mock.post_sampling_requests == 0
    calls = log.events(result.run_id, type=MODEL_CALL)
    assert all(c.payload["logprob_summary"]["post_sampling"] is False for c in calls)


async def test_per_agent_model_override(tmp_path):
    """A preset's model override is sent as the request's model field; without it
    the client's model is used."""
    log = EventLog(tmp_path / "e.db")
    script = {1: chat_response(content="done")}

    mock = MockLlamaCpp(script=script)
    client = OpenAICompatClient("http://test", "primary", transport=mock.transport())
    await Agent(client, ToolRegistry(builtin_tools()), log, capabilities=TIER1_CAPS,
                base_seed=1).run("hi")
    assert mock.chat_bodies[-1]["model"] == "primary"

    mock2 = MockLlamaCpp(script=script)
    client2 = OpenAICompatClient("http://test", "primary", transport=mock2.transport())
    await Agent(client2, ToolRegistry(builtin_tools()), log, capabilities=TIER1_CAPS,
                base_seed=1, model="planner").run("hi")
    assert mock2.chat_bodies[-1]["model"] == "planner"


async def test_final_step_warning_lets_the_model_land(tmp_path):
    # With the budget at 3, the harness warns the model before its last call so
    # the run ends with an answer instead of "max_steps exceeded".
    from local_harness.agent.loop import FINAL_STEP_NOTE

    log = EventLog(tmp_path / "e.db")
    mock = MockLlamaCpp(script=SCRIPT)
    agent = make_agent(mock, log)
    agent.max_steps = 3
    result = await agent.run("compute (2+3) then multiply by 7")

    assert result.status == "completed"
    assert result.answer == "The answer is 35."
    # injected as a user turn on exactly the final budgeted call
    final_msgs = mock.chat_bodies[-1]["messages"]
    assert any(m["role"] == "user" and m.get("content") == FINAL_STEP_NOTE
               for m in final_msgs)
    for body in mock.chat_bodies[:-1]:
        assert all(m.get("content") != FINAL_STEP_NOTE for m in body["messages"])


async def test_max_steps_still_fails_if_the_model_keeps_calling(tmp_path):
    from local_harness.agent.loop import FINAL_STEP_NOTE

    log = EventLog(tmp_path / "e.db")
    mock = MockLlamaCpp(script=SCRIPT)
    agent = make_agent(mock, log)
    agent.max_steps = 2  # SCRIPT's first two turns are both tool calls
    result = await agent.run("compute (2+3) then multiply by 7")

    assert result.status == "max_steps"
    assert any(m.get("content") == FINAL_STEP_NOTE
               for m in mock.chat_bodies[-1]["messages"])
