"""Proxy mode: hermetic tests via httpx.ASGITransport (proxy app) with a
mocked upstream (MockLlamaCpp / custom handlers)."""

import json

import httpx
import pytest

from local_harness.events.log import GUARDRAIL, MODEL_CALL, EventLog
from local_harness.inference.client import OpenAICompatClient
from local_harness.proxy.app import create_app
from local_harness.proxy.config import ProxyConfig
from local_harness.proxy.engine import ProxyEngine

from mocks import MockLlamaCpp, chat_response

from local_harness.skills.skill import BUILTIN_SKILLS_DIR

SKILLS_DIR = str(BUILTIN_SKILLS_DIR)

CALC_TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "math",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}}]


class CapturingUpstream:
    """Wraps a base handler, recording every /v1/chat/completions body."""

    def __init__(self, base_handler):
        self.base = base_handler
        self.chat_bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            self.chat_bodies.append(json.loads(request.content))
        return self.base(request)

    def transport(self):
        return httpx.MockTransport(self.handler)


async def make_proxy(upstream_transport, tmp_path, **cfg_kwargs):
    cfg = ProxyConfig(upstream_url="http://upstream", model="test-model",
                      db=str(tmp_path / "proxy.db"), skills_dir=SKILLS_DIR,
                      profiles_dir=str(tmp_path / "profiles"), **cfg_kwargs)
    engine = ProxyEngine(cfg)
    engine.client = OpenAICompatClient("http://upstream", "test-model",
                                       transport=upstream_transport)
    await engine.start()  # ASGITransport doesn't run lifespan; probe manually
    app = create_app(engine)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                               base_url="http://proxy")
    return client, engine


async def test_pipeline_params_injected(tmp_path):
    upstream = CapturingUpstream(MockLlamaCpp(script={0: chat_response(content="yes"),
                                                      424242: chat_response(content="p")}).handler)
    client, _ = await make_proxy(upstream.transport(), tmp_path)
    r = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Is water wet?"}], "seed": 0,
        "harness": {"skill": "yes_no", "samplers": {"min_p": 0.07, "dry": {}}},
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "yes"
    body = upstream.chat_bodies[-1]
    assert 'root ::= ("yes" | "no")' in body["grammar"]            # grammar compiled in
    assert body["min_p"] == 0.07 and body["dry_multiplier"] == 0.8  # sampler zoo lowered
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "harness" not in body                                    # extension stripped


async def test_rescue_promotes_text_to_tool_calls(tmp_path):
    text = 'I should compute it: {"tool": "calculator", "args": {"expression": "6*7"}}'
    upstream = MockLlamaCpp(script={0: chat_response(content=text),
                                    424242: chat_response(content="p")})
    client, engine = await make_proxy(upstream.transport(), tmp_path)
    r = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "6*7?"}], "seed": 0, "tools": CALC_TOOLS,
    })
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "calculator"
    assert json.loads(tc["function"]["arguments"]) == {"expression": "6*7"}
    log = EventLog(engine.cfg.db)
    run = log.runs()[-1]
    assert any(e.payload.get("rescued") for e in log.events(run.run_id, type=GUARDRAIL))


async def test_internal_retry_on_unknown_tool(tmp_path):
    def base(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            if body.get("seed") == 424242:
                return httpx.Response(200, json=chat_response(content="p"))
            n = len(body["messages"])
            if n <= 1:  # first attempt: hallucinated tool
                return httpx.Response(200, json=chat_response(
                    tool_calls=[("x1", "wolfram", '{"q": "6*7"}')]))
            # after nudge (assistant + tool messages appended): correct call
            return httpx.Response(200, json=chat_response(
                tool_calls=[("x2", "calculator", '{"expression": "6*7"}')]))
        return MockLlamaCpp().handler(request)

    upstream = CapturingUpstream(base)
    client, engine = await make_proxy(upstream.transport(), tmp_path)
    r = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "6*7?"}], "tools": CALC_TOOLS,
    })
    choice = r.json()["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "calculator"
    # the client never saw the hallucination; two upstream calls happened
    log = EventLog(engine.cfg.db)
    run = log.runs()[-1]
    assert len(log.events(run.run_id, type=MODEL_CALL)) == 2
    # nudge rode the tool channel upstream
    retry_body = upstream.chat_bodies[-1]
    assert retry_body["messages"][-1]["role"] == "tool"
    assert "does not exist" in retry_body["messages"][-1]["content"]


async def test_emulated_json_schema_validate_retry(tmp_path):
    """Tier-0 upstream can't enforce schemas: proxy strips response_format,
    validates, and retries until the output conforms."""
    calls = {"n": 0}

    def base(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model", "owned_by": "acme"}]})
        if path == "/v1/chat/completions":
            calls["n"] += 1
            if calls["n"] <= 2:  # probe determinism check eats nothing here (no seed match)
                return httpx.Response(200, json=chat_response(content=f"probe-{calls['n']}",
                                                              with_logprobs=False))
            if calls["n"] == 3:
                return httpx.Response(200, json=chat_response(content="not json at all",
                                                              with_logprobs=False))
            return httpx.Response(200, json=chat_response(content='{"name": "Ada"}',
                                                          with_logprobs=False))
        return httpx.Response(404)

    upstream = CapturingUpstream(base)
    client, _ = await make_proxy(upstream.transport(), tmp_path)
    r = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "who?"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {
            "type": "object", "required": ["name"],
            "properties": {"name": {"type": "string"}}}}},
    })
    assert r.json()["choices"][0]["message"]["content"] == '{"name": "Ada"}'
    assert "response_format" not in upstream.chat_bodies[-1]  # stripped for tier-0


async def test_anthropic_messages_round_trip(tmp_path):
    upstream = CapturingUpstream(MockLlamaCpp(script={
        0: chat_response(tool_calls=[("call_1", "calculator", '{"expression": "6*7"}')]),
        424242: chat_response(content="p"),
    }).handler)
    client, _ = await make_proxy(upstream.transport(), tmp_path)
    r = await client.post("/v1/messages", json={
        "model": "claude-x", "max_tokens": 100, "seed": 0,
        "system": "Be precise.",
        "messages": [
            {"role": "user", "content": "6*7?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "prev_1", "name": "calculator",
                 "input": {"expression": "1+1"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "prev_1", "content": "2"}]},
        ],
        "tools": [{"name": "calculator", "description": "math",
                   "input_schema": {"type": "object",
                                    "properties": {"expression": {"type": "string"}}}}],
    })
    out = r.json()
    assert out["type"] == "message" and out["stop_reason"] == "tool_use"
    tool_use = [b for b in out["content"] if b["type"] == "tool_use"][0]
    assert tool_use["name"] == "calculator" and tool_use["input"] == {"expression": "6*7"}

    sent = upstream.chat_bodies[-1]
    assert sent["messages"][0] == {"role": "system", "content": "Be precise."}
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert sent["messages"][2]["tool_calls"][0]["function"]["name"] == "calculator"
    assert sent["tools"][0]["function"]["name"] == "calculator"
    assert sent["max_tokens"] == 100


async def test_streaming_both_dialects(tmp_path):
    upstream = MockLlamaCpp(script={0: chat_response(content="hello there"),
                                    424242: chat_response(content="p")})
    client, _ = await make_proxy(upstream.transport(), tmp_path)

    r = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "seed": 0, "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = [line for line in r.text.split("\n") if line.startswith("data: ")]
    assert chunks[-1] == "data: [DONE]"
    deltas = [json.loads(c[6:]) for c in chunks[:-1]]
    assert any(d["choices"][0]["delta"].get("content") == "hello there" for d in deltas)

    r = await client.post("/v1/messages", json={
        "model": "m", "max_tokens": 10, "seed": 0, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    events = [line.split(" ", 1)[1] for line in r.text.split("\n") if line.startswith("event: ")]
    assert events[0] == "message_start" and events[-1] == "message_stop"
    assert "content_block_delta" in events


async def test_think_budget_path(tmp_path):
    def completion_fn(prompt, body):
        if "</think>" in prompt:
            return " The answer is 42.", "stop"
        return " let me think about this for a while", "length"

    upstream = MockLlamaCpp(script={424242: chat_response(content="p")},
                            completion_fn=completion_fn)
    client, _ = await make_proxy(upstream.transport(), tmp_path)
    r = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "meaning of life?"}],
        "harness": {"think_budget": 8},
    })
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "The answer is 42."
    assert "think" in msg["reasoning_content"]


async def test_health_reports_capabilities(tmp_path):
    upstream = MockLlamaCpp()
    client, _ = await make_proxy(upstream.transport(), tmp_path)
    health = (await client.get("/health")).json()
    assert health["status"] == "ok"
    assert health["capabilities"]["tier"] == 3
    assert health["model"] == "test-model"
