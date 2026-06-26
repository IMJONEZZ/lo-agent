"""Signals, policies, validate-and-retry skill execution, policy resampling."""

import json

import httpx

from local_harness.agent.loop import Agent
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.log import MODEL_CALL, POLICY_TRIGGERED, EventLog
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.inference.types import TokenLogprob
from local_harness.signals.metrics import StepSignals
from local_harness.signals.policies import Action, StepPolicy
from local_harness.skills.exec import generate_with_skill
from local_harness.skills.ir import Grammar
from local_harness.skills.skill import Skill

from mocks import MockLlamaCpp, chat_response


def lp(logprob, margin=1.0):
    return TokenLogprob("t", logprob, top=[("t", logprob), ("u", logprob - margin)])


def test_signals_metrics():
    s = StepSignals.from_logprobs([lp(-0.1), lp(-0.3), lp(-2.0)])
    assert s.n_tokens == 3
    assert abs(s.mean_logprob - (-0.8)) < 1e-9
    assert s.min_logprob == -2.0
    assert s.mean_top2_margin == 1.0
    assert s.mean_entropy > 0
    assert StepSignals.from_logprobs([]) is None


def test_policy_routes_on_thresholds():
    confident = StepSignals.from_logprobs([lp(-0.05, margin=3.0)] * 5)
    shaky = StepSignals.from_logprobs([lp(-1.5, margin=0.05)] * 5)

    policy = StepPolicy(min_mean_logprob=-0.5)
    assert policy.evaluate(confident).action == Action.ACCEPT
    assert policy.evaluate(shaky).action == Action.RESAMPLE
    assert policy.evaluate(None).action == Action.ACCEPT  # no logprobs = no opinion

    asker = StepPolicy(min_top2_margin=0.5, on_fail=Action.ASK)
    assert asker.evaluate(shaky).action == Action.ASK


YES_NO = Skill(
    name="yes_no",
    grammar=Grammar.from_rules({"v": '"yes" | "no"'}, root="v"),
    sampling_overrides={"temperature": 0.0},
)


async def test_validate_and_retry_on_tier0():
    """Tier-0 server ignores grammar; harness validates and resamples until valid."""
    seq = iter([
        chat_response(content="Well, maybe?"),       # invalid
        chat_response(content="hard to say"),        # invalid
        chat_response(content="yes"),                # valid on attempt 3
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=next(seq))
        return httpx.Response(404)

    client = OpenAICompatClient("http://t", "m", transport=httpx.MockTransport(handler))
    result = await generate_with_skill(client, Capabilities(), YES_NO, "Is water wet?")
    assert result.valid and result.text == "yes" and result.attempts == 3
    assert result.plan.status_of("grammar").value == "emulated"


async def test_constrained_request_carries_grammar():
    """Tier-2 server gets the compiled GBNF in the request body."""
    seen_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            seen_bodies.append(json.loads(request.content))
            return httpx.Response(200, json=chat_response(content="no"))
        return httpx.Response(404)

    caps = Capabilities(server="llama.cpp", seed=True, logprobs=True, grammar="gbnf",
                        logit_bias=True)
    client = OpenAICompatClient("http://t", "m", transport=httpx.MockTransport(handler))
    result = await generate_with_skill(client, caps, YES_NO, "Is fire cold?")
    assert result.valid and result.attempts == 1
    assert 'root ::= ("yes" | "no")' in seen_bodies[0]["grammar"]


async def test_agent_policy_resample(tmp_path):
    """A low-confidence step triggers RESAMPLE: two MODEL_CALLs, one POLICY event."""
    weak = chat_response(content="uh, maybe 5ish?")
    for t in weak["choices"][0]["logprobs"]["content"]:
        t["logprob"] = -3.0
        t["top_logprobs"] = [{"token": "x", "logprob": -3.0}, {"token": "y", "logprob": -3.1}]
    strong = chat_response(content="The answer is 5.")

    script = {1: weak, 1001: strong}  # attempt 1 at seed 1, resample at seed 1+1000
    log = EventLog(tmp_path / "e.db")
    client = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script=script).transport())
    agent = Agent(
        client, ToolRegistry(builtin_tools()), log,
        capabilities=Capabilities(server="llama.cpp", seed=True, logprobs=True),
        policy=StepPolicy(min_mean_logprob=-1.0, max_retries=2), base_seed=1,
    )
    result = await agent.run("what is 2+3?")
    assert result.answer == "The answer is 5."
    run_id = result.run_id
    assert len(log.events(run_id, type=MODEL_CALL)) == 2
    triggers = log.events(run_id, type=POLICY_TRIGGERED)
    assert len(triggers) == 1 and triggers[0].payload["action"] == "resample"
