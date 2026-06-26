"""Prompt optimization → tuned-replay intervention. The mock rewards an
'OPTIMIZED' instruction, so a working optimizer must discover and return it.
"""

import json

import httpx
import pytest

from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.optimize.bootstrap import Example
from local_harness.optimize.prompt_opt import optimize_instruction, instruction_intervention


class _OptServer(httpx.AsyncBaseTransport):
    """Proposal calls return an 'OPTIMIZED' instruction; program runs answer well
    only when the system instruction is the optimized one."""
    async def handle_async_request(self, request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "owned_by": "llamacpp"}]})
        body = json.loads(request.content)
        msgs = body.get("messages", [])
        joined = " ".join((m.get("content") or "") for m in msgs)
        if "improving a task instruction" in joined or "Rewrite this task instruction" in joined:
            content = "OPTIMIZED: reply with just the final word of the input."
        else:
            system = msgs[0]["content"] if msgs else ""
            user = msgs[-1]["content"] if msgs else ""
            content = f"The answer is {user.split()[-1]}" if "OPTIMIZED" in system else "I am unsure."
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}], "usage": {}})


def _client():
    return OpenAICompatClient("http://t", "m", transport=_OptServer())


VALSET = [Example(input="capital of France is Paris", expected="Paris"),
          Example(input="largest planet is Jupiter", expected="Jupiter")]


def _metric(output, ex):
    return 1.0 if ex.expected in output else 0.0


@pytest.mark.parametrize("method", ["gepa", "mipro"])
async def test_optimizer_finds_better_instruction(method):
    caps = Capabilities(server="llama.cpp", seed=True)
    best, score = await optimize_instruction(
        _client(), caps, "Answer the question.", VALSET, _metric, method=method, seed=1)
    assert "OPTIMIZED" in best          # discovered the winning instruction
    assert score == 1.0                 # and it scores perfectly on the valset


async def test_intervention_wraps_instruction():
    iv = instruction_intervention("OPTIMIZED: be concise.", method="gepa")
    assert iv.system_prompt == "OPTIMIZED: be concise."
    assert iv.label == "prompt-opt:gepa"


async def test_unknown_method_raises():
    with pytest.raises(ValueError):
        await optimize_instruction(_client(), Capabilities(), "x", VALSET, _metric, method="nope")
