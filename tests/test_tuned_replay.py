"""Tunable replay: re-run a logged answer step under an intervention (optimized
prompt OR grammar/guidance) and produce a different, attributable, reproducible
output. The mock varies its output by what intervention reached it, so we can
assert the knob — not noise — caused the change.
"""

import json

import httpx
import pytest

from local_harness.events.log import EventLog, MODEL_CALL
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.skills.ir import Grammar
from local_harness.skills.skill import Skill
from local_harness.tuned_replay import Intervention, replay_tuned


class _TunedServer(httpx.AsyncBaseTransport):
    """Answers differently depending on the intervention that reached it:
    an 'optimized' system prompt → a concise answer; a grammar param → a
    grammar-conforming token; otherwise the original answer."""
    async def handle_async_request(self, request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "owned_by": "llamacpp"}]})
        body = json.loads(request.content)
        sys = next((m["content"] for m in body.get("messages", []) if m["role"] == "system"), "")
        content = "ORIGINAL ANSWER, long and rambly."
        if "concise" in (sys or "").lower():
            content = "Concise: B&M sued Reckless Ben."
        if body.get("grammar") or body.get("guided_grammar"):
            content = "yes"  # conforms to the yes/no grammar below
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}], "usage": {}})


def _logged_run(tmp_path):
    log = EventLog(tmp_path / "e.db")
    run_id = log.create_run("explain the lego drama")
    log.append(run_id, MODEL_CALL, {
        "call_index": 0, "seed": 1,
        "request_body": {
            "model": "m", "seed": 1, "logprobs": True,
            "tools": [{"type": "function", "function": {"name": "webfetch"}}],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "explain the lego drama"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "webfetch", "arguments": "{}"}}]},
                {"role": "tool", "content": "…evidence…", "tool_call_id": "c1"},
            ]},
        "response": {"choices": [{"message": {"role": "assistant",
                     "content": "ORIGINAL ANSWER, long and rambly."}}]},
    })
    return log, run_id


def _client():
    return OpenAICompatClient("http://t", "m", transport=_TunedServer())


async def test_prompt_intervention_changes_output(tmp_path):
    log, run_id = _logged_run(tmp_path)
    caps = Capabilities(server="llama.cpp", seed=True)
    rep = await replay_tuned(log, run_id, _client(), caps,
                             Intervention(label="gepa", system_prompt="Be extremely concise."))
    assert rep.fork_index == 0                       # defaulted to the last (only) call
    assert "ORIGINAL" in rep.original
    assert "Concise" in rep.tuned and rep.changed    # the optimized prompt reshaped it


async def test_grammar_intervention_constrains_output(tmp_path):
    log, run_id = _logged_run(tmp_path)
    caps = Capabilities(server="llama.cpp", grammar="gbnf", seed=True)
    skill = Skill(name="yesno", grammar=Grammar.from_rules({"root": '"yes" | "no"'}, "root"))
    rep = await replay_tuned(log, run_id, _client(), caps,
                             Intervention(label="grammar:yes_no", skill=skill))
    assert rep.tuned.startswith("yes")
    assert rep.valid is True                         # output satisfies the grammar
    assert rep.grammar_status == "native"            # llama.cpp enforces GBNF
    assert rep.changed


async def test_tuned_replay_is_reproducible(tmp_path):
    log, run_id = _logged_run(tmp_path)
    caps = Capabilities(server="llama.cpp", seed=True)
    iv = Intervention(label="opt", system_prompt="Be concise.", seed=99)
    a = await replay_tuned(log, run_id, _client(), caps, iv)
    b = await replay_tuned(log, run_id, _client(), caps, iv)
    assert a.tuned == b.tuned                         # same intervention → same output


async def test_no_model_calls_raises(tmp_path):
    log = EventLog(tmp_path / "e.db")
    run_id = log.create_run("empty")
    with pytest.raises(ValueError):
        await replay_tuned(log, run_id, _client(), Capabilities(), Intervention(label="x"))
