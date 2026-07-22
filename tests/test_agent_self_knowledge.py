"""The harness self-description injected into the system prompt."""

from __future__ import annotations

from local_harness.agent.loop import harness_system_block
from local_harness.inference.capabilities import Capabilities


def test_block_reports_tier_and_model():
    caps = Capabilities(server="llama.cpp", model="qwen3-27b", seed=True, logprobs=True,
                        context_window=262144)
    block = harness_system_block(caps)
    assert "lo-agent" in block
    assert "qwen3-27b" in block
    assert "llama.cpp" in block
    assert "262,144 tokens" in block
    assert "tier: 1 of 4" in block
    assert "cumulative" in block  # tier 1 implies tier 0


def test_block_separates_available_from_missing():
    caps = Capabilities(seed=True, logprobs=True, kv_snapshot=True)
    block = harness_system_block(caps)
    available, missing = block.split("NOT available here:")
    assert "KV/slot snapshots" in available
    assert "deterministic seeded sampling" in available
    # unprobed features must be named as absent, not silently omitted
    assert "activation steering" in missing
    assert "parallel sampling" in missing


def test_tier_4_lens_pairing_is_described():
    caps = Capabilities(seed=True, logprobs=True, grammar="gbnf", logit_bias=True,
                        kv_snapshot=True, parallel_n=True, raw_completion=True,
                        activations=True, interventions=True)
    block = harness_system_block(caps)
    assert "tier: 4 of 4" in block
    assert "activation read/steer" in block
    assert "J-Lens" in block
    assert "NOT available here" not in block  # everything listed is present


def test_tier_0_block_still_useful():
    block = harness_system_block(Capabilities())
    assert "tier: 0 of 4" in block
    assert "event-sourced" in block
    assert "NOT available here" in block


def test_grammar_dialect_is_named():
    assert "gbnf" in harness_system_block(Capabilities(grammar="gbnf"))
    assert "grammar" not in harness_system_block(Capabilities()).split("NOT available")[0]


def test_unknown_model_does_not_render_none():
    block = harness_system_block(Capabilities(server="", model=""))
    assert "None" not in block
    assert "unknown" in block


def test_agent_injects_block_and_flag_disables_it(tmp_path):
    from local_harness.agent.loop import Agent
    from local_harness.agent.tools import ToolRegistry
    from local_harness.events.log import EventLog
    from local_harness.inference.client import OpenAICompatClient

    client = OpenAICompatClient("http://127.0.0.1:1", "m")
    caps = Capabilities(server="llama.cpp", model="m", seed=True, logprobs=True)
    common = dict(client=client, tools=ToolRegistry(),
                  log=EventLog(str(tmp_path / "a.db")), capabilities=caps,
                  system_prompt="BASE")

    on = Agent(**common)._system_message()
    assert on.content.startswith("BASE")
    assert "About this harness" in on.content

    off = Agent(**{**common, "self_knowledge": False})._system_message()
    assert off.content == "BASE"
