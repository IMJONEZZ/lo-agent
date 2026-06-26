"""ToolSearch: lazy tool-schema loading. Large toolsets defer the deferrable
(MCP/UTCP) tools behind a tool_search meta-tool that BM25-ranks them and promotes
matches so their schemas appear on the next step."""

from __future__ import annotations

import pytest

from local_harness.agent.loop import TOOL_DEFER_THRESHOLD, Agent
from local_harness.agent.tools import TOOL_SEARCH_NAME, Tool, ToolRegistry
from local_harness.events.log import TOOL_CALL, EventLog
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient

from mocks import MockLlamaCpp, chat_response

_STR_PARAM = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}

# 16 external tools — enough that core(2) + ext(16) > the threshold of 15.
_EXT = [
    ("slack_send", "Send a message to a Slack channel or user"),
    ("pg_query", "Run a SQL query against a Postgres database"),
    ("pg_schema", "Describe the schema of a Postgres table"),
    ("weather_lookup", "Look up the current weather forecast for a location"),
    ("gh_pr", "Open a GitHub pull request"),
    ("gh_issue", "Create a GitHub issue"),
    ("jira_create", "Create a Jira ticket"),
    ("s3_upload", "Upload a file to an AWS S3 bucket"),
    ("email_send", "Send an email message"),
    ("calendar_add", "Add an event to a calendar"),
    ("stripe_charge", "Create a Stripe payment charge"),
    ("notion_page", "Create a Notion page"),
    ("redis_get", "Read a key from Redis"),
    ("dns_lookup", "Resolve a DNS hostname"),
    ("translate_text", "Translate text between languages"),
    ("image_resize", "Resize an image file"),
]


def _registry(weather_value="sunny"):
    core = [Tool(f"core_{i}", f"core builtin {i}", {"type": "object", "properties": {}},
                 (lambda **k: "ok")) for i in range(2)]
    ext = []
    for name, desc in _EXT:
        if name == "weather_lookup":
            fn = lambda q=None, **k: f"weather for {q}: {weather_value}"  # noqa: E731
        else:
            fn = lambda **k: "ran"  # noqa: E731
        ext.append(Tool(name, desc, _STR_PARAM, fn))
    reg = ToolRegistry(core + ext)
    reg.set_deferrable([n for n, _ in _EXT])
    return reg


def _agent(reg, script=None):
    client = OpenAICompatClient(
        "http://x", "test-model", transport=MockLlamaCpp(script=script).transport())
    return Agent(client, reg, EventLog(":memory:"), capabilities=Capabilities(),
                 base_seed=1, max_steps=6)


# --- registry search -------------------------------------------------------

def test_registry_search_ranks_relevant_tools():
    reg = _registry()
    res = reg.search("query a postgres database")
    names = [n for n, _ in res]
    assert "pg_query" in names
    assert res[0][0].startswith("pg_")  # a postgres tool ranks first

    assert reg.search("send a slack message")[0][0] == "slack_send"
    assert reg.search("") == []  # empty query → no matches


def test_set_deferrable_only_keeps_known_tools():
    reg = _registry()
    reg.set_deferrable(["slack_send", "does_not_exist"])
    assert reg.deferrable_names() == {"slack_send"}


# --- schema deferral -------------------------------------------------------

def test_large_toolset_defers_behind_tool_search():
    agent = _agent(_registry())
    names = [s["function"]["name"] for s in agent._tool_schemas()]
    assert TOOL_SEARCH_NAME in names           # the meta-tool is offered
    assert "core_0" in names                   # core is never deferred
    assert "slack_send" not in names           # ext tools are deferred
    # only the meta-tool replaces the 16 deferred ones
    assert len([n for n in names if n in dict(_EXT)]) == 0
    # the count in the tool_search description reflects the deferred tools
    desc = next(s["function"]["description"] for s in agent._tool_schemas()
                if s["function"]["name"] == TOOL_SEARCH_NAME)
    assert "16 more tools" in desc


def test_small_toolset_loads_eagerly():
    reg = ToolRegistry([Tool(f"t{i}", f"d{i}", _STR_PARAM, lambda **k: "x") for i in range(5)])
    reg.set_deferrable([f"t{i}" for i in range(3)])
    names = [s["function"]["name"] for s in _agent(reg)._tool_schemas()]
    assert TOOL_SEARCH_NAME not in names       # <= threshold → no deferral
    assert "t0" in names and "t4" in names
    assert len(reg.search("d1")) >= 0  # search still works regardless


def test_tool_search_promotes_matches_into_schema():
    agent = _agent(_registry())
    out = agent._run_tool_search('{"query": "send a slack message"}')
    assert "slack_send" in out and "slack_send" in agent._promoted
    # now slack_send's real schema is exposed again, tool_search still present for the rest
    names = [s["function"]["name"] for s in agent._tool_schemas()]
    assert "slack_send" in names
    assert TOOL_SEARCH_NAME in names           # 15 others still deferred


def test_tool_search_no_match_message():
    agent = _agent(_registry())
    out = agent._run_tool_search('{"query": "xyzzy nonexistent capability"}')
    assert "no tools matched" in out
    assert not agent._promoted


# --- full loop integration -------------------------------------------------

@pytest.mark.asyncio
async def test_loop_searches_then_calls_the_promoted_tool():
    script = {
        1: chat_response(tool_calls=[("c1", TOOL_SEARCH_NAME, '{"query": "weather forecast"}')]),
        2: chat_response(tool_calls=[("c2", "weather_lookup", '{"q": "NYC"}')]),
        3: chat_response(content="It's sunny in NYC."),
    }
    reg = _registry(weather_value="sunny")
    agent = _agent(reg, script=script)
    result = await agent.run("what's the weather in NYC")
    assert result.status == "completed"
    called = [e.payload["name"] for e in agent.log.events(result.run_id, type=TOOL_CALL)]
    assert TOOL_SEARCH_NAME in called and "weather_lookup" in called
    # the promoted tool actually ran
    weather_results = [e.payload["result"] for e in agent.log.events(result.run_id, type=TOOL_CALL)
                       if e.payload["name"] == "weather_lookup"]
    assert weather_results and "sunny" in weather_results[0]
