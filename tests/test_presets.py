"""Agent presets: profiles bundle system prompt + sampling + permissions +
exposed toolset, and the agent honors them."""

from local_harness.agent.loop import Agent
from local_harness.agent.presets import get_preset
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.log import EventLog


def test_presets_define_distinct_profiles():
    build, plan, explore = get_preset("build"), get_preset("plan"), get_preset("explore")
    # build lets read-only run free, asks for writes
    assert build.permissions().decide("read_file") == "allow"
    assert build.permissions().decide("write_file") == "ask"
    # plan/explore deny edits and shell
    for p in (plan, explore):
        assert p.permissions().decide("write_file") == "deny"
        assert p.permissions().decide("bash") == "deny"
        assert p.permissions().decide("grep") == "allow"
    assert "PLAN" in plan.system_prompt and "EXPLORE" in explore.system_prompt
    assert get_preset("nonsense").name == "build"   # unknown falls back


def test_agent_exposes_only_preset_tools(tmp_path):
    explore = get_preset("explore")
    agent = Agent(None, ToolRegistry(builtin_tools()), EventLog(tmp_path / "e.db"),
                  system_prompt=explore.system_prompt, sampling=explore.sampling,
                  exposed_tools=explore.exposed())
    names = {s["function"]["name"] for s in agent._tool_schemas()}
    assert "read_file" in names and "grep" in names      # exposed
    assert "write_file" not in names and "bash" not in names  # hidden in explore
    assert agent.system_prompt == explore.system_prompt


def test_unrestricted_agent_exposes_all_tools(tmp_path):
    agent = Agent(None, ToolRegistry(builtin_tools()), EventLog(tmp_path / "e.db"))
    names = {s["function"]["name"] for s in agent._tool_schemas()}
    assert {"write_file", "bash", "read_file"} <= names  # nothing filtered
