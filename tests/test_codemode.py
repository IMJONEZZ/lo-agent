"""Code-mode: one run_code tool the model writes Python against; chained tool
calls in a single round-trip, with a restricted namespace and exposed-tool policy."""

from __future__ import annotations

import json

from local_harness.agent.codemode import (
    CodeMode, RUN_CODE_NAME, api_reference, run_code_schema,
)
from local_harness.agent.loop import Agent
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.log import EventLog, TOOL_CALL
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient

from mocks import MockLlamaCpp, chat_response


def _reg():
    return ToolRegistry(builtin_tools())


# --- in-process executor ---------------------------------------------------

async def test_code_mode_chains_tools_in_one_block():
    cm = CodeMode(_reg())
    out = await cm.run(
        "vals = []\n"
        "for e in ['2+2', '10*5']:\n"
        "    vals.append(int(await tools.calculator(expression=e)))\n"
        "print('done')\n"
        "return {'sum': sum(vals)}")
    assert '"sum": 54' in out and "[logs]" in out and "done" in out


async def test_code_mode_restricts_namespace():
    cm = CodeMode(_reg())
    assert "error" in await cm.run("import os\nreturn os.listdir('/')")
    assert "error" in await cm.run("return open('/etc/passwd').read()")


async def test_code_mode_enforces_exposed_tools():
    cm = CodeMode(_reg(), exposed={"read_file", "calculator"})
    out = await cm.run('return await tools.write_file(path="x", content="y")')
    assert "isn't available" in out  # write_file not exposed → blocked


# --- agent integration -----------------------------------------------------

def test_code_mode_replaces_schemas_with_run_code():
    agent = Agent(None, _reg(), EventLog(":memory:"), capabilities=Capabilities(),
                  code_mode=True)
    schemas = agent._tool_schemas()
    assert [s["function"]["name"] for s in schemas] == [RUN_CODE_NAME]
    # the api reference (in the description) lists the tools
    desc = schemas[0]["function"]["description"]
    assert "tools.calculator(expression)" in desc and "tools.read_file(path)" in desc


def test_api_reference_respects_exposed():
    ref = api_reference(_reg(), {"read_file", "calculator"})
    assert "tools.read_file(path)" in ref and "tools.calculator" in ref
    assert "write_file" not in ref


async def test_loop_in_code_mode_runs_code_and_uses_a_tool():
    code = json.dumps({"code": 'r = await tools.calculator(expression="6*7")\nreturn r'})
    script = {
        1: chat_response(tool_calls=[("c1", RUN_CODE_NAME, code)]),
        2: chat_response(content="It is 42."),
    }
    client = OpenAICompatClient(
        "http://x", "test-model", transport=MockLlamaCpp(script=script).transport())
    log = EventLog(":memory:")
    agent = Agent(client, _reg(), log, capabilities=Capabilities(), base_seed=1,
                  max_steps=4, code_mode=True)
    result = await agent.run("compute 6*7")
    assert result.status == "completed"
    ran = [e.payload for e in log.events(result.run_id, type=TOOL_CALL)
           if e.payload["name"] == RUN_CODE_NAME]
    assert ran and "42" in ran[0]["result"]


# --- per-request override (server) -----------------------------------------

def test_manager_code_mode_override(tmp_path):
    from test_server import make_manager
    mgr, _ = make_manager(tmp_path)  # its factory builds classic agents (code_mode=False)
    assert mgr._agent_for("r").code_mode is False          # factory default
    assert mgr._agent_for("r", code_mode=True).code_mode is True  # per-request override
