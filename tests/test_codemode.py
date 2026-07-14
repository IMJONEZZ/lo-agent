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


async def test_code_mode_positional_args():
    # Models naturally write positional calls; they must bind to schema params,
    # not raise TypeError (the bug that made code-mode burn turns retrying).
    cm = CodeMode(_reg())
    assert '"391"' in await cm.run('return await tools.calculator("17*23")')
    # positional + keyword both work; and the dotted call() escape hatch too
    assert '"65536"' in await cm.run('return await tools.calculator(expression="2**16")')
    assert '"42"' in await cm.run('return await call("calculator", "6*7")')


async def test_code_mode_too_many_positionals_errors():
    cm = CodeMode(_reg())
    out = await cm.run('return await tools.calculator("1", "2")')
    assert "positional" in out and "error" in out


async def test_code_mode_positional_multi_arg_tool(tmp_path):
    cm = CodeMode(_reg())
    p = tmp_path / "r.txt"
    # write_file(path, content) by position, then read_file(path) by position
    out = await cm.run(
        f'await tools.write_file("{p}", "hi there")\n'
        f'return await tools.read_file("{p}")')
    assert "hi there" in out


async def test_code_mode_restricts_namespace():
    cm = CodeMode(_reg())
    assert "error" in await cm.run("import os\nreturn os.listdir('/')")
    assert "error" in await cm.run("return open('/etc/passwd').read()")


async def test_code_mode_import_error_teaches_the_tools_api():
    # The real-session failure mode: a model writes `import os`, gets an opaque
    # ImportError, and retries the identical code forever. The error must (a)
    # start with "error:" so the loop's tool-error budget counts it, (b) point
    # at the tools API, and (c) not leak harness-internal traceback frames.
    cm = CodeMode(_reg())
    out = await cm.run("import os\nreturn os.getcwd()")
    assert out.startswith("error:")
    assert "tools.list_dir" in out and "tools.read_file" in out
    assert "codemode.py" not in out  # the model sees ITS code, not our plumbing


async def test_code_mode_allows_pure_python_imports():
    # `import re` / `import math` are what models reflexively write; refusing
    # them buys no safety and costs a failed round-trip.
    cm = CodeMode(_reg())
    out = await cm.run(
        "import re\nimport math\nfrom collections import Counter\n"
        "c = Counter('aab')\n"
        "return {'m': re.findall(r'\\d+', 'a1b22')[1], 'pi': round(math.pi, 2), "
        "'top': c.most_common(1)[0][0]}")
    assert '"m": "22"' in out and '"pi": 3.14' in out and '"top": "a"' in out


async def test_code_mode_io_imports_stay_blocked():
    cm = CodeMode(_reg())
    for mod in ("os", "subprocess", "pathlib", "socket", "shutil", "sys"):
        out = await cm.run(f"import {mod}\nreturn 1")
        assert out.startswith("error:") and "isn't available in code mode" in out


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
    assert "tools.calculator(expression)" in desc
    assert "tools.read_file(path" in desc  # now also start_line/end_line


def test_api_reference_respects_exposed():
    ref = api_reference(_reg(), {"read_file", "calculator"})
    assert "tools.read_file(path" in ref and "tools.calculator" in ref
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


def test_code_mode_is_fronted_in_the_system_prompt():
    # The run_code tool description alone isn't enough: local models weight the
    # system prompt over tool schemas, so code-mode usage (chain many calls per
    # block, turns are budgeted) must be stated there too.
    coded = Agent(None, _reg(), EventLog(":memory:"), capabilities=Capabilities(),
                  code_mode=True)
    sys_msg = coded._system_message("task").content
    assert "run_code" in sys_msg and "ONE run_code block" in sys_msg
    # a worked few-shot example is included so models see the exact shape
    assert "await tools.grep(" in sys_msg and "```python" in sys_msg

    classic = Agent(None, _reg(), EventLog(":memory:"), capabilities=Capabilities(),
                    code_mode=False)
    assert "run_code" not in classic._system_message("task").content


async def test_code_mode_few_shot_example_actually_runs():
    """The few-shot in the system prompt must be executable as written — a broken
    example would teach models the wrong shape."""
    from local_harness.agent.loop import CODE_MODE_SYSTEM_NOTE

    # extract the fenced python block from the note and run it through CodeMode
    block = CODE_MODE_SYSTEM_NOTE.split("```python\n", 1)[1].split("```", 1)[0]
    out = await CodeMode(_reg()).run(block)
    assert not out.startswith("error:") and "[result]" in out
