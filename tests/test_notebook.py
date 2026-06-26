"""Self-editing memory: MEMORY.md/USER.md files, the memory + session_search
tools, and the agent injecting the frozen memory snapshot into the system prompt."""

from local_harness.agent.memory import Memory
from local_harness.agent.notebook import Notebook, memory_tool, session_search_tool
from local_harness.agent.tools import ToolRegistry


def test_memory_file_add_replace_remove(tmp_path):
    nb = Notebook(tmp_path)
    assert nb.edit("add", "memory", text="prefers tabs over spaces") == "ok"
    assert "prefers tabs" in nb.memory.read()
    assert nb.edit("replace", "memory", old_text="tabs", text="spaces") == "ok"
    assert "prefers spaces" in nb.memory.read()
    assert nb.edit("remove", "memory", text="prefers spaces over spaces") == "ok"
    assert "prefers" not in nb.memory.read()


def test_no_auto_compaction_overflow_errors(tmp_path):
    nb = Notebook(tmp_path)
    nb.user.limit = 30  # tiny, to force the boundary
    assert nb.edit("add", "user", text="x" * 20) == "ok"
    err = nb.edit("add", "user", text="y" * 20)
    assert err.startswith("error:") and "full" in err   # errors instead of dropping
    assert "y" * 20 not in nb.user.read()               # nothing silently written


def test_system_block_renders_both_files(tmp_path):
    nb = Notebook(tmp_path)
    nb.edit("add", "user", text="name is Sam")
    nb.edit("add", "memory", text="the build command is `uv run pytest`")
    block = nb.system_block()
    assert "USER.md" in block and "name is Sam" in block
    assert "MEMORY.md" in block and "uv run pytest" in block


async def test_memory_tool_via_registry(tmp_path):
    nb = Notebook(tmp_path)
    reg = ToolRegistry([memory_tool(nb)])
    out = await reg.execute("memory", '{"action": "add", "text": "uses ruff"}')
    assert out == "ok" and "uses ruff" in nb.memory.read()


async def test_session_search_tool(tmp_path):
    mem = Memory(tmp_path / "m.db")
    mem.store("lesson", "always run migrations before the test suite")
    reg = ToolRegistry([session_search_tool(mem)])
    out = await reg.execute("session_search", '{"query": "migrations test"}')
    assert "always run migrations" in out


def test_structured_fact_grammar_parse_and_store(tmp_path):
    from local_harness.agent.structured_memory import Fact, fact_grammar, parse_fact

    f = Fact("the build tool", "is", "uv")
    assert f.format() == "the build tool | is | uv"
    assert parse_fact(f.format()) == f
    assert parse_fact("only two | fields") is None       # wrong arity

    g = fact_grammar()
    assert g.validate("the build tool | is | uv")        # valid by construction
    assert not g.validate("only one field")

    mem = Memory(tmp_path / "m.db")
    mem.store("fact", f.format(), agreement=1.0)          # machine-queryable memory
    hit = mem.recall("build tool")[0]
    assert parse_fact(hit.text) == f and hit.agreement == 1.0


def test_agent_injects_memory_into_system_prompt(tmp_path):
    from local_harness.agent.loop import Agent
    from local_harness.agent.tools import builtin_tools
    from local_harness.events.log import EventLog

    nb = Notebook(tmp_path)
    nb.edit("add", "memory", text="the deploy target is staging by default")
    agent = Agent(None, ToolRegistry(builtin_tools()), EventLog(tmp_path / "e.db"), notebook=nb)
    sys_msg = agent._system_message()
    assert "deploy target is staging" in sys_msg.content    # memory is in the system prompt
    assert agent.system_prompt in sys_msg.content           # base prompt preserved


def test_project_scope_separate_dir_and_system_block(tmp_path):
    mem = tmp_path / "mem"
    proj = tmp_path / "proj"
    nb = Notebook(mem, project_dir=proj)
    nb.edit("add", "user", text="name is Sam")
    nb.edit("add", "memory", text="prefers tabs")
    assert nb.edit("add", "project", text="this repo uses uv and pytest") == "ok"
    # PROJECT.md lives in the project dir, not the shared memory dir
    assert (proj / "PROJECT.md").exists()
    assert not (mem / "PROJECT.md").exists()
    block = nb.system_block()
    # ordering: USER -> MEMORY -> PROJECT
    assert block.index("USER.md") < block.index("MEMORY.md") < block.index("PROJECT.md")
    assert "this repo uses uv and pytest" in block


def test_no_project_scope_when_dir_absent(tmp_path):
    nb = Notebook(tmp_path)  # no project_dir
    assert nb.project is None
    assert nb.edit("add", "project", text="x").startswith("error:")
    # the memory tool only offers project when the scope exists
    assert "project" not in memory_tool(nb).parameters["properties"]["target"]["enum"]
    nb2 = Notebook(tmp_path / "m", project_dir=tmp_path / "p")
    assert "project" in memory_tool(nb2).parameters["properties"]["target"]["enum"]


def test_memory_panel_renders(tmp_path):
    import io
    from rich.console import Console
    from local_harness.tui import render
    panel = render.memory_panel(
        [("USER.md", "name is Sam"), ("MEMORY.md", "prefers tabs"), ("PROJECT.md", "uses uv")],
        recall=[("episode", "fixed the parser bug")])
    console = Console(width=80, record=True, file=io.StringIO())
    console.print(panel)
    text = console.export_text()
    assert "memory" in text and "PROJECT.md" in text and "uses uv" in text
    assert "fixed the parser bug" in text


async def test_auto_consolidate_writes_one_episode(tmp_path):
    from local_harness.events.log import EventLog, RUN_COMPLETED
    from local_harness.inference.client import OpenAICompatClient
    from local_harness.tui.app import HarnessApp
    from mocks import MockLlamaCpp, chat_response

    db = str(tmp_path / "h.db")
    log = EventLog(db)
    rid = log.create_run("fix the parser")
    log.append(rid, RUN_COMPLETED, {"answer": "fixed it"})
    mem = Memory(tmp_path / "m.db")
    client = OpenAICompatClient(
        "http://x", "test-model",
        transport=MockLlamaCpp(script={1: chat_response(content="Fixed the parser bug.")}).transport())

    app = HarnessApp.__new__(HarnessApp)
    app.db_path = db
    app._memory = mem
    app.client = client
    app._stats_dirty = False
    app._refresh_banner = lambda: None

    assert mem.count() == 0
    await app._auto_consolidate()          # the run-end automatic capture
    assert mem.count() == 1 and mem.has_run(rid)
    await app._auto_consolidate()          # idempotent — already summarized
    assert mem.count() == 1
