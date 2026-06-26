"""Closing the self-improvement loop: retrieval-augmented runs (lessons injected
at run start) and Hermes-style auto-skill docs from successful multi-tool runs."""

from local_harness.agent.loop import Agent
from local_harness.agent.memory import Memory
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.background import auto_skills, background_cycle, reflect, summarize_cycle
from local_harness.events.log import RUN_COMPLETED, RUN_FAILED, TOOL_CALL, EventLog
from local_harness.inference.client import OpenAICompatClient

from mocks import MockLlamaCpp, chat_response


def test_retrieval_injects_relevant_lessons(tmp_path):
    mem = Memory(tmp_path / "m.db")
    mem.store("lesson", "always run database migrations before the test suite")
    mem.store("lesson", "the frobnicator is unrelated to this")
    agent = Agent(None, ToolRegistry(builtin_tools()), EventLog(tmp_path / "e.db"), retrieval=mem)

    sys_for_task = agent._system_message("how do I run the test suite?").content
    assert "run database migrations" in sys_for_task   # relevant lesson surfaced
    # no retrieval without a task (e.g. a bare system message)
    assert "run database migrations" not in agent._system_message().content


def _seed_successful_multitool_run(db) -> str:
    log = EventLog(db)
    rid = log.create_run("refactor the parser and add tests")
    for i in range(5):  # >= min_tools
        log.append(rid, TOOL_CALL, {"tool_call_id": f"t{i}", "name": "edit_file",
                                    "arguments": "{}", "result": "edited"})
    log.append(rid, RUN_COMPLETED, {"answer": "done: parser refactored, tests pass"})
    log.close()
    return rid


async def test_auto_skills_writes_doc_for_multitool_run(tmp_path):
    db = str(tmp_path / "e.db")
    _seed_successful_multitool_run(db)
    drafts = tmp_path / "drafts"
    mem = Memory(tmp_path / "m.db")
    client = OpenAICompatClient(
        "http://t", "m",
        transport=MockLlamaCpp(script={1: chat_response(
            content="## Approach\nRead, then edit incrementally.\n## Edge cases\nNone.\n"
                    "## Domain knowledge\nThe parser is recursive-descent.")}).transport())

    created = await auto_skills(EventLog(db), client, str(drafts), memory=mem, min_tools=5)

    assert len(created) == 1
    doc = (drafts / created[0].split("/")[-1]).read_text()
    assert "## Approach" in doc and "recursive-descent" in doc
    # a one-line summary is stored so retrieval surfaces it next time
    assert mem.recall("parser refactor", kind="skill")


async def test_reflect_keeps_self_consistent_lessons_gates_inconsistent(tmp_path):
    # A lesson enters memory only if the model agrees with itself across resamples
    # (self_consistency uses seeds base_seed+0..2 = 200,201,202 on a non-parallel server).
    log = EventLog(tmp_path / "e.db")
    rid = log.create_run("do the thing")
    log.append(rid, RUN_FAILED, {"error": "boom"})

    same = "validate inputs first"
    agree = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script={
        200: chat_response(content=same), 201: chat_response(content=same),
        202: chat_response(content=same)}).transport())
    kept = Memory(tmp_path / "kept.db")
    assert await reflect(log, kept, agree, min_agreement=0.9) == 1
    hit = kept.recall("validate inputs")[0]
    assert hit.agreement == 1.0                         # full agreement recorded

    # three different lessons → agreement 1/3 → below the bar → not stored
    disagree = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script={
        200: chat_response(content="alpha"), 201: chat_response(content="beta"),
        202: chat_response(content="gamma")}).transport())
    gated = Memory(tmp_path / "gated.db")
    assert await reflect(log, gated, disagree, min_agreement=0.9) == 0
    assert gated.count() == 0


async def test_background_cycle_learns_and_summarizes(tmp_path):
    db = str(tmp_path / "e.db")
    _seed_successful_multitool_run(db)                 # → an auto-skill
    log = EventLog(db)
    rid = log.create_run("a failed thing")
    log.append(rid, RUN_FAILED, {"error": "boom"})     # → a reflect lesson
    mem = Memory(tmp_path / "m.db")
    skill_doc = "## Approach\nx\n## Edge cases\ny\n## Domain knowledge\nz"
    lesson = "always check the config first"
    client = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script={
        1: chat_response(content=skill_doc),            # consolidate + auto_skills (seed 1)
        200: chat_response(content=lesson),             # reflect self_consistency (seeds 200-202)
        201: chat_response(content=lesson),
        202: chat_response(content=lesson)}).transport())

    counts = await background_cycle(log, client=client, memory=mem,
                                    drafts_dir=str(tmp_path / "drafts"), min_agreement=0.5)
    assert counts["skills"] >= 1 and counts["lessons"] >= 1
    line = summarize_cycle(counts)
    assert "skills" in line and "$0 spent" in line     # the cost-story readout


async def test_auto_skills_skips_short_runs(tmp_path):
    log = EventLog(tmp_path / "e.db")
    rid = log.create_run("trivial")
    log.append(rid, TOOL_CALL, {"tool_call_id": "t0", "name": "read_file",
                                "arguments": "{}", "result": "x"})
    log.append(rid, RUN_COMPLETED, {"answer": "ok"})
    client = OpenAICompatClient("http://t", "m",
                                transport=MockLlamaCpp(script={1: chat_response(content="x")}).transport())
    created = await auto_skills(log, client, str(tmp_path / "drafts"), min_tools=5)
    assert created == []   # only 1 tool call — below threshold
