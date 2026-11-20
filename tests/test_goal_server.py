"""Goal mode on the session server (server/goals.py, server/sessions.py): the goal
loop runs rounds, asks the judge, runs the verify command, and always stops.
Driven against a mock model server that tells the judge's calls (by their system
prompt) from the agent's."""

from __future__ import annotations

import asyncio
import inspect
import json

import httpx
import pytest

from local_harness.agent import goal as g
from local_harness.agent.goal import goal_state
from local_harness.agent.loop import Agent
from local_harness.agent.presets import get_preset
from local_harness.agent.tools import ToolRegistry, builtin_tools
from local_harness.events.bus import EventBus
from local_harness.events.log import (
    GOAL_CLAIM,
    GOAL_JUDGE,
    GOAL_ROUND,
    GOAL_SET,
    GOAL_STATUS,
    GOAL_VERIFY,
    MODEL_CALL,
    RUN_COMPLETED,
    RUN_FAILED,
    TOOL_CALL,
    USER_MESSAGE,
    EventLog,
)
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.sandbox import HostSandbox
from local_harness.server.app import create_server_app
from local_harness.server.sessions import GoalConflict, SessionManager

from mocks import chat_response

CAPS = Capabilities(server="llama.cpp", seed=True, logprobs=True)


class Model:
    """A mock model server. `agent(body, n)` answers the agent's n-th call and
    `judge(body, n)` the judge's; either may be async (to hang a call) and may
    return an httpx.Response (to fail one)."""

    def __init__(self, agent=None, judge=None):
        self.agent = agent or (lambda body, n: chat_response(content="done"))
        self.judge = judge or (lambda body, n: chat_response(content="false"))
        self.agent_bodies: list[dict] = []
        self.judge_bodies: list[dict] = []

    async def handler(self, request):
        await asyncio.sleep(0)  # yield like a network call, so a runaway loop can't starve the test
        body = json.loads(request.content)
        is_judge = body["messages"][0]["content"] == g.JUDGE_SYSTEM
        bodies = self.judge_bodies if is_judge else self.agent_bodies
        bodies.append(body)
        r = (self.judge if is_judge else self.agent)(body, len(bodies))
        if inspect.isawaitable(r):
            r = await r
        return r if isinstance(r, httpx.Response) else httpx.Response(200, json=r)


def make(tmp_path, model, *, max_steps=95, caps=CAPS, code_mode=False, log=None):
    """A manager whose agents run on `model`, with presets applied the way the
    real server's factory applies them (permissions + exposed toolset)."""
    log = log or EventLog(tmp_path / "e.db")
    transport = httpx.MockTransport(model.handler)

    def factory(on_token, on_tool, on_notice, preset=None):
        client = OpenAICompatClient("http://t", "test-model", transport=transport)
        tools = ToolRegistry(builtin_tools())
        exposed = None
        if preset:
            p = get_preset(preset)
            tools.permissions = p.permissions(approver=None)
            exposed = p.exposed()
        return Agent(client, tools, log, capabilities=caps, base_seed=1, max_steps=max_steps,
                     exposed_tools=exposed, code_mode=code_mode,
                     sandbox=HostSandbox(str(tmp_path)))

    return SessionManager(EventBus(log), factory), log


def tool_call(n, name="calculator", **args):
    return chat_response(tool_calls=[(f"c{n}", name, json.dumps(args))])


def calls_goal(status="complete", summary="done"):
    """An agent that reports on the goal in its first call of every round, then answers."""
    def agent(body, n):
        if body["messages"][-1]["role"] == "tool":
            return chat_response(content="reported")
        return tool_call(n, "goal", status=status, summary=summary)
    return agent


def of(events, type):
    return [e.payload for e in events if e.type == type]


async def settle(mgr, timeout=10):
    """Wait for every turn and goal loop, including turns they start as they end.
    A goal that never stops fails the test here instead of hanging the suite."""
    async def _settle():
        while True:
            await mgr.drain()
            await asyncio.gather(*list(mgr._finalizers))
            if all(t.done() for t in mgr._tasks.values()):
                return
    try:
        await asyncio.wait_for(_settle(), timeout)
    except asyncio.TimeoutError:
        raise AssertionError(f"the session was still running after {timeout}s") from None


# --- every goal ends -------------------------------------------------------------


async def test_a_goal_that_never_finishes_stops_at_the_cap(tmp_path):
    """The LO2 requirement, "even with a /goal active, it must end eventually": a
    model that makes a new tool call on every step and a judge that never agrees
    stop at the cap, after exactly `cap` rounds and a bounded number of calls."""
    model = Model(agent=lambda body, n: tool_call(n, expression=f"{n}+1"))
    mgr, log = make(tmp_path, model, max_steps=2)
    run_id = mgr.start_goal("the number reaches a million", cap=3)
    await settle(mgr)

    events = log.events(run_id)
    assert [p["round"] for p in of(events, GOAL_ROUND)] == [1, 2, 3]
    assert [p["error"] for p in of(events, RUN_FAILED)] == ["max_steps exceeded"] * 3
    assert len(model.agent_bodies) == 3 * 2 and len(model.judge_bodies) == 3
    assert of(events, GOAL_STATUS)[-1] == {
        "status": g.CAPPED, "round": 3,
        "reason": "all 3 rounds were used without meeting the goal"}
    assert not mgr.is_running(run_id)


async def test_rounds_without_new_work_stall(tmp_path):
    model = Model(agent=lambda body, n: chat_response(content="still thinking"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("the tests pass")
    await settle(mgr)
    assert goal_state(log.events(run_id)).status == g.STALLED
    assert len(model.agent_bodies) == g.STALL_ROUNDS


# --- how a goal is met -----------------------------------------------------------


async def test_the_judge_answering_true_meets_the_goal(tmp_path):
    model = Model(judge=lambda body, n: chat_response(content="true"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("say done", judge_model="small-judge")
    await settle(mgr)

    events = log.events(run_id)
    assert goal_state(events).status == g.MET
    judge = of(events, GOAL_JUDGE)[0]
    assert (judge["mode"], judge["answer"], judge["seed"]) == (g.CHECK, "true", 1 + 9_000_000 + 1)
    assert model.judge_bodies[0]["model"] == "small-judge"
    assert "tools" not in model.judge_bodies[0]
    # the judge's call isn't part of the agent's conversation
    assert len(of(events, MODEL_CALL)) == len(model.agent_bodies) == 1


async def test_the_judge_runs_on_the_chats_model_by_default(tmp_path):
    model = Model(judge=lambda body, n: chat_response(content="true"))
    mgr, _ = make(tmp_path, model)
    mgr.start_goal("say done")
    await settle(mgr)
    assert model.judge_bodies[0]["model"] == "test-model"


async def test_the_judges_answer_is_forced_by_a_grammar_when_the_server_has_one(tmp_path):
    model = Model(judge=lambda body, n: chat_response(content="true"))
    mgr, _ = make(tmp_path, model,
                  caps=Capabilities(server="llama.cpp", seed=True, logprobs=True, grammar="gbnf"))
    mgr.start_goal("say done")
    await settle(mgr)
    body = model.judge_bodies[0]
    assert '"true"' in body["grammar"] and body["max_tokens"] == 8
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


async def test_a_complete_claim_gets_a_verdict(tmp_path):
    model = Model(agent=calls_goal(summary="2+3 printed 5"),
                  judge=lambda body, n: chat_response(content="met - the calculator printed 5"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("compute 2+3")
    await settle(mgr)

    events = log.events(run_id)
    assert of(events, GOAL_CLAIM) == [{"round": 1, "kind": "complete", "summary": "2+3 printed 5"}]
    assert of(events, GOAL_JUDGE)[0]["mode"] == g.VERDICT
    assert "The agent says the goal is complete: 2+3 printed 5" in (
        model.judge_bodies[0]["messages"][1]["content"])
    assert of(events, GOAL_STATUS)[-1] == {
        "status": g.MET, "round": 1, "reason": "the calculator printed 5"}


async def test_a_blocked_claim_stops_the_goal(tmp_path):
    model = Model(agent=calls_goal(status="blocked", summary="needs the API key"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("deploy it")
    await settle(mgr)
    assert of(log.events(run_id), GOAL_STATUS)[-1]["status"] == g.BLOCKED
    assert model.judge_bodies == []


async def test_the_verify_command_decides_and_its_output_is_fed_back(tmp_path):
    """A verify command that fails on its first run and passes on its second:
    round 2 opens with the first run's output, and no judge is asked."""
    cmd = 'n=$(cat n 2>/dev/null || echo 0); n=$((n+1)); echo $n > n; echo "run $n"; [ $n -ge 2 ]'
    model = Model(agent=calls_goal())
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("the check passes", verify=cmd)
    await settle(mgr)

    events = log.events(run_id)
    assert [(p["exit_code"], p["output"].split()[-2:]) for p in of(events, GOAL_VERIFY)] == [
        (1, ["run", "1"]), (0, ["run", "2"])]
    round_2 = [p for p in of(events, USER_MESSAGE) if p.get("source") == "goal"][0]["content"]
    assert "This is round 2 of 150" in round_2
    assert "Feedback from the check:" in round_2 and "exited 1:" in round_2 and "run 1" in round_2
    assert goal_state(events).status == g.MET and model.judge_bodies == []


async def test_a_verify_command_that_times_out_pauses_the_goal(tmp_path):
    model = Model(agent=calls_goal())
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("the check passes", verify="sleep 5", verify_timeout=0.2)
    await settle(mgr)
    status = of(log.events(run_id), GOAL_STATUS)[-1]
    assert status["status"] == g.PAUSED
    assert status["reason"].startswith("the verify command is broken")


# --- stopping and failing ----------------------------------------------------------


async def test_an_interrupt_during_the_judge_pauses_the_goal(tmp_path):
    in_flight = asyncio.Event()

    async def judge(body, n):
        in_flight.set()
        await asyncio.Event().wait()  # hangs until cancelled

    model = Model(judge=judge)
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("say done")
    await asyncio.wait_for(in_flight.wait(), 5)
    assert mgr.interrupt(run_id) is True
    await settle(mgr)

    events = log.events(run_id)
    assert of(events, GOAL_STATUS)[-1] == {
        "status": g.PAUSED, "round": 1, "reason": "the goal was interrupted"}
    assert of(events, RUN_FAILED) == []  # the round itself had finished
    assert of(events, GOAL_JUDGE) == [] and not mgr.is_running(run_id)


async def test_an_interrupt_during_a_round_pauses_the_goal(tmp_path):
    in_flight = asyncio.Event()

    async def agent(body, n):
        in_flight.set()
        await asyncio.Event().wait()

    mgr, log = make(tmp_path, Model(agent=agent))
    run_id = mgr.start_goal("say done")
    await asyncio.wait_for(in_flight.wait(), 5)
    mgr.interrupt(run_id)
    await settle(mgr)

    events = log.events(run_id)
    assert of(events, RUN_FAILED) == [{"error": "interrupted"}]
    assert of(events, GOAL_STATUS)[-1]["status"] == g.PAUSED
    assert log.run(run_id).status == "failed"


async def test_a_failing_judge_pauses_the_goal(tmp_path):
    model = Model(judge=lambda body, n: httpx.Response(500, json={"error": "boom"}))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("say done")
    await settle(mgr)
    status = of(log.events(run_id), GOAL_STATUS)[-1]
    assert status["status"] == g.PAUSED
    assert status["reason"].startswith("the judge's model call failed") and "500" in status["reason"]
    assert len(model.judge_bodies) == 1  # asked once, not in a loop


async def test_failed_rounds_are_retried_then_the_goal_pauses(tmp_path):
    model = Model(agent=lambda body, n: httpx.Response(500, json={"error": "boom"}))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("say done")
    await settle(mgr)

    events = log.events(run_id)
    assert len(of(events, GOAL_ROUND)) == g.RETRY_ROUNDS
    retry = [p for p in of(events, USER_MESSAGE) if p.get("source") == "goal"][0]["content"]
    assert retry.startswith("The last round stopped with an error: HTTP 500")
    status = of(events, GOAL_STATUS)[-1]
    assert status["status"] == g.PAUSED and "3 rounds in a row failed" in status["reason"]


async def test_a_round_that_cant_start_pauses_the_goal(tmp_path):
    log = EventLog(tmp_path / "e.db")

    def factory(on_token, on_tool, on_notice, preset=None):
        raise RuntimeError("upstream unreachable")

    mgr = SessionManager(EventBus(log), factory)
    run_id = mgr.start_goal("say done")
    await settle(mgr)
    events = log.events(run_id)
    assert of(events, GOAL_ROUND) == []  # never opened
    assert of(events, GOAL_STATUS)[-1] == {
        "status": g.PAUSED, "round": 1,
        "reason": "round 1 couldn't start: RuntimeError: upstream unreachable"}
    # the run was created for the goal, so its turn needs an end too
    assert of(events, RUN_FAILED) == [{"error": "RuntimeError: upstream unreachable"}]
    assert log.run(run_id).status == "failed"


async def test_a_goal_left_active_by_a_stopped_server_is_paused_at_startup(tmp_path):
    """A goal still active when lo quit has no loop driving it. Unpaused, it would
    leave the session looking busy for good."""
    log = EventLog(tmp_path / "e.db")
    run_id = log.create_run("task")
    log.append(run_id, GOAL_SET, g.goal_set_payload("the tests pass"))
    log.append(run_id, GOAL_ROUND, {"round": 1, "source": "goal"})
    log.append(run_id, USER_MESSAGE, {"content": "round 1", "source": "goal"})

    mgr, _ = make(tmp_path, Model(), log=log)
    events = log.events(run_id)
    assert of(events, RUN_FAILED) == [{"error": "interrupted"}]
    assert of(events, GOAL_STATUS) == [{"status": g.PAUSED, "round": 1,
                                        "reason": "lo stopped while the goal was active"}]
    assert log.run(run_id).status == "failed"


# --- user messages, presets, code mode ---------------------------------------------


async def test_a_message_sent_during_a_goal_opens_the_next_round(tmp_path):
    gate, in_flight = asyncio.Event(), asyncio.Event()

    async def agent(body, n):
        if n == 1:
            in_flight.set()
            await gate.wait()
        return chat_response(content="done")

    model = Model(agent=agent, judge=lambda body, n: chat_response(
        content="true" if n == 2 else "false"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("say done")
    await asyncio.wait_for(in_flight.wait(), 5)
    assert mgr.send(run_id, "also check the docs") == run_id  # queued, not a second turn
    gate.set()
    await settle(mgr)

    events = log.events(run_id)
    assert [(p["round"], p["source"]) for p in of(events, GOAL_ROUND)] == [(1, "goal"), (2, "user")]
    assert of(events, USER_MESSAGE)[-1] == {"content": "also check the docs", "source": "user"}
    assert goal_state(events).status == g.MET


async def test_a_message_still_queued_when_the_goal_is_met_is_sent_after_it(tmp_path):
    gate, in_flight = asyncio.Event(), asyncio.Event()

    async def agent(body, n):
        if n == 1:
            in_flight.set()
            await gate.wait()
        return chat_response(content="done")

    model = Model(agent=agent, judge=lambda body, n: chat_response(content="true"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("say done")
    await asyncio.wait_for(in_flight.wait(), 5)
    mgr.send(run_id, "thanks, now summarise")
    gate.set()
    await settle(mgr)

    events = log.events(run_id)
    types = [e.type for e in events]
    assert types.index(GOAL_STATUS) < types.index(USER_MESSAGE)  # sent once the goal stopped
    assert of(events, USER_MESSAGE)[-1] == {"content": "thanks, now summarise", "source": "user"}
    assert len(of(events, GOAL_ROUND)) == 1 and types[-1] == RUN_COMPLETED


async def test_an_interrupt_drops_queued_messages_and_says_so(tmp_path):
    in_flight = asyncio.Event()

    async def agent(body, n):
        in_flight.set()
        await asyncio.Event().wait()

    mgr, log = make(tmp_path, Model(agent=agent))
    run_id = mgr.start_goal("say done")
    await asyncio.wait_for(in_flight.wait(), 5)
    mgr.send(run_id, "and this")
    mgr.interrupt(run_id)
    await settle(mgr)
    status = of(log.events(run_id), GOAL_STATUS)[-1]
    assert status["status"] == g.PAUSED
    assert status["reason"] == ("the goal was interrupted; 1 message sent during the goal "
                                "wasn't delivered")
    assert mgr.take_queued(run_id) == ""


async def test_the_goal_tool_works_under_a_read_only_preset(tmp_path):
    """plan exposes only read tools and denies anything unlisted; the goal tool is
    added to both its toolset and its allow list for goal rounds."""
    model = Model(agent=calls_goal(), judge=lambda body, n: chat_response(content="met - ok"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("write the plan", preset="plan")
    await settle(mgr)

    names = {t["function"]["name"] for t in model.agent_bodies[0]["tools"]}
    assert "goal" in names and "write_file" not in names
    result = [p for p in of(log.events(run_id), TOOL_CALL) if p["name"] == "goal"][0]["result"]
    assert result.startswith("Recorded")
    assert goal_state(log.events(run_id)).status == g.MET


async def test_the_goal_tool_works_from_code_mode(tmp_path):
    def agent(body, n):
        if body["messages"][-1]["role"] == "tool":
            return chat_response(content="reported")
        code = 'await tools.goal(status="complete", summary="done")\nreturn "ok"'
        return tool_call(n, "run_code", code=code)

    model = Model(agent=agent, judge=lambda body, n: chat_response(content="met - ok"))
    mgr, log = make(tmp_path, model, code_mode=True)
    run_id = mgr.start_goal("report it")
    await settle(mgr)

    tools = model.agent_bodies[0]["tools"]
    assert [t["function"]["name"] for t in tools] == ["run_code"]
    assert "tools.goal(status, summary)" in tools[0]["function"]["description"]
    assert of(log.events(run_id), GOAL_CLAIM) == [
        {"round": 1, "kind": "complete", "summary": "done"}]
    assert goal_state(log.events(run_id)).status == g.MET


# --- a goal on an existing session; pause, resume, clear -----------------------------


async def test_a_new_session_for_a_goal_runs_the_kickoff_as_its_task(tmp_path):
    model = Model(judge=lambda body, n: chat_response(content="true"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start_goal("say done", cap=7)
    await settle(mgr)

    meta = log.run(run_id)
    assert meta.task.startswith("Goal: say done") and "up to 7 rounds" in meta.task
    assert meta.label == "goal: say done"
    first = model.agent_bodies[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user"] and first[1]["content"] == meta.task
    assert of(log.events(run_id), USER_MESSAGE) == []  # round 1 was the run's task


async def test_a_goal_on_an_existing_session_continues_the_conversation(tmp_path):
    model = Model(judge=lambda body, n: chat_response(content="true"))
    mgr, log = make(tmp_path, model)
    run_id = mgr.start("hello")
    await settle(mgr)
    await mgr.goal(run_id, "set", condition="say done")
    await settle(mgr)

    events = log.events(run_id)
    opener = of(events, USER_MESSAGE)[0]
    assert opener["source"] == "goal" and opener["content"].startswith("Goal: say done")
    assert goal_state(events).status == g.MET


async def test_pause_resume_and_clear(tmp_path):
    gate, in_flight = asyncio.Event(), asyncio.Event()

    async def agent(body, n):
        in_flight.set()
        await gate.wait()
        return chat_response(content="done")

    mgr, log = make(tmp_path, Model(agent=agent))
    run_id = mgr.start_goal("say done")
    await asyncio.wait_for(in_flight.wait(), 5)

    summary = await mgr.goal(run_id, "pause")
    assert (summary["status"], summary["reason"], summary["running"]) == (
        g.PAUSED, "paused by the user", False)
    with pytest.raises(GoalConflict):
        await mgr.goal(run_id, "pause")

    in_flight.clear()
    summary = await mgr.goal(run_id, "resume")
    assert (summary["status"], summary["round"], summary["running"]) == (g.ACTIVE, 0, True)
    await asyncio.wait_for(in_flight.wait(), 5)
    rounds = of(log.events(run_id), GOAL_ROUND)
    assert [p["round"] for p in rounds] == [1, 1]  # the new stretch starts at 1 again

    assert await mgr.goal(run_id, "clear") is None
    await settle(mgr)
    events = log.events(run_id)
    assert of(events, GOAL_STATUS)[-1]["status"] == g.CLEARED
    assert goal_state(events) is None and not mgr.is_running(run_id)


async def test_goal_actions_that_dont_fit_the_session(tmp_path):
    gate, in_flight = asyncio.Event(), asyncio.Event()

    async def agent(body, n):
        in_flight.set()
        await gate.wait()
        return chat_response(content="done")

    mgr, log = make(tmp_path, Model(agent=agent, judge=lambda b, n: chat_response(content="true")))
    with pytest.raises(KeyError):
        await mgr.goal("nope", "pause")
    run_id = mgr.start("hello")
    await asyncio.wait_for(in_flight.wait(), 5)
    with pytest.raises(GoalConflict):  # a turn is running
        await mgr.goal(run_id, "set", condition="say done")
    gate.set()
    await settle(mgr)
    for action in ("pause", "resume", "clear"):
        with pytest.raises(GoalConflict):  # no goal yet
            await mgr.goal(run_id, action)
    with pytest.raises(ValueError):
        await mgr.goal(run_id, "set", condition="  ")
    with pytest.raises(ValueError):
        await mgr.goal(run_id, "explode")

    await mgr.goal(run_id, "set", condition="say done")
    await settle(mgr)
    with pytest.raises(GoalConflict):  # met is final: set a new goal instead
        await mgr.goal(run_id, "resume")


# --- HTTP ----------------------------------------------------------------------------


async def test_http_goal_routes(tmp_path):
    model = Model(judge=lambda body, n: chat_response(content="true"))
    mgr, log = make(tmp_path, model)
    app = create_server_app(mgr)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://srv") as c:
        r = await c.post("/session", json={"goal": {"condition": "say done", "cap": 5}})
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        assert r.json()["goal"]["cap"] == 5
        await settle(mgr)

        r = await c.get(f"/session/{run_id}/goal")
        assert r.json()["goal"]["status"] == g.MET

        r = await c.post(f"/session/{run_id}/goal", json={"condition": "say it twice"})
        assert r.status_code == 200 and r.json()["goal"]["condition"] == "say it twice"
        await settle(mgr)
        # the reply's seq is the new goal's GOAL_SET: a client follows events after it
        assert r.json()["seq"] == [e.seq for e in log.events(run_id) if e.type == GOAL_SET][-1]

        assert (await c.post(f"/session/{run_id}/goal",
                             json={"action": "pause"})).status_code == 409
        assert (await c.post(f"/session/{run_id}/goal", json={"condition": ""})).status_code == 400
        assert (await c.post("/session/nope/goal", json={"action": "clear"})).status_code == 404
        assert (await c.get("/session/nope/goal")).status_code == 404
        assert (await c.post("/session", json={"goal": {"condition": ""}})).status_code == 400


async def test_http_a_message_during_a_goal_is_queued(tmp_path):
    gate, in_flight = asyncio.Event(), asyncio.Event()

    async def agent(body, n):
        if n == 1:
            in_flight.set()
            await gate.wait()
        return chat_response(content="done")

    mgr, _ = make(tmp_path, Model(agent=agent, judge=lambda b, n: chat_response(content="true")))
    app = create_server_app(mgr)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://srv") as c:
        r = await c.post("/session", json={"goal": {"condition": "say done"}})
        run_id = r.json()["run_id"]
        await asyncio.wait_for(in_flight.wait(), 5)
        r = await c.post(f"/session/{run_id}/message", json={"content": "and this"})
        assert r.json() == {"run_id": run_id, "queued": True}
        gate.set()
        await settle(mgr)
        r = await c.post(f"/session/{run_id}/message", json={"content": "thanks"})
        assert r.json() == {"run_id": run_id, "queued": False}
        await settle(mgr)
