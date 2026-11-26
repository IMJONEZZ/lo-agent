"""Goal mode, the pure part (agent/goal.py): the goal's state and the next step
are rebuilt from the event log alone; the judge's request and answers; the goal
tool; and replay of judge calls."""

from __future__ import annotations

import json

from local_harness.agent import goal as g
from local_harness.agent.goal import Decision, decide, goal_state
from local_harness.agent.loop import Agent
from local_harness.agent.tools import ToolRegistry, builtin_tools
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
from local_harness.events.replay import replay_run
from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient

from mocks import MockLlamaCpp, chat_response

CAPS = Capabilities(server="llama.cpp", seed=True, logprobs=True)


class Run:
    """A goal run written event by event, the way the session server will."""

    def __init__(self, **goal):
        self.log = EventLog(":memory:")
        self.rid = self.log.create_run("task")
        self.calls = 0
        self.log.append(self.rid, GOAL_SET, {"condition": "the tests pass", "cap": 150,
                                             "verify": None, "mode": "goal", "fields": {},
                                             **goal})

    def events(self):
        return self.log.events(self.rid)

    def decide(self) -> Decision:
        return decide(self.events())

    def model_call(self, **reply):
        self.log.append(self.rid, MODEL_CALL, {
            "call_index": self.calls, "seed": 1 + self.calls,
            "request_body": {"messages": [{"role": "user", "content": "round"}]},
            "response": chat_response(**reply)})
        self.calls += 1

    def round(self, n, *, tools=(), end="completed", claim=None, source="goal"):
        self.log.append(self.rid, GOAL_ROUND, {"round": n, "source": source})
        self.log.append(self.rid, USER_MESSAGE, {"content": f"round {n}", "source": source})
        for name, args in tools:
            self.model_call(tool_calls=[(f"c{self.calls}", name, args)])
            self.log.append(self.rid, TOOL_CALL, {"tool_call_id": f"c{self.calls - 1}",
                                                  "name": name, "arguments": args,
                                                  "result": "ok"})
        if claim:
            self.log.append(self.rid, GOAL_CLAIM, {"round": n, "kind": claim[0],
                                                   "summary": claim[1]})
        if end == "completed":
            self.model_call(content="done for now")
            self.log.append(self.rid, RUN_COMPLETED, {"answer": "done for now"})
        elif end is not None:
            self.log.append(self.rid, RUN_FAILED, {"error": end})
        return self

    def judge(self, n, answer, reason="", mode="check"):
        self.log.append(self.rid, GOAL_JUDGE, {"round": n, "mode": mode, "answer": answer,
                                               "reason": reason, "parsed": True})
        return self

    def verify(self, n, code, output=""):
        self.log.append(self.rid, GOAL_VERIFY, {"round": n, "command": "pytest -q",
                                                "exit_code": code, "output": output})
        return self

    def status(self, status, reason=""):
        self.log.append(self.rid, GOAL_STATUS, {"status": status, "reason": reason, "round": 0})
        return self


def work(i):
    """A tool call that differs from round to round (new work, so no stall)."""
    return [("read_file", json.dumps({"path": f"f{i}.py"}))]


# --- state and decisions -------------------------------------------------------


def test_no_goal_means_nothing_to_do():
    log = EventLog(":memory:")
    rid = log.create_run("task")
    assert goal_state(log.events(rid)) is None
    assert decide(log.events(rid)) == Decision("idle")


def test_a_new_goal_starts_round_one_with_the_kickoff():
    d = Run().decide()
    assert (d.action, d.round) == ("round", 1)
    assert "Goal: the tests pass" in d.message
    assert "`goal` tool" in d.message and '"complete"' in d.message and '"blocked"' in d.message
    assert "up to 150 rounds" in d.message


def test_a_running_round_waits():
    assert Run().round(1, end=None).decide() == Decision("wait", round=1)


def test_the_judge_checks_a_round_without_a_claim():
    run = Run().round(1, tools=work(1))
    assert run.decide() == Decision("judge", round=1, mode=g.CHECK)
    d = run.judge(1, "false").decide()
    assert (d.action, d.round) == ("round", 2)
    assert "round 2 of 150" in d.message and "the tests pass" in d.message
    assert Run().round(1).judge(1, "true").decide() == Decision(
        "stop", round=1, status=g.MET, reason="the judge answered true")


def test_a_round_that_ran_out_of_steps_is_still_checked():
    run = Run().round(1, tools=work(1), end="max_steps exceeded")
    assert goal_state(run.events()).rounds[0].end == g.OUT_OF_STEPS
    assert run.decide() == Decision("judge", round=1, mode=g.CHECK)


def test_a_complete_claim_without_a_verify_command_gets_a_verdict():
    def claimed():
        return Run().round(1, tools=work(1), claim=("complete", "all 40 tests pass"))

    assert claimed().decide() == Decision("judge", round=1, mode=g.VERDICT)
    d = claimed().judge(1, "met", "the test output shows 40 passed", g.VERDICT).decide()
    assert (d.action, d.status, d.reason) == ("stop", g.MET, "the test output shows 40 passed")
    d = claimed().judge(1, "impossible", "needs a GPU", g.VERDICT).decide()
    assert (d.status, d.reason) == (g.IMPOSSIBLE, "needs a GPU")
    d = claimed().judge(1, g.NOT_YET, "test_io still fails", g.VERDICT).decide()
    assert (d.action, d.round, d.reason) == ("round", 2, "test_io still fails")
    assert "Feedback from the check:\ntest_io still fails" in d.message


def test_the_verify_command_decides_when_there_is_one():
    def claimed():
        return Run(verify="pytest -q").round(1, tools=work(1), claim=("complete", "done"))

    assert claimed().decide() == Decision("verify", round=1)  # no judge for a claim
    d = claimed().verify(1, 0, "40 passed").decide()
    assert (d.action, d.status) == ("stop", g.MET)
    assert d.reason == "`pytest -q` exited 0:\n40 passed"
    d = claimed().verify(1, 1, "1 failed, 39 passed").decide()
    assert (d.action, d.round) == ("round", 2)
    assert "`pytest -q` exited 1:\n1 failed, 39 passed" in d.message
    d = claimed().verify(1, 127, "pytest: command not found").decide()
    assert (d.action, d.status) == ("stop", g.PAUSED)
    assert d.reason.startswith("the verify command is broken: `pytest -q` exited 127")

    # without a claim the judge goes first, and a "true" still has to pass the command
    run = Run(verify="pytest -q").round(1, tools=work(1))
    assert run.decide() == Decision("judge", round=1, mode=g.CHECK)
    assert run.judge(1, "true").decide() == Decision("verify", round=1)
    assert run.verify(1, 0).decide().status == g.MET
    d = Run(verify="pytest -q").round(1, tools=work(1)).judge(1, "false").decide()
    assert (d.action, d.round) == ("round", 2)  # a "false" doesn't run the command


def test_a_blocked_claim_stops_the_goal():
    d = Run().round(1, claim=("blocked", "I need the API key")).decide()
    assert (d.action, d.status, d.reason) == ("stop", g.BLOCKED, "I need the API key")


def test_an_interrupted_round_pauses_the_goal():
    d = Run().round(1, tools=work(1), end="interrupted").decide()
    assert (d.action, d.status) == ("stop", g.PAUSED)


def test_failed_rounds_are_retried_then_the_goal_pauses():
    run = Run().round(1, end="HTTP 500 from upstream")
    d = run.decide()
    assert (d.action, d.round) == ("round", 2)
    assert "stopped with an error: HTTP 500 from upstream" in d.message
    run.round(2, end="HTTP 500 from upstream").round(3, end="HTTP 500 from upstream")
    d = run.decide()
    assert (d.action, d.status) == ("stop", g.PAUSED)
    assert d.reason.startswith("3 rounds in a row failed")


def test_the_cap_stops_the_goal():
    run = Run(cap=2).round(1, tools=work(1)).judge(1, "false")
    assert run.decide().action == "round"
    d = run.round(2, tools=work(2)).judge(2, "false").decide()
    assert (d.action, d.status) == ("stop", g.CAPPED)
    assert "all 2 rounds were used" in d.reason


def test_rounds_without_new_work_stall():
    run = Run().round(1).judge(1, "false").round(2).judge(2, "false")
    assert run.decide().action == "round"  # two quiet rounds: keep going
    d = run.round(3).judge(3, "false").decide()
    assert (d.action, d.status) == ("stop", g.STALLED)

    # the same calls again count as no new work: round 1's calls are new, and
    # the three rounds that repeat them stall the goal
    same = [("grep", '{"pattern": "TODO"}')]
    run = Run()
    for n in (1, 2, 3):
        run.round(n, tools=same).judge(n, "false")
    assert run.decide().action == "round"
    run.round(4, tools=same).judge(4, "false")
    assert run.decide().status == g.STALLED
    run = Run()
    for n in (1, 2, 3, 4):
        run.round(n, tools=work(n)).judge(n, "false")
    assert run.decide().action == "round"  # new work every round
    # calling only the goal tool isn't work
    run = Run()
    for n in (1, 2, 3):
        run.round(n, tools=[("goal", '{"status": "complete"}')]).judge(n, "false")
    assert run.decide().status == g.STALLED


def test_resume_starts_a_new_stretch():
    run = Run().round(1, tools=work(1)).judge(1, "false").round(2, end="interrupted")
    run.status(g.PAUSED, "the round was interrupted")
    goal = goal_state(run.events())
    assert (goal.status, goal.round, goal.last_reason) == (
        g.PAUSED, 2, "the round was interrupted")
    assert run.decide() == Decision("idle")
    run.status(g.ACTIVE, "resumed")
    assert goal_state(run.events()).round == 0
    assert run.decide().round == 1


def test_clearing_removes_the_goal():
    run = Run().round(1, end=None).status(g.CLEARED)
    assert goal_state(run.events()) is None
    assert run.decide() == Decision("idle")


def test_a_turn_after_a_stop_is_not_a_goal_round():
    run = Run().round(1, tools=work(1), claim=("blocked", "need a key")).status(g.BLOCKED)
    run.log.append(run.rid, USER_MESSAGE, {"content": "what now?", "source": "user"})
    run.model_call(tool_calls=[("x", "grep", "{}")])
    run.log.append(run.rid, TOOL_CALL, {"tool_call_id": "x", "name": "grep",
                                        "arguments": "{}", "result": ""})
    run.log.append(run.rid, RUN_COMPLETED, {"answer": "ok"})
    goal = goal_state(run.events())
    assert [r.work for r in goal.rounds] == [['read_file:{"path": "f1.py"}']]


def test_a_new_goal_replaces_the_old_one():
    run = Run().round(1, tools=work(1)).judge(1, "false")
    run.log.append(run.rid, GOAL_SET, {"condition": "the docs build", "cap": 10})
    goal = goal_state(run.events())
    assert (goal.condition, goal.cap, goal.round, goal.verify) == ("the docs build", 10, 0, None)


def test_a_users_message_can_open_a_round():
    run = Run().round(1, tools=work(1)).judge(1, "false").round(2, source="user", end=None)
    assert goal_state(run.events()).rounds[-1].source == "user"


def test_goal_set_payload_fills_in_the_defaults():
    assert g.goal_set_payload("  the tests pass ") == {
        "condition": "the tests pass", "cap": 150, "verify": None, "verify_timeout": 600,
        "judge_model": None, "mode": "goal", "fields": {}}
    p = g.goal_set_payload("x", cap=5, verify=" pytest -q ", verify_timeout=1200.5,
                           judge_model="small", mode="guided", fields={"scope": 3})
    assert (p["cap"], p["verify"], p["verify_timeout"], p["judge_model"], p["mode"],
            p["fields"]) == (5, "pytest -q", 1200.5, "small", "guided", {"scope": "3"})
    assert g.goal_set_payload("x", verify="   ")["verify"] is None


def test_goal_set_payload_refuses_what_cant_make_a_goal():
    bad = [dict(condition=""), dict(condition="   "), dict(condition=None),
           dict(condition="x", cap=0), dict(condition="x", cap=True), dict(condition="x", cap="5"),
           dict(condition="x", verify=["pytest"]), dict(condition="x", verify_timeout=0),
           dict(condition="x", verify_timeout=True), dict(condition="x", judge_model=3),
           dict(condition="x", mode="loop"), dict(condition="x", fields=["scope"])]
    for kw in bad:
        condition = kw.pop("condition")
        try:
            g.goal_set_payload(condition, **kw)
        except ValueError:
            continue
        raise AssertionError(f"accepted {condition!r} {kw}")


def test_goal_state_reads_the_verify_timeout_and_judge_model():
    goal = goal_state(Run(verify_timeout=30, judge_model="small").events())
    assert (goal.verify_timeout, goal.judge_model) == (30, "small")
    goal = goal_state(Run().events())  # a GOAL_SET without them gets the defaults
    assert (goal.verify_timeout, goal.judge_model) == (600, None)


# --- the judge -----------------------------------------------------------------


def test_parse_check():
    assert g.parse_check("true") == ("true", True)
    assert g.parse_check(" False.") == ("false", True)
    assert g.parse_check("**true**") == ("true", True)
    assert g.parse_check("I think it's true") == ("false", False)
    assert g.parse_check("") == ("false", False)


def test_parse_verdict():
    assert g.parse_verdict("met - all green") == ("met", "all green", True)
    assert g.parse_verdict("Not yet met: README missing") == (g.NOT_YET, "README missing", True)
    assert g.parse_verdict("impossible — needs root") == ("impossible", "needs root", True)
    assert g.parse_verdict("maybe?") == (g.NOT_YET, "", False)


def test_the_judge_grammars_accept_only_the_answers():
    assert g.check_grammar().validate("true") and g.check_grammar().validate("false")
    assert not g.check_grammar().validate("maybe")
    v = g.verdict_grammar()
    assert v.validate("not yet met - the README isn't updated")
    assert v.validate("met - 40 passed") and v.validate("impossible - needs root")
    assert not v.validate("met")


def _judged_run():
    run = Run()
    run.log.append(run.rid, GOAL_ROUND, {"round": 1, "source": "goal"})
    # an early, long-gone call: a compacted context no longer carries its messages
    run.log.append(run.rid, MODEL_CALL, {"call_index": 0, "request_body": {"messages": [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "OLD MESSAGE"}]},
        "response": chat_response(content="old reply")})
    run.log.append(run.rid, MODEL_CALL, {"call_index": 1, "request_body": {"messages": [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "[summary of earlier work]"},
        {"role": "user", "content": "Goal: the tests pass"}]},
        "response": chat_response(tool_calls=[("c1", "bash", '{"cmd": "pytest -q"}')])})
    run.log.append(run.rid, TOOL_CALL, {"tool_call_id": "c1", "name": "bash",
                                        "arguments": '{"cmd": "pytest -q"}',
                                        "result": "40 passed"})
    run.log.append(run.rid, RUN_FAILED, {"error": "max_steps exceeded"})
    return run


def test_the_judge_sees_the_conversation_as_the_agent_last_saw_it():
    text = g.judge_transcript(_judged_run().events())
    assert "SYSTEM PROMPT" not in text   # the harness's system prompt is left out
    assert "OLD MESSAGE" not in text     # compacted away before the last call
    assert "[summary of earlier work]" in text
    assert '[called bash({"cmd": "pytest -q"})]' in text
    assert "[tool result: bash]\n40 passed" in text  # logged after the last call


def test_the_judge_request():
    run = _judged_run()
    goal = goal_state(run.events())
    body = g.judge_request(goal, 1, g.CHECK, run.events(), model="m", base_seed=5,
                           server_grammar="gbnf")
    assert body["model"] == "m" and "tools" not in body
    assert body["seed"] == 5 + g.JUDGE_SEED_OFFSET + 1 and body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": g.JUDGE_SYSTEM}
    user = body["messages"][1]["content"]
    assert user.startswith("Goal: the tests pass")
    assert user.endswith("Is the goal complete based on the work so far? "
                         "Answer with one word: true or false.")
    assert body["grammar"] == g.check_grammar().to_gbnf()
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 8
    # the same events always give the same request
    assert body == g.judge_request(goal, 1, g.CHECK, run.events(), model="m",
                                   base_seed=5, server_grammar="gbnf")

    guided = g.judge_request(goal, 1, g.VERDICT, run.events(), model="m",
                             server_grammar="guided", claim_summary="40 passed")
    assert guided["guided_grammar"] == g.verdict_grammar().to_gbnf()
    assert "The agent says the goal is complete: 40 passed" in guided["messages"][1]["content"]
    plain = g.judge_request(goal, 1, g.CHECK, run.events(), model="m", server_grammar=None)
    assert not ({"grammar", "guided_grammar", "max_tokens", "chat_template_kwargs"} & set(plain))


def test_judge_payload_parses_the_answer():
    body = {"seed": 7}
    p = g.judge_payload(1, g.CHECK, body, chat_response(content="true"), 12.0)
    assert (p["answer"], p["parsed"], p["seed"], p["round"]) == ("true", True, 7, 1)
    p = g.judge_payload(2, g.VERDICT, body, chat_response(content="met - 40 passed"), 12.0)
    assert (p["answer"], p["reason"], p["parsed"]) == ("met", "40 passed", True)
    p = g.judge_payload(3, g.CHECK, body, chat_response(content="hmm"), 12.0)
    assert (p["answer"], p["parsed"]) == ("false", False)  # unreadable counts as false


# --- the goal tool -------------------------------------------------------------


async def test_the_goal_tool_records_a_claim():
    run = Run()
    run.log.append(run.rid, GOAL_ROUND, {"round": 1, "source": "goal"})
    tools = ToolRegistry([g.goal_tool(run.log, run.rid, 1)])
    out = await tools.execute("goal", '{"status": "complete", "summary": "40 passed"}')
    assert out.startswith("Recorded")
    bad = await tools.execute("goal", '{"status": "done", "summary": ""}')
    assert bad.startswith("error")
    claims = [e.payload for e in run.log.events(run.rid, type=GOAL_CLAIM)]
    assert claims == [{"round": 1, "kind": "complete", "summary": "40 passed"}]
    run.log.append(run.rid, RUN_COMPLETED, {"answer": "done"})
    assert run.decide() == Decision("judge", round=1, mode=g.VERDICT)


# --- the goal round's final-step note, and replay ------------------------------


async def test_a_goal_round_gets_the_goal_final_step_note(tmp_path):
    log = EventLog(tmp_path / "e.db")
    mock = MockLlamaCpp(script={1: chat_response(content="still working")})
    client = OpenAICompatClient("http://t", "test-model", transport=mock.transport())
    agent = Agent(client, ToolRegistry(builtin_tools()), log, capabilities=CAPS,
                  base_seed=1, max_steps=1)
    agent.final_step_note = g.GOAL_FINAL_STEP_NOTE
    await agent.run("work toward the goal")
    sent = mock.chat_bodies[-1]["messages"]
    assert sent[-1] == {"role": "user", "content": g.GOAL_FINAL_STEP_NOTE}


async def test_replay_covers_the_judges_calls(tmp_path):
    judge_seed = g.judge_seed(1, 1)
    script = {1: chat_response(content="I ran the tests: 40 passed."),
              judge_seed: chat_response(content="true")}
    log = EventLog(tmp_path / "e.db")
    client = OpenAICompatClient("http://t", "test-model",
                                transport=MockLlamaCpp(script=script).transport())
    rid = log.create_run("task")
    log.append(rid, GOAL_SET, {"condition": "the tests pass"})
    log.append(rid, GOAL_ROUND, {"round": 1, "source": "goal"})
    agent = Agent(client, ToolRegistry([]), log, capabilities=CAPS, base_seed=1)
    await agent.run("task", run_id=rid)

    goal = goal_state(log.events(rid))
    body = g.judge_request(goal, 1, g.CHECK, log.events(rid), model="test-model")
    response = await client.chat_body(body)
    log.append(rid, GOAL_JUDGE, g.judge_payload(1, g.CHECK, body, response.raw,
                                                response.timing_ms))
    assert decide(log.events(rid)).status == g.MET

    report = await replay_run(log, rid, client)
    assert (report.total, report.matched) == (2, 2)  # the agent's call and the judge's

    # a judge that now answers differently shows up as a mismatch
    other = {**script, judge_seed: chat_response(content="false")}
    changed = OpenAICompatClient("http://t", "test-model",
                                 transport=MockLlamaCpp(script=other).transport())
    report = await replay_run(log, rid, changed)
    assert not report.identical
    assert report.mismatches[0].seq == log.events(rid, type=GOAL_JUDGE)[0].seq


# --- gaps found by mutation testing (mutmut 3.8.1 on agent/goal.py, 23 Nov 2026) ---
# Each test below kills mutants that survived the suite above; see
# docs/software-testing/LO3-evidence/mutation-goal.md.


def test_a_failed_round_at_the_cap_stops_the_goal():
    # survived: `r.number >= goal.cap` -> `>` in the failed-round branch, which let
    # a goal run past its cap when the last round failed
    d = Run(cap=2).round(1, tools=work(1)).judge(1, "false").round(2, end="HTTP 500").decide()
    assert (d.action, d.status, d.round, d.reason) == ("stop", g.CAPPED, 2,
                                                       "all 2 rounds were used")


def test_the_retry_prompt_names_the_next_round():
    d = Run(cap=5).round(1, end="HTTP 500 from upstream").decide()
    assert "This is round 2 of 5." in d.message


def test_a_true_from_the_judge_still_needs_the_verify_command_to_pass():
    d = (Run(verify="pytest -q").round(1, tools=work(1)).judge(1, "true")
         .verify(1, 1, "1 failed").decide())
    assert (d.action, d.round) == ("round", 2)
    assert "`pytest -q` exited 1:\n1 failed" in d.message


def test_stalling_compares_each_round_with_the_one_before():
    # survived: comparing with the round two back, which would stall A, B, A, B
    a, b = [("grep", '{"pattern": "A"}')], [("grep", '{"pattern": "B"}')]
    run = Run()
    for n, calls in enumerate((a, b, a, b, a), start=1):
        run.round(n, tools=calls).judge(n, "false")
    assert run.decide().action == "round"


def test_a_chat_turn_between_pause_and_resume_doesnt_hide_the_resume():
    # survived: `continue` -> `break` in goal_state, which stopped reading the log
    # at the first event outside a round
    run = Run().round(1, tools=work(1)).judge(1, "false").round(2, end="interrupted")
    run.status(g.PAUSED, "the round was interrupted")
    run.log.append(run.rid, USER_MESSAGE, {"content": "quick question", "source": "user"})
    run.model_call(content="an answer")
    run.log.append(run.rid, RUN_COMPLETED, {"answer": "an answer"})
    run.status(g.ACTIVE, "resumed by the user")
    assert goal_state(run.events()).status == g.ACTIVE
    d = run.decide()
    assert (d.action, d.round) == ("round", 1)


def test_the_stop_reasons_the_user_sees():
    assert Run().round(1, tools=work(1), end="interrupted").decide().reason == \
        "the round was interrupted"
    assert Run().round(1, claim=("blocked", "")).decide().reason == \
        "the agent reported it is blocked"

    def verdict(answer):
        return (Run().round(1, tools=work(1), claim=("complete", "done"))
                .judge(1, answer, mode=g.VERDICT).decide())

    assert verdict(g.MET).reason == "the judge agreed the goal is met"
    assert verdict(g.IMPOSSIBLE).reason == "the judge found the goal impossible"
    stalled = Run()
    for n in (1, 2, 3):
        stalled.round(n).judge(n, "false")
    assert stalled.decide().reason == "3 rounds in a row made no new tool calls"
    # a plain "false" gives the next round no feedback section
    d = Run().round(1, tools=work(1)).judge(1, "false").decide()
    assert "Feedback from the check" not in d.message


def test_goal_set_payload_accepts_a_cap_of_one_and_explains_refusals():
    assert g.goal_set_payload("x", cap=1)["cap"] == 1  # the boundary is allowed
    cases = {
        "a goal needs a condition": dict(condition=""),
        "the round cap must be a whole number, at least 1": dict(condition="x", cap=0),
        "the verify command must be a string": dict(condition="x", verify=["pytest"]),
        "the verify timeout must be a positive number of seconds":
            dict(condition="x", verify_timeout=0),
        "the judge model must be a string": dict(condition="x", judge_model=3),
        "the goal mode must be one of goal, guided": dict(condition="x", mode="loop"),
        "the goal's fields must be an object of names to text":
            dict(condition="x", fields=["scope"]),
    }
    for message, kw in cases.items():
        condition = kw.pop("condition")
        try:
            g.goal_set_payload(condition, **kw)
        except ValueError as e:
            assert str(e) == message
        else:
            raise AssertionError(f"accepted {condition!r} {kw}")


def test_a_goal_set_without_optional_fields_gets_the_defaults():
    log = EventLog(":memory:")
    rid = log.create_run("task")
    log.append(rid, GOAL_SET, {"condition": "the docs build"})
    goal = goal_state(log.events(rid))
    assert (goal.mode, goal.fields, goal.cap, goal.verify) == ("goal", {}, g.DEFAULT_CAP, None)


async def test_the_goal_tools_reply_says_what_happens_next():
    run = Run()
    run.log.append(run.rid, GOAL_ROUND, {"round": 1, "source": "goal"})
    tool = g.goal_tool(run.log, run.rid, 1)
    tools = ToolRegistry([tool])
    done = await tools.execute("goal", '{"status": "complete", "summary": "40 passed"}')
    stuck = await tools.execute("goal", '{"status": "blocked", "summary": "no key"}')
    assert done.startswith("Recorded: the goal will be checked when this reply ends.")
    assert stuck.startswith("Recorded: the goal will stop and the user will see your reason.")
    # a claim with no summary is logged with an empty one (decide() supplies the reason)
    await tools.execute("goal", '{"status": "blocked"}')
    claims = [e.payload for e in run.log.events(run.rid, type=GOAL_CLAIM)]
    assert claims[-1] == {"round": 1, "kind": "blocked", "summary": ""}
    # the schema the model sees
    assert tool.parameters["type"] == "object"
    props = tool.parameters["properties"]
    assert props["status"] == {"type": "string", "enum": ["complete", "blocked"]}
    assert props["summary"]["type"] == "string" and props["summary"]["description"]
    assert tool.parameters["required"] == ["status", "summary"]
    assert "complete" in tool.description and "blocked" in tool.description


def test_a_guided_goals_mode_and_fields_are_read_back_from_the_log():
    run = Run(mode="guided", fields={"scope": "docs/", "done when": "the build passes"})
    goal = goal_state(run.events())
    assert (goal.mode, goal.fields) == ("guided", {"scope": "docs/",
                                                   "done when": "the build passes"})


def test_judge_payload_keeps_the_timing_and_no_reason_for_a_check():
    p = g.judge_payload(1, g.CHECK, {"seed": 7}, chat_response(content="false"), 12.5)
    assert (p["timing_ms"], p["reason"]) == (12.5, "")


def _log_with(*events):
    """A run holding exactly these (type, payload) events after its GOAL_SET."""
    log = EventLog(":memory:")
    rid = log.create_run("task")
    log.append(rid, GOAL_SET, {"condition": "the tests pass"})
    for t, p in events:
        log.append(rid, t, p)
    return log.events(rid)


def _model_call(messages, **reply):
    return MODEL_CALL, {"request_body": {"messages": messages}, "response": chat_response(**reply)}


def test_the_judge_request_carries_the_transcript():
    # survived: the transcript replaced by None, so the judge saw no conversation
    run = _judged_run()
    goal = goal_state(run.events())
    for mode in (g.CHECK, g.VERDICT):
        user = g.judge_request(goal, 1, mode, run.events(), model="m")["messages"][1]["content"]
        assert f"The conversation so far:\n\n{g.judge_transcript(run.events())}\n\n" in user
        assert "The agent says" not in user  # no claim, no claim line


def test_the_judge_reads_the_latest_model_call_wherever_it_sits():
    # survived: searching back two events at a time, which can skip the latest call
    calls = [_model_call([{"role": "user", "content": "FIRST"}], content="reply 1"),
             _model_call([{"role": "user", "content": "SECOND"}], content="reply 2")]
    for tail in ([], [(RUN_COMPLETED, {"answer": "x"})]):  # either parity
        text = g.judge_transcript(_log_with(*calls, *tail))
        assert text == "[user]\nSECOND\n\n[assistant]\nreply 2"


def test_tool_results_from_before_the_last_call_appear_once():
    call = {"id": "c1", "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
    events = _log_with(
        _model_call([{"role": "user", "content": "go"}],
                    tool_calls=[("c1", "bash", '{"cmd": "ls"}')]),
        (TOOL_CALL, {"name": "bash", "arguments": '{"cmd": "ls"}', "result": "EARLIER"}),
        _model_call([{"role": "user", "content": "go"},
                     {"role": "assistant", "content": None, "tool_calls": [call]},
                     {"role": "tool", "name": "bash", "content": "EARLIER"}], content="done"))
    assert g.judge_transcript(events) == (
        '[user]\ngo\n\n[assistant]\n[called bash({"cmd": "ls"})]\n\n'
        "[tool result: bash]\nEARLIER\n\n[assistant]\ndone")


def test_the_judge_transcript_cuts_long_tool_results_and_then_old_text():
    exact, long = "x" * 2000, "y" * 2001  # the default cut is 2000 characters
    events = _log_with(
        _model_call([{"role": "user", "content": "go"}],
                    tool_calls=[("c1", "bash", "{}")]),
        (TOOL_CALL, {"name": "bash", "arguments": "{}", "result": exact}),
        (TOOL_CALL, {"name": "bash", "arguments": "{}", "result": long}))
    text = g.judge_transcript(events)
    assert f"[tool result: bash]\n{exact}\n\n" in text  # at the limit: kept whole
    assert text.endswith(f"[tool result: bash]\n{'y' * 2000} [cut]")
    # the whole transcript, when too long, keeps its end
    assert g.judge_transcript(events, max_chars=len(text)) == text
    short = g.judge_transcript(events, max_chars=100)
    assert short == "[earlier conversation left out]\n\n" + text[-100:]


def test_the_judge_reads_messages_whose_content_is_a_list_of_parts():
    parts = [{"type": "text", "text": "PART-ONE"},
             {"type": "image_url", "image_url": {"url": "data:,"}},
             {"type": "text", "text": "PART-TWO"}]
    text = g.judge_transcript(_log_with(_model_call([{"role": "user", "content": parts}],
                                                    content="seen")))
    assert text == "[user]\nPART-ONE  PART-TWO\n\n[assistant]\nseen"


def test_the_judge_request_is_a_system_then_a_user_message():
    goal = goal_state(_judged_run().events())
    body = g.judge_request(goal, 1, g.CHECK, _judged_run().events(), model="m")
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_verify_feedback_keeps_the_last_20_lines_and_2000_characters():
    lines = "\n".join(f"line {i}" for i in range(1, 26))
    d = (Run(verify="pytest -q").round(1, tools=work(1), claim=("complete", "done"))
         .verify(1, 1, lines).decide())
    assert "line 6\n" in d.message and "line 5\n" not in d.message
    wide = "z" * 2500
    d = (Run(verify="pytest -q").round(1, tools=work(1), claim=("complete", "done"))
         .verify(1, 1, wide).decide())
    assert f"`pytest -q` exited 1:\n{'z' * 2000}\n" in d.message
    assert "z" * 2001 not in d.message


def test_a_guided_goals_fields_reach_the_prompts():
    goal = goal_state(Run(mode="guided", fields={"scope": "docs/"}).events())
    assert "\nscope: docs/\n" in g.kickoff_prompt(goal)
    body = g.judge_request(goal, 1, g.CHECK, _judged_run().events(), model="m")
    assert body["messages"][1]["content"].startswith("Goal: the tests pass\n\nscope: docs/\n\n")
    plain = goal_state(Run().events())
    assert "scope:" not in g.kickoff_prompt(plain)
