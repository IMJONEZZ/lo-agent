"""The goal loop: runs a goal (agent/goal.py) round by round on the session server.

agent/goal.py decides what happens next as a pure function of the run's events;
this module carries each decision out. It starts a round as an agent turn, asks
the judge, or runs the verify command, logs what that produced, and asks decide()
again, until the goal stops. SessionManager runs one loop per goal as the run's
task, so the session counts as running for the whole goal (through the judge's
calls and the verify command too), and an interrupt stops the goal rather than
just the round it lands in.
"""

from __future__ import annotations

import os

from ..agent.goal import (
    GOAL_FINAL_STEP_NOTE,
    GOAL_TOOL_NAME,
    PAUSED,
    Decision,
    Goal,
    decide,
    goal_state,
    goal_tool,
    judge_payload,
    judge_request,
)
from ..events.log import GOAL_JUDGE, GOAL_ROUND, GOAL_STATUS, GOAL_VERIFY, RUN_FAILED
from ..sandbox import HostSandbox

VERIFY_OUTPUT_CHARS = 8000  # how much of the verify command's output GOAL_VERIFY keeps (the end)


def _error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def log_status(log, run_id: str, status: str, reason: str, round_number: int) -> None:
    log.append(run_id, GOAL_STATUS, {"status": status, "reason": reason, "round": round_number})


def equip_round_agent(agent, log, run_id: str, round_number: int):
    """Give a round's agent the goal tool and the goal's final-step note. The tool
    is allowed without asking, and shown even under a preset that limits the
    toolset (plan, explore), since reporting on the goal reads and changes nothing."""
    agent.tools.register(goal_tool(log, run_id, round_number))
    perms = getattr(agent.tools, "permissions", None)
    if perms is not None and GOAL_TOOL_NAME not in perms.allow:
        perms.allow.append(GOAL_TOOL_NAME)
    if agent.exposed_tools is not None:
        agent.exposed_tools = set(agent.exposed_tools) | {GOAL_TOOL_NAME}
    agent.final_step_note = GOAL_FINAL_STEP_NOTE
    return agent


async def drive_goal(manager, run_id: str, *, preset: str | None = None,
                     code_mode: bool | None = None, task_opens_round_1: bool = False) -> None:
    """Run the run's active goal until it stops.

    `task_opens_round_1`: the run was created for this goal with the kickoff
    prompt as its task, so round 1 is agent.run(). A continue_run would send the
    kickoff a second time, as a second user message in a row, which some chat
    templates reject."""
    log = manager.bus.log
    agent = None  # the latest round's agent: the judge uses its client and model
    while True:
        events = log.events(run_id)
        d = decide(events)
        if d.action == "idle":
            return
        if d.action == "stop":
            log_status(log, run_id, d.status, d.reason, d.round)
            return
        if d.action == "wait":
            # Only this loop starts rounds, and it waits for each one to end, so a
            # round without an end event is a bug. Stop instead of waiting forever.
            log_status(log, run_id, PAUSED,
                       f"round {d.round} ended without an end event", d.round)
            return
        goal = goal_state(events)
        if d.action == "round":
            agent = await _round(manager, run_id, d, preset, code_mode,
                                 opens_with_task=task_opens_round_1 and d.round == 1)
            task_opens_round_1 = False
            if agent is None:
                return
        elif d.action == "judge":
            if not await _judge(manager, run_id, goal, d, events, agent, preset, code_mode):
                return
        elif d.action == "verify":
            if not await _verify(manager, run_id, goal, d, agent):
                return


async def _round(manager, run_id: str, d: Decision, preset, code_mode, *, opens_with_task: bool):
    """Run one round's turn. Returns the round's agent, or None if the goal stopped
    because the agent couldn't be built."""
    log = manager.bus.log
    try:  # before GOAL_ROUND, so a round that can't start is never opened
        agent = manager._agent_for(run_id, preset, code_mode)
    except Exception as e:  # noqa: BLE001 — e.g. the upstream is down
        if manager._needs_terminal(run_id):  # a run created for this goal: end its turn
            log.append(run_id, RUN_FAILED, {"error": _error(e)})
        log_status(log, run_id, PAUSED, f"round {d.round} couldn't start: {_error(e)}", d.round)
        return None
    equip_round_agent(agent, log, run_id, d.round)
    # Messages the user sent during the goal open the next round in place of the
    # goal's prompt. Round 1 of a run created for the goal is its task, so they wait.
    queued = "" if opens_with_task else manager.take_queued(run_id)
    source = "user" if queued else "goal"
    log.append(run_id, GOAL_ROUND, {"round": d.round, "source": source})
    error = "the round ended without an end event"
    try:
        if opens_with_task:
            await agent.run(log.run(run_id).task, run_id=run_id)
        else:
            await agent.continue_run(run_id, queued or d.message, source=source)
    except Exception as e:  # noqa: BLE001 — _loop logged RUN_FAILED; decide() retries or pauses
        error = _error(e)
    if manager._needs_terminal(run_id):  # failed before _loop could log the turn's end
        log.append(run_id, RUN_FAILED, {"error": error})
    return agent


async def _judge(manager, run_id: str, goal: Goal, d: Decision, events, agent,
                 preset, code_mode) -> bool:
    """Ask the judge about round d.round and log its answer. On failure the goal
    pauses: decide() has no state for a judge that failed, so going on would
    only ask again."""
    log = manager.bus.log
    try:
        if agent is None:
            agent = manager._agent_for(run_id, preset, code_mode)
        claim = (goal.rounds[-1].claim or {}).get("summary") or ""
        body = judge_request(goal, d.round, d.mode, events,
                             model=goal.judge_model or agent.model or agent.client.model,
                             base_seed=agent.base_seed, server_grammar=agent.caps.grammar,
                             claim_summary=claim)
        # Straight to the client, not through the agent: the judge's tokens
        # mustn't stream into the chat, and it gets no tools.
        response = await agent.client.chat_body(body)
        payload = judge_payload(d.round, d.mode, body, response.raw, response.timing_ms)
    except Exception as e:  # noqa: BLE001
        log_status(log, run_id, PAUSED, f"the judge's model call failed: {_error(e)}", d.round)
        return False
    log.append(run_id, GOAL_JUDGE, payload)
    return True


async def _verify(manager, run_id: str, goal: Goal, d: Decision, agent) -> bool:
    """Run the goal's verify command where the agent's own commands run, and log
    its exit code and the end of its output."""
    log = manager.bus.log
    sandbox = getattr(agent, "sandbox", None) or HostSandbox(os.getcwd())
    try:
        output, code = await sandbox.exec(goal.verify, timeout=goal.verify_timeout)
    except Exception as e:  # noqa: BLE001
        log_status(log, run_id, PAUSED, f"the verify command couldn't run: {_error(e)}", d.round)
        return False
    log.append(run_id, GOAL_VERIFY, {"round": d.round, "command": goal.verify,
                                     "exit_code": code, "output": output[-VERIFY_OUTPUT_CHARS:]})
    return True
