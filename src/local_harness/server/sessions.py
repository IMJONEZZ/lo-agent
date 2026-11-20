"""Session manager: owns agent runs, drives them onto the event bus.

One agent run = one session. `start` creates the run, builds a fresh Agent whose
stream callbacks (token/reasoning/tool/notice) publish EPHEMERAL events to the
bus, and drives it in a background task; the Agent's own persisted events flow to
the bus automatically because it shares the bus's EventLog (see events/bus.py).
`stream` is just `bus.subscribe`. No HTTP here — that's server/app.py.

Goal mode: `start_goal` and `goal` set, pause, resume and clear a run's goal.
A goal runs as one task for the whole goal (server/goals.py drives its rounds),
so the session counts as running until the goal stops, an interrupt stops the
goal, and a message sent meanwhile waits for the goal's next round.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator, Callable

from ..agent.goal import (
    ACTIVE,
    CLEARED,
    MET,
    PAUSED,
    Goal,
    goal_set_payload,
    goal_state,
    kickoff_prompt,
)
from ..agent.loop import Agent
from ..events.bus import (
    EventBus,
    NOTICE,
    PERMISSION_REQUEST,
    REASONING_DELTA,
    TERMINAL,
    TOKEN_DELTA,
    TOOL_PROGRESS,
)
from ..events.log import AGENT_SPAWNED, GOAL_SET, Event, RUN_COMPLETED, RUN_FAILED
from ..inference.types import Message
from ..tree.search.plan_search import plan_search
from .goals import drive_goal, log_status

MAX_COORDINATOR_DEPTH = 1  # only the top-level lead may spawn workers (no recursion)

# An agent factory takes the three run-scoped stream callbacks plus an optional
# preset name, and returns an Agent bound to the bus's EventLog. The preset lets a
# client pick the per-session agent profile (build/plan/explore/general) — system
# prompt, sampling, and exposed toolset — so plan/explore mode is enforced
# server-side, not silently ignored. The CLI builds this from parsed args; tests
# pass a trivial one.
AgentFactory = Callable[..., Agent]


class GoalConflict(Exception):
    """A goal action that doesn't fit the session's state (HTTP 409), such as
    setting a goal while a turn runs, or pausing a goal that isn't active."""


class SessionManager:
    def __init__(
        self,
        bus: EventBus,
        agent_factory: AgentFactory,
        interactive_permissions: bool = False,
        permission_timeout: float = 120.0,
    ):
        self.bus = bus
        self.agent_factory = agent_factory
        # When True, an "ask"-tier tool pauses the run and asks the connected
        # client(s) over the bus (PERMISSION_REQUEST) instead of auto-approving.
        # The TUI's embedded server sets this (unless --allow-all); standalone
        # `lo serve` defaults False (headless) unless --approval prompt.
        self.interactive_permissions = interactive_permissions
        self.permission_timeout = permission_timeout
        self._pending_perms: dict[str, asyncio.Future] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._finalizers: set[asyncio.Task] = set()  # keep interrupt finalizers alive
        # Coordinator state: who spawned whom, how deep, and per-run peer inboxes.
        self._depth: dict[str, int] = {}
        self._inbox: dict[str, list[str]] = {}
        # Goal mode: runs whose task is a goal loop; messages sent during a goal
        # (they open its next round); and the (status, reason) that the next cancel
        # of a goal loop stops the goal with (set by pause and clear).
        self._goal_runs: set[str] = set()
        self._queued: dict[str, list[str]] = {}
        self._goal_stop: dict[str, tuple[str, str]] = {}
        self._pause_orphaned_goals()

    # --- callbacks -------------------------------------------------------

    def _callbacks(self, run_id: str):
        def on_token(kind: str, text: str) -> None:
            if not text:
                return  # "start" sentinel carries no text
            t = REASONING_DELTA if kind == "reasoning" else TOKEN_DELTA
            self.bus.publish_delta(run_id, t, {"text": text})

        def on_tool(name: str, phase: str) -> None:
            self.bus.publish_delta(
                run_id, TOOL_PROGRESS, {"name": name, "phase": phase}
            )

        def on_notice(msg: str) -> None:
            self.bus.publish_delta(run_id, NOTICE, {"message": msg})

        return on_token, on_tool, on_notice

    def _agent_for(
        self, run_id: str, preset: str | None = None, code_mode: bool | None = None
    ) -> Agent:
        agent = self.agent_factory(*self._callbacks(run_id), preset=preset)
        if code_mode is not None:  # per-request override of the factory default
            agent.code_mode = code_mode
        # Route the "ask" tier through the bus when interactive: replace the
        # factory's baked-in auto-approver with one that asks the client. Only
        # touches presets that actually have an approver (an ask tier); "deny"
        # still denies before the approver is ever consulted.
        if self.interactive_permissions:
            perms = getattr(getattr(agent, "tools", None), "permissions", None)
            if perms is not None and getattr(perms, "approver", None) is not None:
                perms.approver = self._make_approver(run_id)
        # Coordinator: peer inbox on every agent; send_message everywhere; spawn_agents
        # only at the top level (depth 0) so workers can't recursively fan out.
        agent.inbox = lambda: self._drain_inbox(run_id)
        tools = getattr(agent, "tools", None)
        if tools is not None:
            from .coordinator import send_message_tool, spawn_agents_tool

            tools.register(send_message_tool(self, run_id))
            if self._depth.get(run_id, 0) < MAX_COORDINATOR_DEPTH:
                tools.register(spawn_agents_tool(self, run_id))
        return agent

    # --- coordinator: fan-out + peer messaging ---------------------------

    def spawn_child(self, task: str, preset: str | None, parent_run_id: str) -> str:
        """Start a worker run for `task`, tagged as a child of `parent_run_id`."""
        run_id = self.bus.create_run(task)
        self._depth[run_id] = self._depth.get(parent_run_id, 0) + 1
        self.bus.log.append(
            parent_run_id, AGENT_SPAWNED, {"child_run_id": run_id, "task": task}
        )
        self._spawn(
            run_id,
            self._drive(run_id, lambda a: a.run(task, run_id=run_id), preset=preset),
        )
        return run_id

    async def await_children(self, run_ids: list[str]) -> list[tuple[str, str]]:
        """Wait for each child run to terminate; return (run_id, answer)."""

        async def _one(rid: str) -> tuple[str, str]:
            try:
                async for ev in self.bus.subscribe(rid, replay=True, stop_on=TERMINAL):
                    if ev.type in TERMINAL:
                        break
            except Exception:  # noqa: BLE001 — gather what we can
                pass
            done = self.bus.log.events(rid, type=RUN_COMPLETED)
            if done:
                return rid, (done[-1].payload.get("answer") or "(no answer)")
            failed = self.bus.log.events(rid, type=RUN_FAILED)
            reason = failed[-1].payload.get("error") if failed else "unknown"
            return rid, f"(worker failed: {reason})"

        return list(await asyncio.gather(*(_one(r) for r in run_ids)))

    def _resolve_run(self, ref: str) -> str | None:
        ref = (ref or "").strip()
        if self.bus.log.run(ref) is not None:
            return ref
        for r in self.bus.log.runs():  # accept a short id prefix
            if r.run_id.startswith(ref):
                return r.run_id
        return None

    def deliver_message(self, to_agent: str, from_run_id: str, content: str) -> bool:
        rid = self._resolve_run(to_agent)
        if rid is None or not self.is_running(rid):
            return False
        self._inbox.setdefault(rid, []).append(
            f"[message from agent {from_run_id[:8]}] {content}"
        )
        return True

    def _drain_inbox(self, run_id: str) -> list[str]:
        return self._inbox.pop(run_id, [])

    def _make_approver(self, run_id: str):
        async def approve(tool: str, arguments: str) -> bool:
            return await self.request_permission(run_id, tool, arguments)

        return approve

    async def _await_subscriber(self, run_id: str, timeout: float) -> bool:
        """Wait (up to `timeout`) for at least one live bus subscriber. PERMISSION_
        REQUEST is ephemeral, so a subscriber must be attached at publish time or the
        ask is lost — but a subscriber that raced the run start (or dropped and is
        reconnecting) shows up within millimeters of a second, so we poll briefly
        rather than deny outright."""
        if self.bus.subscriber_count(run_id) > 0:
            return True
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            await asyncio.sleep(0.05)
            if self.bus.subscriber_count(run_id) > 0:
                return True
        return False

    async def request_permission(self, run_id: str, tool: str, arguments: str) -> bool:
        """Ask the connected client(s) to approve a tool call. Publishes a
        PERMISSION_REQUEST over the bus and awaits the client's POST back. Denies
        if the client doesn't answer in time — fail-safe.

        On zero subscribers we distinguish two cases: a headless manager (nobody
        will ever answer) denies immediately, but an interactive one waits for a
        subscriber to (re)attach — otherwise a momentary SSE gap at the instant an
        ask-tier tool fires would auto-deny every such call for the rest of the run,
        even with a human actively watching (the code-mode denial cascade)."""
        if self.bus.subscriber_count(run_id) == 0:
            if not self.interactive_permissions:
                return False  # headless: no one to ask → deny (don't hang the run)
            if not await self._await_subscriber(run_id, self.permission_timeout):
                return False  # interactive, but no client attached within the window
        request_id = uuid.uuid4().hex[:12]
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_perms[request_id] = fut
        self.bus.publish_delta(
            run_id,
            PERMISSION_REQUEST,
            {"request_id": request_id, "tool": tool, "arguments": arguments},
        )
        try:
            return bool(await asyncio.wait_for(fut, self.permission_timeout))
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending_perms.pop(request_id, None)

    def resolve_permission(self, request_id: str, approved: bool) -> bool:
        """Client's answer to a PERMISSION_REQUEST. Returns False if the request is
        unknown/already settled (stale double-tap)."""
        fut = self._pending_perms.get(request_id)
        if fut is not None and not fut.done():
            fut.set_result(bool(approved))
            return True
        return False

    # --- lifecycle -------------------------------------------------------

    def start(
        self, task: str, preset: str | None = None, code_mode: bool | None = None
    ) -> str:
        """Create a run and drive an agent against it; returns the run_id."""
        run_id = self.bus.create_run(task)
        self._spawn(
            run_id,
            self._drive(
                run_id,
                lambda a: a.run(task, run_id=run_id),
                preset=preset,
                code_mode=code_mode,
            ),
        )
        return run_id

    def send(
        self,
        run_id: str,
        message: str,
        preset: str | None = None,
        code_mode: bool | None = None,
    ) -> str:
        """Continue an existing run with a new user turn. While a goal runs, the
        message waits instead, and opens the goal's next round."""
        if self.bus.log.run(run_id) is None:
            raise KeyError(run_id)
        if self.goal_running(run_id):
            self._queued.setdefault(run_id, []).append(message)
            return run_id
        self._spawn(
            run_id,
            self._drive(
                run_id,
                lambda a: a.continue_run(run_id, message),
                preset=preset,
                code_mode=code_mode,
            ),
        )
        return run_id

    async def plan(self, task: str, n: int = 4) -> list[dict]:
        """Fan-out planning over HTTP: fork N candidate plans for `task` and
        return them best-first. The client (e.g. the TUI in --server mode) has no
        model access of its own, so the server — which owns the live client+caps —
        runs plan_search on its behalf. We borrow the agent factory's bound client
        and capabilities (it closes over them) rather than holding a second copy."""
        agent = self._agent_for(
            "__plan__", preset=None
        )  # built only to reach its client/caps
        messages = [Message(role="user", content=task)]
        candidates = await plan_search(agent.client, agent.caps, messages, n=n)
        return [{"text": c.text, "score": c.score} for c in candidates]

    def interrupt(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            # A task cancelled BEFORE it starts gets CancelledError thrown at
            # coroutine entry, bypassing _drive's except — so _drive can't always
            # log the terminal. A finalizer awaits the settled task and guarantees
            # exactly one terminal event regardless of when the cancel landed.
            self._finalizers.add(
                asyncio.ensure_future(self._ensure_terminal(run_id, task))
            )
            return True
        return False

    async def _ensure_terminal(self, run_id: str, task: asyncio.Task) -> None:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        if self._needs_terminal(run_id):
            self.bus.log.append(run_id, RUN_FAILED, {"error": "interrupted"})
        self._settle_goal(run_id)
        self._finalizers.discard(asyncio.current_task())

    def _needs_terminal(self, run_id: str) -> bool:
        """Whether the run's current turn still lacks its end event. The run's
        status is "running" from create_run, or from continue_run's reopen, until
        that turn logs RUN_COMPLETED or RUN_FAILED, so an earlier turn's end event
        doesn't count. A deleted run needs nothing."""
        meta = self.bus.log.run(run_id)
        return meta is not None and meta.status == "running"

    def is_running(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def last_seq(self, run_id: str) -> int:
        """The run's latest event number (-1 for none): a client that wants only
        what happens next follows the events after it."""
        return self.bus.log.event_count(run_id) - 1

    # --- goal mode -------------------------------------------------------

    def start_goal(self, condition, *, preset: str | None = None,
                   code_mode: bool | None = None, **options) -> str:
        """Start a new session for a goal; returns its run_id. The run's task is
        the goal's kickoff prompt, so round 1 is the run's first turn, and the
        session is named after the goal. `options` are goal_set_payload's."""
        payload = goal_set_payload(condition, **options)
        run_id = self.bus.create_run(kickoff_prompt(Goal.from_set(payload)))
        self.bus.log.rename_run(run_id, f"goal: {payload['condition']}")
        self.bus.log.append(run_id, GOAL_SET, payload)
        self._spawn_goal(run_id, preset, code_mode, task_opens_round_1=True)
        return run_id

    async def goal(self, run_id: str, action: str = "set", *, condition=None,
                   preset: str | None = None, code_mode: bool | None = None,
                   **options) -> dict | None:
        """Set, pause, resume or clear the run's goal; returns goal_summary().
        Raises KeyError for an unknown run, ValueError for a bad request, and
        GoalConflict when the action doesn't fit the session's state."""
        log = self.bus.log
        if log.run(run_id) is None:
            raise KeyError(run_id)
        current = goal_state(log.events(run_id))
        if action == "set":
            payload = goal_set_payload(condition, **options)
            self._check_idle(run_id)
            log.append(run_id, GOAL_SET, payload)
            self._spawn_goal(run_id, preset, code_mode)
        elif action == "pause":
            if current is None or not current.active:
                raise GoalConflict("there's no active goal to pause")
            await self._stop_goal(run_id, current, PAUSED, "paused by the user")
        elif action == "resume":
            if current is None:
                raise GoalConflict("there's no goal to resume")
            if current.status == MET:
                raise GoalConflict("the goal is already met: set a new goal instead")
            self._check_idle(run_id)
            log_status(log, run_id, ACTIVE, "resumed by the user", current.round)
            self._spawn_goal(run_id, preset, code_mode)
        elif action == "clear":
            if current is None:
                raise GoalConflict("there's no goal to clear")
            await self._stop_goal(run_id, current, CLEARED, "cleared by the user")
        else:
            raise ValueError(f"unknown goal action {action!r}: use set, pause, resume or clear")
        return self.goal_summary(run_id)

    def goal_summary(self, run_id: str) -> dict | None:
        """The run's goal as a client shows it; None when the run has no goal."""
        goal = goal_state(self.bus.log.events(run_id))
        if goal is None:
            return None
        return {"condition": goal.condition, "status": goal.status, "round": goal.round,
                "cap": goal.cap, "verify": goal.verify, "mode": goal.mode,
                "reason": goal.last_reason, "running": self.goal_running(run_id)}

    def take_queued(self, run_id: str) -> str:
        """The messages sent during the goal, joined into one; "" if there are none."""
        return "\n\n".join(self._queued.pop(run_id, []))

    def goal_running(self, run_id: str) -> bool:
        """Whether a goal loop is running this run (so a message sent now waits)."""
        return run_id in self._goal_runs and self.is_running(run_id)

    def _check_idle(self, run_id: str) -> None:
        if self.goal_running(run_id):
            raise GoalConflict("a goal is already running in this session: pause or "
                               "clear it first")
        if self.is_running(run_id):
            raise GoalConflict("a turn is running in this session: wait for it to "
                               "finish, or interrupt it, first")

    def _spawn_goal(self, run_id: str, preset: str | None, code_mode: bool | None,
                    task_opens_round_1: bool = False) -> None:
        self._goal_runs.add(run_id)
        self._spawn(run_id, self._drive_goal(run_id, preset, code_mode, task_opens_round_1))

    async def _drive_goal(self, run_id: str, preset: str | None, code_mode: bool | None,
                          task_opens_round_1: bool) -> None:
        try:
            await drive_goal(self, run_id, preset=preset, code_mode=code_mode,
                             task_opens_round_1=task_opens_round_1)
        except asyncio.CancelledError:
            raise  # the interrupt finalizer ends the round and stops the goal
        except Exception as e:  # noqa: BLE001 — a bug in the loop; don't leave the goal active
            if self._needs_terminal(run_id):
                self.bus.log.append(run_id, RUN_FAILED, {"error": f"{type(e).__name__}: {e}"})
            goal = goal_state(self.bus.log.events(run_id))
            if goal is not None and goal.active:
                log_status(self.bus.log, run_id, PAUSED,
                           f"the goal loop failed: {type(e).__name__}: {e}", goal.round)
        finally:
            self._tasks.pop(run_id, None)
            self._goal_runs.discard(run_id)
        queued = self.take_queued(run_id)  # the goal stopped with messages still waiting
        if queued and self.bus.log.run(run_id) is not None:
            self.send(run_id, queued, preset=preset, code_mode=code_mode)

    async def _stop_goal(self, run_id: str, goal: Goal, status: str, reason: str) -> None:
        task = self._tasks.get(run_id)
        if run_id in self._goal_runs and task is not None and not task.done():
            # Cancel the loop and wait for its finalizer, which ends the round the
            # cancel lands in and logs the stop with this status and reason.
            self._goal_stop[run_id] = (status, reason)
            task.cancel()
            await self._ensure_terminal(run_id, task)
        else:
            log_status(self.bus.log, run_id, status, reason, goal.round)

    def _settle_goal(self, run_id: str) -> None:
        """After an interrupt: a goal still active has lost its loop, so stop it,
        paused unless pause or clear asked for something else. Messages waiting
        for its next round are dropped, and the reason says so."""
        stop = self._goal_stop.pop(run_id, None)
        dropped = len(self._queued.pop(run_id, []))
        if self.bus.log.run(run_id) is None:  # deleted
            return
        goal = goal_state(self.bus.log.events(run_id))
        if goal is None or not goal.active:
            return
        status, reason = stop or (PAUSED, "the goal was interrupted")
        if dropped == 1:
            reason += "; 1 message sent during the goal wasn't delivered"
        elif dropped:
            reason += f"; {dropped} messages sent during the goal weren't delivered"
        log_status(self.bus.log, run_id, status, reason, goal.round)

    def _pause_orphaned_goals(self) -> None:
        """Pause the goals that were still active when lo last stopped. Their loop
        stopped with that process, so nothing else would ever stop them, and a
        client would see the session as busy for good. A round that was running
        gets its end event first. This assumes one lo server per database: a
        second server started on the same database would pause the first one's
        running goals."""
        log = self.bus.log
        for meta in log.runs():
            if not log.events(meta.run_id, type=GOAL_SET):
                continue
            goal = goal_state(log.events(meta.run_id))
            if goal is None or not goal.active:
                continue
            if meta.status == "running":
                log.append(meta.run_id, RUN_FAILED, {"error": "interrupted"})
            log_status(log, meta.run_id, PAUSED, "lo stopped while the goal was active",
                       goal.round)

    async def _drive(
        self,
        run_id: str,
        call,
        preset: str | None = None,
        code_mode: bool | None = None,
    ) -> None:
        try:
            agent = self._agent_for(
                run_id, preset, code_mode
            )  # may fail (upstream down)
            await call(agent)
        except asyncio.CancelledError:
            raise  # the interrupt finalizer guarantees the terminal event
        except Exception as e:
            # _loop logs RUN_FAILED itself once the loop is running, but a failure
            # in agent construction (no upstream client/caps) happens before that —
            # guarantee a terminal so subscribers don't hang.
            if self._needs_terminal(run_id):
                self.bus.log.append(
                    run_id, RUN_FAILED, {"error": f"{type(e).__name__}: {e}"}
                )
        finally:
            self._tasks.pop(run_id, None)

    def _spawn(self, run_id: str, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks[run_id] = task
        return task

    # --- read side -------------------------------------------------------

    def stream(
        self, run_id: str, *, replay: bool = True, stop_on: set[str] | None = None
    ) -> AsyncIterator[Event]:
        return self.bus.subscribe(run_id, replay=replay, stop_on=stop_on)

    def sessions(self) -> list[dict]:
        return [
            {
                "run_id": r.run_id,
                "task": r.task,
                "status": r.status,
                "created_at": r.created_at,
                "running": self.is_running(r.run_id),
            }
            for r in self.bus.log.runs()
        ]

    def delete(self, run_id: str) -> None:
        self.interrupt(run_id)
        self.bus.log.delete_run(run_id)

    async def drain(self) -> None:
        """Await all in-flight runs (used on shutdown / in tests)."""
        tasks = [t for t in self._tasks.values() if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Re-exported for the app layer's SSE serializer.
__all__ = ["SessionManager", "TERMINAL"]
