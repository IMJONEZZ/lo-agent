"""Coordinator: tools that let a lead agent fan work out to parallel workers and
talk to its peers — built on the SessionManager (which already drives concurrent
runs onto one bus). The lead calls spawn_agents([...]); each subtask runs as its
own event-sourced child run, and their answers come back together.

These tools are injected per-run by SessionManager._agent_for (bound to the
manager + the calling run's id), so the agent loop stays orchestration-agnostic.
"""

from __future__ import annotations

from ..agent.tools import Tool

SPAWN_AGENTS_NAME = "spawn_agents"
SEND_MESSAGE_NAME = "send_message"
MAX_FANOUT = 8  # cap a single fan-out so a runaway lead can't spawn unbounded workers


def spawn_agents_tool(manager, parent_run_id: str) -> Tool:
    async def spawn_agents(tasks, preset: str = "build") -> str:
        if isinstance(tasks, str):
            tasks = [tasks]
        tasks = [str(t).strip() for t in (tasks or []) if str(t).strip()][:MAX_FANOUT]
        if not tasks:
            return "error: spawn_agents needs a non-empty list of subtask strings"
        ids = [manager.spawn_child(t, preset, parent_run_id) for t in tasks]
        results = await manager.await_children(ids)
        out = [f"Ran {len(results)} worker agent(s) in parallel; their results:"]
        for rid, answer in results:
            out.append(f"\n── worker {rid[:8]} ──\n{answer}")
        return "\n".join(out)

    return Tool(
        name=SPAWN_AGENTS_NAME,
        description=(
            "Fan out independent subtasks to parallel worker agents and get all their "
            "results back together. Pass `tasks` as a list of self-contained subtask "
            "descriptions; they run concurrently (sharing the working directory) and you "
            "receive each worker's answer. Use it to split a large job into independent "
            "parts (audit these files, research these topics, review these modules). Avoid "
            "having two workers write the same file."),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {"type": "array", "items": {"type": "string"},
                          "description": "self-contained subtasks to run in parallel"},
                "preset": {"type": "string", "description": "worker preset (default build)"},
            },
            "required": ["tasks"],
        },
        fn=spawn_agents,
    )


def send_message_tool(manager, from_run_id: str) -> Tool:
    def send_message(to_agent: str, content: str) -> str:
        ok = manager.deliver_message(to_agent, from_run_id, content)
        return (f"delivered to {to_agent}" if ok
                else f"no running agent matches {to_agent!r} (it may have finished)")

    return Tool(
        name=SEND_MESSAGE_NAME,
        description=(
            "Send a message to another running agent by its id (e.g. a sibling worker). "
            "It arrives at that agent's next step. Useful for handing a finding to a peer "
            "mid-task. The recipient must still be running."),
        parameters={
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "the target agent's run id"},
                "content": {"type": "string"},
            },
            "required": ["to_agent", "content"],
        },
        fn=send_message,
    )
