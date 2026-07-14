"""Corrective nudge messages (after forge, MIT). Channel separation matters:
format problems go back as user messages; tool-shaped problems go back on the
tool channel so the model sees them as tool feedback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Nudge:
    role: str       # "user" | "tool"
    content: str
    kind: str       # retry | unknown_tool | bad_args | step | prerequisite
    tool_call_id: str | None = None  # set for role="tool" injections


def retry_nudge() -> str:
    return (
        "Your previous response was not a valid tool call or final answer for "
        "this task. Either call one of the available tools, or finish with a "
        "plain final answer. Do not describe a tool call in prose."
    )


def unknown_tool_nudge(tool_name: str, available: list[str]) -> str:
    return (
        f"Tool '{tool_name}' does not exist. "
        f"Available tools: {', '.join(available)}. Call one of them."
    )


def bad_args_nudge(tool_name: str, raw_args: str) -> str:
    return (
        f"Tool call to '{tool_name}' had malformed arguments: {raw_args!r}. "
        'Arguments must be a JSON object — {} for no-arg tools or {"key": value}. '
        "Re-emit the tool call with valid JSON arguments."
    )


def step_nudge(attempted: str, pending: list[str], tier: int = 1) -> str:
    tier = max(1, min(3, tier))
    steps = ", ".join(pending)
    if tier == 1:
        return (
            f"You cannot finish with {attempted} yet. Required steps remain: "
            f"{steps}. Call one of them now."
        )
    if tier == 2:
        return f"You must call one of these tools now: {steps}. Pick one."
    return (
        f"STOP. You MUST call one of: {steps}. Do NOT use {attempted}. "
        f"Your next response MUST be a tool call to one of: {steps}."
    )


def prerequisite_nudge(tool_name: str, missing: list[str]) -> str:
    return (
        f"You cannot call {tool_name} yet. You must first call: "
        f"{', '.join(missing)}. Call the prerequisite tool now."
    )


def doom_loop_nudge(tool_name: str, n: int) -> str:
    return (
        f"You have called {tool_name} with the same arguments {n} times — it is "
        "not making progress. Stop repeating it: change the arguments, try a "
        "different approach, or finish with a plain final answer stating what you "
        "found and what is still blocked. Do NOT repeat that exact call."
    )
