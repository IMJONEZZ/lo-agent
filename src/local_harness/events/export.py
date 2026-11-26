"""Render a run's event log as a Markdown transcript.

Shared by the TUI's /editor (dump to a temp file) and /export (write run-<id>.md),
and the `lo export` CLI — so they all produce the same readable transcript.

`full=True` (`lo export --full`) is the analysis view. It adds what the default
view leaves out: reasoning blocks, tool results in full, guardrail nudges, how
each turn ended (an error is never left out), the run's tags, and goal rounds.
It also follows sub-agent runs (AGENT_SPAWNED) into nested sections, so one file
holds a run's whole nested conversation history.
"""

from __future__ import annotations

import re

from .log import (
    AGENT_SPAWNED,
    CONTEXT_COMPACTED,
    GOAL_JUDGE,
    GOAL_ROUND,
    GOAL_STATUS,
    GOAL_VERIFY,
    GUARDRAIL,
    MODEL_CALL,
    RUN_COMPLETED,
    RUN_FAILED,
    TOOL_CALL,
    USER_MESSAGE,
    EventLog,
)

TOOL_RESULT_CHARS = 2000  # the default view's cut; the full view keeps every result whole
MAX_NESTING = 8  # sub-agent levels the full view follows (the Coordinator itself allows 1)


def transcript_markdown(log: EventLog, run_id: str, *, full: bool = False) -> str:
    """A human-readable Markdown transcript of a run: the task heading, then each
    user / assistant turn and the tool calls + results inline. `full` gives the
    complete nested history (see the module docstring)."""
    if full:
        return "".join(_full_run(log, run_id, depth=0, seen=set()))
    meta = log.run(run_id)
    out: list[str] = [f"# {meta.task if meta else run_id}\n"]
    for ev in log.events(run_id):
        if ev.type == USER_MESSAGE:
            out.append(f"\n## User\n\n{ev.payload.get('content', '')}\n")
        elif ev.type == MODEL_CALL:
            msg = _message(ev.payload)
            content = (msg.get("content") or "").strip()
            if content:
                out.append(f"\n## Assistant\n\n{content}\n")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                out.append(
                    f"\n### tool · {fn.get('name')}\n\n```\n{fn.get('arguments', '')}\n```\n"
                )
        elif ev.type == TOOL_CALL:
            res = (ev.payload.get("result") or "")[:TOOL_RESULT_CHARS]
            out.append(f"\n### result · {ev.payload.get('name')}\n\n```\n{res}\n```\n")
    return "".join(out)


# --- the full (nested) view -----------------------------------------------------


def _message(payload: dict) -> dict:
    return ((payload.get("response") or {}).get("choices") or [{}])[0].get("message") or {}


def _heading(level: int, text: str) -> str:
    return f"\n{'#' * min(level, 6)} {text}\n"


def _fenced(text: str) -> str:
    """A code block that can't be closed early by backticks inside `text`."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"\n{fence}\n{text}\n{fence}\n"


def _quoted(text: str) -> str:
    return "\n" + "\n".join(f"> {line}" if line else ">" for line in text.splitlines()) + "\n"


def _full_run(log: EventLog, run_id: str, depth: int, seen: set[str]) -> list[str]:
    seen.add(run_id)
    meta = log.run(run_id)
    top = depth + 1  # heading level of this run's title; its turns sit one below
    title = meta.task if meta else run_id
    out = [_heading(top, f"sub-agent · {title}" if depth else title).lstrip("\n")]
    status = meta.status if meta else "unknown"
    out.append(f"\nrun `{run_id}` · status: {status}\n")
    tags = log.run_tags(run_id)
    if tags:
        out.append("tags: " + ", ".join(f"`{k}={v}`" for k, v in tags.items()) + "\n")
    turn, detail = top + 1, top + 2
    for ev in log.events(run_id):
        p = ev.payload
        if ev.type == USER_MESSAGE:
            source = p.get("source")
            label = "User" if source in (None, "user") else f"User ({source})"
            out.append(_heading(turn, label) + f"\n{p.get('content', '')}\n")
        elif ev.type == MODEL_CALL:
            msg = _message(p)
            reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            if reasoning:
                out.append(_heading(turn, "Reasoning") + _quoted(reasoning))
            content = (msg.get("content") or "").strip()
            if content:
                out.append(_heading(turn, "Assistant") + f"\n{content}\n")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                out.append(_heading(detail, f"tool · {fn.get('name')}")
                           + _fenced(str(fn.get("arguments", ""))))
        elif ev.type == TOOL_CALL:
            out.append(_heading(detail, f"result · {p.get('name')}")
                       + _fenced(str(p.get("result") or "")))
        elif ev.type == GUARDRAIL and p.get("action") in ("nudge", "fatal"):
            what = p.get("kind") or p.get("action")
            line = f"\n*guardrail · {p.get('action')} ({what})*"
            if p.get("reason"):
                line += f": {p['reason']}"
            out.append(line + "\n")
            nudge = (p.get("nudge") or {}).get("content")
            if nudge:
                out.append(_quoted(str(nudge)))
        elif ev.type == RUN_COMPLETED:
            out.append("\n*turn ended: completed*\n")
        elif ev.type == RUN_FAILED:
            out.append(f"\n**turn ended: failed** — {p.get('error', '')}\n")
        elif ev.type == CONTEXT_COMPACTED:
            out.append(f"\n*context compacted: {p.get('before_tokens')} → "
                       f"{p.get('after_tokens')} tokens ({p.get('method')})*\n")
        elif ev.type == GOAL_ROUND:
            out.append(_heading(turn, f"Goal round {p.get('round')}"))
        elif ev.type == GOAL_JUDGE:
            out.append(f"\n*goal judge (round {p.get('round')}): {p.get('answer')}*"
                       + (f" — {p['reason']}" if p.get("reason") else "") + "\n")
        elif ev.type == GOAL_VERIFY:
            out.append(f"\n*goal verify (round {p.get('round')}): `{p.get('command')}` "
                       f"exited {p.get('exit_code')}*\n" + _fenced(str(p.get("output") or "")))
        elif ev.type == GOAL_STATUS:
            out.append(f"\n*goal {p.get('status')}: {p.get('reason')}*\n")
        elif ev.type == AGENT_SPAWNED:
            child = p.get("child_run_id")
            if not child or log.run(child) is None:
                out.append(f"\n*sub-agent run `{child}` isn't in this log*\n")
            elif child in seen:
                out.append(f"\n*sub-agent run `{child}` is shown above*\n")
            elif depth + 1 > MAX_NESTING:
                out.append(f"\n*sub-agent run `{child}` not shown: more than {MAX_NESTING} "
                           f"levels deep*\n")
            else:
                out.append("\n")
                out.extend(_full_run(log, child, depth + 1, seen))
    return out
