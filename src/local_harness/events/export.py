"""Render a run's event log as a Markdown transcript.

Shared by the TUI's /editor (dump to a temp file) and /export (write run-<id>.md),
and the `lo export` CLI — so they all produce the same readable transcript.
"""

from __future__ import annotations

from .log import MODEL_CALL, TOOL_CALL, USER_MESSAGE, EventLog


def transcript_markdown(log: EventLog, run_id: str) -> str:
    """A human-readable Markdown transcript of a run: the task heading, then each
    user / assistant turn and the tool calls + results inline."""
    meta = log.run(run_id)
    out: list[str] = [f"# {meta.task if meta else run_id}\n"]
    for ev in log.events(run_id):
        if ev.type == USER_MESSAGE:
            out.append(f"\n## User\n\n{ev.payload.get('content', '')}\n")
        elif ev.type == MODEL_CALL:
            msg = (
                (ev.payload.get("response") or {})
                .get("choices", [{}])[0]
                .get("message", {})
            )
            content = (msg.get("content") or "").strip()
            if content:
                out.append(f"\n## Assistant\n\n{content}\n")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                out.append(
                    f"\n### tool · {fn.get('name')}\n\n```\n{fn.get('arguments', '')}\n```\n"
                )
        elif ev.type == TOOL_CALL:
            res = (ev.payload.get("result") or "")[:2000]
            out.append(f"\n### result · {ev.payload.get('name')}\n\n```\n{res}\n```\n")
    return "".join(out)
