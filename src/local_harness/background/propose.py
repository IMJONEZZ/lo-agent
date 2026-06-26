"""Autonomous-action drafts (opt-in).

Our default is "learn, don't act". This narrows the gap a little, safely: for
runs that STALLED (failed / hit max steps), draft a PROPOSED next action and
write it to the drafts dir as a Markdown file. Nothing is executed — a human
reviews and promotes it. Enabled only via `lo background --autonomous-actions`.
"""

from __future__ import annotations

from pathlib import Path

from ..events.log import RUN_FAILED, TOOL_CALL, EventLog
from ..inference.client import OpenAICompatClient
from ..inference.types import GenerationRequest, Message, SamplingParams

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


async def propose_actions(
    log: EventLog, client: OpenAICompatClient, drafts_dir, limit: int = 5
) -> list[str]:
    """Draft proposed next actions for up to `limit` stalled runs (never executed).
    Idempotent: skips runs that already have a draft. Returns the files written."""
    out: list[str] = []
    d = Path(drafts_dir)
    d.mkdir(parents=True, exist_ok=True)
    for run in log.runs():
        if len(out) >= limit:
            break
        if run.status != "failed":  # only stalled/failed runs need a proposal
            continue
        dest = d / f"proposed-{run.run_id[:8]}.md"
        if dest.exists():
            continue
        failed = log.events(run.run_id, type=RUN_FAILED)
        err = failed[-1].payload.get("error", "") if failed else ""
        tools = [e.payload["name"] for e in log.events(run.run_id, type=TOOL_CALL)]
        resp = await client.chat(
            GenerationRequest(
                messages=[
                    Message(
                        role="user",
                        content=(
                            "An agent task did not complete. Draft ONE concrete proposed next action to "
                            "move it forward (a fix or a plan), in a few sentences. Do NOT execute "
                            "anything — only describe the proposed action.\n"
                            f"Task: {run.task}\nTools used: {tools}\nFailure: {err[:300]}"
                        ),
                    )
                ],
                sampling=SamplingParams(
                    temperature=0.3, max_tokens=256, seed=1, extra=dict(NO_THINK)
                ),
            )
        )
        text = resp.text.strip()
        if not text:
            continue
        dest.write_text(
            f"# Proposed action — run {run.run_id[:8]}\n\n"
            f"**Task:** {run.task}\n\n**Why it stalled:** {err[:300]}\n\n"
            f"**Proposed next action (NOT executed — review before acting):**\n\n{text}\n"
        )
        out.append(str(dest))
    return out
