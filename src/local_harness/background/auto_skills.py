"""Hermes-style self-improvement: after a *successful* run that used several
tools, write a reusable skill document (approach / edge cases / domain knowledge).

The doc lands in the drafts dir; a one-line summary is stored in memory (kind
'skill') so retrieval-augmented runs surface it on the next similar task — closing
the loop. Free local tokens mean this can run as an always-on background pass.
"""

from __future__ import annotations

from pathlib import Path

from ..agent.memory import Memory
from ..events.log import RUN_COMPLETED, TOOL_CALL, EventLog
from ..inference.client import OpenAICompatClient
from ..inference.types import GenerationRequest, Message, SamplingParams

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


async def auto_skills(
    log: EventLog,
    client: OpenAICompatClient,
    drafts_dir: str | Path,
    memory: Memory | None = None,
    min_tools: int = 5,
    limit: int = 5,
    max_tokens: int = 400,
) -> list[str]:
    """Returns the paths of skill docs written this pass."""
    drafts = Path(drafts_dir)
    drafts.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for run in log.runs():
        if len(created) >= limit or run.status != "completed":
            continue
        tools = log.events(run.run_id, type=TOOL_CALL)
        if len(tools) < min_tools:
            continue
        path = drafts / f"auto_{run.run_id[:8]}.md"
        if path.exists():
            continue
        tool_seq = ", ".join(t.payload["name"] for t in tools)
        done = log.events(run.run_id, type=RUN_COMPLETED)
        answer = done[-1].payload.get("answer", "") if done else ""
        response = await client.chat(GenerationRequest(
            messages=[Message(role="user", content=(
                "Write a reusable skill document (Markdown) for this kind of task, using "
                "exactly these sections: ## Approach, ## Edge cases, ## Domain knowledge. "
                "Be concise and concrete.\n\n"
                f"Task: {run.task}\nTools used: {tool_seq}\nResult: {answer[:600]}"))],
            sampling=SamplingParams(temperature=0.2, max_tokens=max_tokens, seed=1,
                                    extra=dict(NO_THINK))))
        doc = response.text.strip()
        if not doc:
            continue
        path.write_text(f"# {run.task[:80]}\n\n{doc}\n")
        created.append(str(path))
        if memory is not None:
            first = next((ln for ln in doc.splitlines() if ln.strip()
                          and not ln.startswith("#")), "")
            memory.store("skill", f"For '{run.task[:60]}': {first[:160]}",
                         run_id=run.run_id)
    return created
