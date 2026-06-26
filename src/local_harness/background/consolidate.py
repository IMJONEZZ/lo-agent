"""Memory consolidation: summarize completed runs into episodic memory.

Free-tokens background cognition: runs during idle time (or via
`lo background`), costs nothing but local watts.
"""

from __future__ import annotations

from ..agent.memory import Memory
from ..events.log import RUN_COMPLETED, TOOL_CALL, EventLog
from ..inference.client import OpenAICompatClient
from ..inference.types import GenerationRequest, Message, SamplingParams

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


async def consolidate(
    log: EventLog,
    memory: Memory,
    client: OpenAICompatClient,
    limit: int = 10,
    max_tokens: int = 96,
) -> int:
    """Summarize up to `limit` un-consolidated completed runs. Returns count."""
    done = 0
    for run in log.runs():
        if done >= limit or run.status != "completed" or memory.has_run(run.run_id):
            continue
        completed = log.events(run.run_id, type=RUN_COMPLETED)
        tools = [e.payload["name"] for e in log.events(run.run_id, type=TOOL_CALL)]
        answer = completed[-1].payload.get("answer", "") if completed else ""
        response = await client.chat(
            GenerationRequest(
                messages=[
                    Message(
                        role="user",
                        content=(
                            "Summarize this completed agent task in one sentence for future recall. "
                            f"Task: {run.task}\nTools used: {tools}\nAnswer: {answer[:400]}"
                        ),
                    )
                ],
                sampling=SamplingParams(
                    temperature=0.2, max_tokens=max_tokens, seed=1, extra=dict(NO_THINK)
                ),
            )
        )
        summary = response.text.strip()
        if summary:
            memory.store("episode", summary, run_id=run.run_id)
            done += 1
    return done
