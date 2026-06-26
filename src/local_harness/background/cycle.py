"""One pass of background cognition — the 'overnight apprentice'.

Runs consolidate + reflect + auto_skills together and reports what it learned.
On local models this is *free marginal cost*, so it can run on every idle moment
instead of being rationed like a frontier agent's background token spend.
"""

from __future__ import annotations

from pathlib import Path

from ..agent.memory import Memory
from ..events.log import EventLog
from ..inference.client import OpenAICompatClient
from .auto_skills import auto_skills
from .consolidate import consolidate
from .reflect import reflect


async def background_cycle(
    log: EventLog,
    memory: Memory,
    client: OpenAICompatClient,
    drafts_dir: str | Path,
    limit: int = 5,
    caps=None,
    min_agreement: float = 0.5,
) -> dict[str, int]:
    """Returns counts: {episodes, lessons, skills}. Lessons are gated on
    sample-consistency (`min_agreement`), not token-logprob."""
    episodes = await consolidate(log, memory, client, limit=limit)
    lessons = await reflect(log, memory, client, limit=limit, caps=caps,
                            min_agreement=min_agreement)
    skills = await auto_skills(log, client, drafts_dir, memory=memory, limit=limit)
    return {"episodes": episodes, "lessons": lessons, "skills": len(skills)}


def summarize_cycle(counts: dict[str, int]) -> str:
    return (f"overnight apprentice: +{counts['skills']} skills · "
            f"+{counts['lessons']} lessons · +{counts['episodes']} memories · $0 spent")
