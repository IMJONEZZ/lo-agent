"""Plan search: fork N candidate plans and pick the best by a verifier.

For the Plan preset — instead of a single-shot plan, generate several and choose
the safest/clearest. Forks share the prompt prefix, so on local hardware this is
nearly free (prefix cache); on a frontier API it would be N× the cost.
"""

from __future__ import annotations

from ...inference.capabilities import Capabilities
from ...inference.client import OpenAICompatClient
from ...inference.types import Message, SamplingParams
from .best_of_n import Candidate, JudgeVerifier, MeanLogprobVerifier, Verifier, best_of_n

DEFAULT_RUBRIC = ("a clear, safe, minimal step-by-step plan that avoids risky or "
                  "irreversible actions and reads as low-risk")


async def plan_search(
    client: OpenAICompatClient,
    caps: Capabilities,
    messages: list[Message],
    n: int = 4,
    verifier: Verifier | None = None,
    rubric: str | None = None,
    sampling: SamplingParams | None = None,
    base_seed: int = 100,
) -> list[Candidate]:
    """Return candidate plans, best-first. Default verifier judges plan safety
    (grammar-constrained 0-9) where the server supports it, else ranks by model
    confidence."""
    if verifier is None:
        verifier = (JudgeVerifier(client, caps, rubric or DEFAULT_RUBRIC)
                    if caps.grammar else MeanLogprobVerifier())
    sampling = sampling or SamplingParams(temperature=0.8, max_tokens=512)
    return await best_of_n(client, caps, messages, verifier, n=n,
                           sampling=sampling, base_seed=base_seed)
