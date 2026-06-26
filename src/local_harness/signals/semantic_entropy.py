"""Semantic entropy — the SOTA hallucination/uncertainty signal.

Farquhar et al., *Detecting hallucinations in large language models using semantic
entropy*, Nature 2024 (https://www.nature.com/articles/s41586-024-07421-0).

Sample a step K times, cluster the samples by *meaning* (bidirectional entailment),
and take the entropy over the meaning-clusters. Lexical agreement
(`tree.search.self_consistency`) counts surface-form votes — "Paris" and "It's
Paris." look like disagreement. Semantic entropy counts *distinct meanings*, so
those collapse into one cluster while two genuinely different answers stay apart.
High semantic entropy = the model is spreading probability across incompatible
answers = the calibrated signal that it's likely confabulating.

Why this is a local-only advantage: it needs (1) cheap unlimited sampling and
(2) an entailment judge. Both are free on a local model (the judge is the model
itself); on frontier APIs sampling is metered and there's no free judge. Token
logprobs alone don't give it — we established a single token's logprob ≠ truth.

The entailment judge is the local model. If a judge call fails, that pair falls
back to lexical equality so the function still returns a (degraded) clustering
rather than throwing — `judged=False` flags when any fallback happened.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..inference.capabilities import Capabilities
from ..inference.client import OpenAICompatClient
from ..inference.types import GenerationRequest, Message, SamplingParams
from ..tree.search.self_consistency import _normalize, collect_samples


@dataclass
class SemanticEntropyResult:
    entropy: float            # nats, over meaning-clusters
    normalized: float         # entropy / log(n_samples) ∈ [0,1] (0 = all agree)
    n_samples: int
    n_clusters: int
    consensus: str            # representative of the largest meaning-cluster
    clusters: list[list[str]] = field(default_factory=list)
    judged: bool = True       # False if any pair fell back to lexical equality

    def to_dict(self) -> dict:
        return {
            "entropy": self.entropy,
            "normalized": self.normalized,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "consensus": self.consensus,
            "judged": self.judged,
        }


_ENTAIL_SYS = (
    "You judge whether two answers to a question carry the same information. "
    "Think briefly if you must, but end your reply with exactly one word: yes or no."
)


def _last_user(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content or ""
    return ""


def _verdict_yes(text: str) -> bool:
    """The judge's conclusion is the LAST yes/no it utters — reasoning models
    deliberate ('...so no, actually yes') before committing."""
    hits = re.findall(r"\b(yes|no)\b", text.lower())
    return bool(hits) and hits[-1] == "yes"


async def _entails(client: OpenAICompatClient, question: str, premise: str,
                   hypothesis: str) -> bool:
    msgs = [
        Message(role="system", content=_ENTAIL_SYS),
        Message(role="user", content=(
            f"Question: {question}\n"
            f"Answer A: {premise}\n"
            f"Answer B: {hypothesis}\n"
            "Does Answer A convey the same information as Answer B for this question? "
            "Reply yes or no.")),
    ]
    # Deterministic + seeded so the uncertainty estimate is itself reproducible.
    r = await client.chat(GenerationRequest(
        messages=msgs, sampling=SamplingParams(temperature=0.0, seed=7)))
    msg = r.raw["choices"][0].get("message", {})
    text = (msg.get("content") or msg.get("reasoning_content")
            or msg.get("reasoning") or "")
    return _verdict_yes(text)


async def _equivalent(client: OpenAICompatClient, question: str, a: str, b: str) -> bool:
    """Bidirectional entailment (Farquhar's semantic-equivalence relation)."""
    return (await _entails(client, question, a, b)
            and await _entails(client, question, b, a))


async def _cluster(client: OpenAICompatClient, question: str,
                   samples: list[str]) -> tuple[list[list[str]], bool]:
    clusters: list[list[str]] = []
    reps: list[str] = []
    judged = True
    for s in samples:
        placed = False
        for i, rep in enumerate(reps):
            if _normalize(rep) == _normalize(s):
                same = True  # identical surface form — no judge call needed
            else:
                try:
                    same = await _equivalent(client, question, rep, s)
                except Exception:
                    judged = False
                    same = False  # judge unavailable → treat as distinct (lexical fallback)
            if same:
                clusters[i].append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
            reps.append(s)
    return clusters, judged


async def semantic_entropy(
    client: OpenAICompatClient,
    caps: Capabilities,
    messages: list[Message],
    n: int = 5,
    sampling: SamplingParams | None = None,
    base_seed: int = 300,
) -> SemanticEntropyResult:
    """Sample N answers, cluster by meaning, return entropy over clusters.

    `normalized` ∈ [0,1] (entropy / log N): 0 when every sample means the same
    thing, →1 when each means something different. Use it to route — escalate or
    ask the user when it's high."""
    sampling = sampling or SamplingParams(temperature=1.0, max_tokens=64)
    samples = await collect_samples(client, caps, messages, n, sampling, base_seed)
    if not samples:
        return SemanticEntropyResult(0.0, 0.0, 0, 0, "", [], judged=False)

    clusters, judged = await _cluster(client, _last_user(messages), samples)
    total = len(samples)
    probs = [len(c) / total for c in clusters]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    normalized = entropy / math.log(total) if total > 1 else 0.0
    consensus = max(clusters, key=len)[0]
    return SemanticEntropyResult(
        entropy=entropy, normalized=normalized, n_samples=total,
        n_clusters=len(clusters), consensus=consensus, clusters=clusters, judged=judged)
