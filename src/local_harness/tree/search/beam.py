"""Beam search over agent steps (generic over expand/score/terminal)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..state import ConversationTree, Node


async def beam_search(
    tree: ConversationTree,
    expand: Callable[[Node, int], Awaitable[list[Node]]],
    score: Callable[[Node], Awaitable[float]],
    is_terminal: Callable[[Node], bool],
    width: int = 2,
    expansions: int = 2,
    max_depth: int = 4,
) -> Node:
    """Expand the frontier `expansions` ways per node, keep the best `width`
    nodes by score, stop when the best node is terminal or depth runs out.

    `expand(node, k)` generates k children (via tree.extend / tree.fork);
    `score` typically combines StepSignals confidence with a verifier.
    """
    frontier = [tree.root]
    best: tuple[float, Node] | None = None

    for _ in range(max_depth):
        children: list[Node] = []
        for node in frontier:
            if is_terminal(node):
                continue
            children.extend(await expand(node, expansions))
        if not children:
            break
        scored = [(await score(c), c) for c in children]
        scored.sort(key=lambda sc: sc[0], reverse=True)
        for s, node in scored:
            if is_terminal(node) and (best is None or s > best[0]):
                best = (s, node)
        frontier = [c for _, c in scored[:width]]
        if best is not None and all(is_terminal(c) for c in frontier):
            break

    if best is not None:
        return best[1]
    # No terminal found: return the best-scored frontier node.
    scored = [(await score(c), c) for c in frontier]
    return max(scored, key=lambda sc: sc[0])[1]
