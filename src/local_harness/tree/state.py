"""Tree-shaped agent state: conversation history as a tree, fork() first-class.

Cache mapping per backend:
- llama.cpp: `cache_prompt: true` (set by the adapter) means any branch that
  shares a prefix with the server's last context reuses its KV; true snapshots
  via /slots/{id}?action=save|restore when the server runs with
  --slot-save-path.
- vLLM: automatic prefix caching makes sibling rollouts nearly free; n>1
  parallel sampling forks at the sampler level.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..inference.client import OpenAICompatClient
from ..inference.types import Message

_ids = itertools.count()


@dataclass
class Node:
    id: int
    messages: list[Message]            # full history at this node
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # signals, scores, kv snapshot name

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1


class ConversationTree:
    def __init__(self, root_messages: list[Message]):
        self.root = Node(id=next(_ids), messages=list(root_messages))
        self.nodes: dict[int, Node] = {self.root.id: self.root}

    def extend(self, node: Node, *messages: Message, **meta: Any) -> Node:
        child = Node(
            id=next(_ids), messages=node.messages + list(messages), parent=node, meta=meta
        )
        node.children.append(child)
        self.nodes[child.id] = child
        return child

    def fork(self, node: Node, n: int) -> list[Node]:
        """n siblings sharing this node's full history — branch points for
        parallel rollouts. Same messages, distinct identities/metadata."""
        return [self.extend(node, fork_index=i) for i in range(n)]

    def leaves(self) -> list[Node]:
        return [n for n in self.nodes.values() if not n.children]

    def path(self, node: Node) -> list[Node]:
        out, cur = [], node
        while cur is not None:
            out.append(cur)
            cur = cur.parent
        return list(reversed(out))


class SlotSnapshots:
    """True KV-state snapshots via llama.cpp's slot persistence.

    Requires the server to run with --slot-save-path; `available()` probes by
    attempting a real save. Falls back gracefully (callers then rely on
    prefix-cache reuse instead)."""

    def __init__(self, client: OpenAICompatClient):
        self.client = client

    async def save(self, name: str, slot_id: int = 0) -> bool:
        try:
            resp = await self.client.post(
                f"/slots/{slot_id}?action=save", json={"filename": f"{name}.bin"}
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def restore(self, name: str, slot_id: int = 0) -> bool:
        try:
            resp = await self.client.post(
                f"/slots/{slot_id}?action=restore", json={"filename": f"{name}.bin"}
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def available(self) -> bool:
        return await self.save("_harness_probe")
