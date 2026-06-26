"""Phase 3: tree state, slot snapshots, best-of-N, beam, anti-slop rewind."""

from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.inference.types import Message, SamplingParams, TokenLogprob
from local_harness.logits.antislop import generate_antislop
from local_harness.tree.search.beam import beam_search
from local_harness.tree.search.best_of_n import (
    Candidate,
    MeanLogprobVerifier,
    SkillValidityVerifier,
    best_of_n,
)
from local_harness.tree.state import ConversationTree, SlotSnapshots

from mocks import MockLlamaCpp, chat_response, mock_token_id

CAPS = Capabilities(server="llama.cpp", seed=True, logprobs=True, grammar="gbnf", logit_bias=True)


def test_tree_fork_and_path():
    tree = ConversationTree([Message(role="user", content="hi")])
    forks = tree.fork(tree.root, 3)
    assert len(forks) == 3
    assert all(f.messages == tree.root.messages for f in forks)
    assert all(f.parent is tree.root for f in forks)

    child = tree.extend(forks[0], Message(role="assistant", content="hello"), score=1.0)
    assert child.depth == 2
    assert [n.id for n in tree.path(child)] == [tree.root.id, forks[0].id, child.id]
    assert child in tree.leaves() and forks[1] in tree.leaves()


async def test_slot_snapshots():
    mock = MockLlamaCpp()
    async with OpenAICompatClient("http://t", "m", transport=mock.transport()) as client:
        snaps = SlotSnapshots(client)
        assert await snaps.available()
        assert await snaps.save("branch_a")
        assert await snaps.restore("branch_a")
        assert "branch_a.bin" in mock.saved_slots

    disabled = MockLlamaCpp(slot_save_enabled=False)
    async with OpenAICompatClient("http://t", "m", transport=disabled.transport()) as client:
        assert not await SlotSnapshots(client).available()


async def test_best_of_n_picks_highest_confidence():
    # Seeds 100..102 produce answers with increasing confidence.
    def resp(content, lp):
        r = chat_response(content=content)
        for t in r["choices"][0]["logprobs"]["content"]:
            t["logprob"] = lp
        return r

    script = {100: resp("meh", -2.0), 101: resp("best", -0.1), 102: resp("ok", -1.0)}
    client = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script=script).transport())
    ranked = await best_of_n(
        client, CAPS, [Message(role="user", content="q")],
        MeanLogprobVerifier(), n=3, base_seed=100,
    )
    assert [c.text for c in ranked] == ["best", "ok", "meh"]


async def test_best_of_n_with_validity_verifier():
    from local_harness.skills.ir import Grammar
    from local_harness.skills.skill import Skill

    skill = Skill(name="yn", grammar=Grammar.from_rules({"v": '"yes" | "no"'}, root="v"))
    script = {
        100: chat_response(content="well, probably yes"),  # invalid
        101: chat_response(content="yes"),                 # valid
    }
    client = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script=script).transport())
    ranked = await best_of_n(
        client, CAPS, [Message(role="user", content="q")],
        SkillValidityVerifier(skill), n=2, base_seed=100,
    )
    assert ranked[0].text == "yes" and ranked[0].score > 0 > ranked[1].score


async def test_beam_search_finds_terminal():
    """Toy beam: states are strings of 'a'/'b'; terminal at depth 3; 'bbb' is best."""
    tree = ConversationTree([Message(role="user", content="start")])

    async def expand(node, k):
        prefix = node.meta.get("s", "")
        return [tree.extend(node, s=prefix + c) for c in ("a", "b")[:k]]

    async def score(node):
        return node.meta.get("s", "").count("b")

    def is_terminal(node):
        return len(node.meta.get("s", "")) >= 3

    best = await beam_search(tree, expand, score, is_terminal, width=2, expansions=2, max_depth=5)
    assert best.meta["s"] == "bbb"


async def test_antislop_rewinds_and_bans():
    """Mock model says 'we delve into X' until 'delve'-ish tokens are banned."""
    delve_ids = {str(mock_token_id("delve")), str(mock_token_id(" delve"))}

    def completion_fn(prompt, body):
        banned = set((body.get("logit_bias") or {}).keys())
        if delve_ids & banned:
            return " we examine the data closely.", "stop"
        return " we delve into the data.", "stop"

    mock = MockLlamaCpp(completion_fn=completion_fn)
    async with OpenAICompatClient("http://t", "m", transport=mock.transport()) as client:
        result = await generate_antislop(
            client, [Message(role="user", content="describe")],
            banned_phrases=["delve"], seed=1,
        )
    assert "delve" not in result.text.lower()
    assert "examine" in result.text
    assert result.rewinds == 1
    assert set(map(str, result.banned_token_ids)) == delve_ids


async def test_antislop_rewinds_on_slop_structure():
    """A slop *structure* (not a fixed phrase) is rewound and the seed bumped to
    force a different, clean continuation."""
    from local_harness.logits.slop import SlopDetector

    def completion_fn(prompt, body):
        if body.get("seed", 0) == 5:  # original seed → antithesis slop
            return " It's not a tool, it's a platform.", "stop"
        return " A small CLI that runs models locally.", "stop"  # after the seed bump

    mock = MockLlamaCpp(completion_fn=completion_fn)
    async with OpenAICompatClient("http://t", "m", transport=mock.transport()) as client:
        result = await generate_antislop(
            client, [Message(role="user", content="describe it")],
            banned_phrases=[], seed=5, slop_detector=SlopDetector())
    assert result.rewinds == 1
    assert "it's not a tool, it's" not in result.text.lower()  # structure gone
    assert "small cli" in result.text.lower()                  # clean continuation kept


async def test_self_consistency_takes_the_majority(tmp_path):
    from local_harness.inference.capabilities import Capabilities
    from local_harness.tree.search.self_consistency import self_consistency

    # seeds 200,201 → "yes", 202 → "no"  ⇒ consensus "yes", agreement 2/3
    script = {200: chat_response(content="yes"), 201: chat_response(content="yes"),
              202: chat_response(content="no")}
    client = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script=script).transport())
    answer, agreement = await self_consistency(
        client, Capabilities(), [Message(role="user", content="?")], n=3, base_seed=200)
    assert answer == "yes"
    assert abs(agreement - 2 / 3) < 0.01


async def test_plan_search_picks_the_safest_plan(tmp_path):
    from local_harness.inference.capabilities import Capabilities
    from local_harness.tree.search.plan_search import plan_search

    script = {100: chat_response(content="plan A: drop the table"),
              101: chat_response(content="plan B: migrate with a rollback step"),
              102: chat_response(content="plan C: edit configs in place")}
    client = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script=script).transport())

    class SafetyVerifier:
        async def score(self, c):
            return 1.0 if "rollback" in c.text else 0.0

    cands = await plan_search(client, Capabilities(), [Message(role="user", content="migrate?")],
                              n=3, verifier=SafetyVerifier(), base_seed=100)
    assert len(cands) == 3
    assert "rollback" in cands[0].text          # the safest plan is chosen first


async def test_self_consistency_full_agreement(tmp_path):
    from local_harness.inference.capabilities import Capabilities
    from local_harness.tree.search.self_consistency import self_consistency

    script = {200: chat_response(content="42"), 201: chat_response(content="42"),
              202: chat_response(content="42")}
    client = OpenAICompatClient("http://t", "m", transport=MockLlamaCpp(script=script).transport())
    answer, agreement = await self_consistency(
        client, Capabilities(), [Message(role="user", content="?")], n=3, base_seed=200)
    assert answer == "42" and agreement == 1.0


async def test_antislop_clean_generation_no_rewind():
    mock = MockLlamaCpp(completion_fn=lambda p, b: (" a perfectly clean sentence.", "stop"))
    async with OpenAICompatClient("http://t", "m", transport=mock.transport()) as client:
        result = await generate_antislop(
            client, [Message(role="user", content="x")], banned_phrases=["delve", "tapestry"]
        )
    assert result.rewinds == 0 and "clean" in result.text
