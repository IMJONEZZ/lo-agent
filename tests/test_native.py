"""Phase 5: native backend, exact anti-slop, CFG/contrastive guidance,
steering, LoRA. Hermetic: tiny random GPT-2 from config + byte tokenizer."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from local_harness.logits.guidance import cfg_generate, contrastive_generate  # noqa: E402
from local_harness.native.backend import NativeBackend  # noqa: E402
from local_harness.native.lora import AdapterManager  # noqa: E402
from local_harness.native.steering import SteeringVector  # noqa: E402
from local_harness.optimize.bootstrap import Example  # noqa: E402
from local_harness.optimize.weight_opt import finetune_lora  # noqa: E402


class ByteTokenizer:
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [b + 1 for b in text.encode()]

    def decode(self, ids) -> str:
        return bytes(i - 1 for i in ids if i > 0).decode(errors="ignore")


class NoEos:
    """Test processor: mask EOS so generations always reach max_tokens."""

    def process(self, input_ids, scores):
        scores[..., 0] = -float("inf")
        return scores


@pytest.fixture(scope="module")
def backend():
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        n_layer=2, n_head=2, n_embd=32, n_positions=256, vocab_size=257,
        bos_token_id=0, eos_token_id=0,
    )
    model = transformers.GPT2LMHeadModel(config)
    return NativeBackend(model, ByteTokenizer())


def test_capabilities_tier_4(backend):
    caps = backend.capabilities()
    assert caps.in_process and caps.tier() == 4


def test_seeded_determinism(backend):
    a = backend.generate("hello", max_tokens=12, seed=7, processors=[NoEos()])
    b = backend.generate("hello", max_tokens=12, seed=7, processors=[NoEos()])
    c = backend.generate("hello", max_tokens=12, seed=8, processors=[NoEos()])
    assert a.token_ids == b.token_ids and len(a.token_ids) == 12
    assert a.token_ids != c.token_ids
    assert len(a.logprobs) == 12 and all(lp <= 0 for lp in a.logprobs)


def test_logit_processor_forces_token(backend):
    class Force:
        def process(self, input_ids, scores):
            out = torch.full_like(scores, -float("inf"))
            out[..., 66] = 0.0  # byte 'A' (65) shifted by 1
            return out

    r = backend.generate("x", max_tokens=5, seed=0, processors=[Force()])
    assert r.text == "AAAAA"


def test_exact_antislop_kv_rewind(backend):
    """Ban a substring the greedy generation provably produces; the rewind
    must eliminate it while keeping generation going."""
    plain = backend.generate("the story", max_tokens=16, temperature=0.0, processors=[NoEos()])
    assert len(plain.text) >= 4
    phrase = plain.text[1:4]  # something greedy decoding definitely emits

    banned = backend.generate(
        "the story", max_tokens=16, temperature=0.0, processors=[NoEos()],
        banned_phrases=[phrase],
    )
    assert banned.rewinds >= 1
    assert phrase.lower() not in banned.text.lower()
    assert len(banned.token_ids) > 0


def test_cfg_scale_one_matches_plain_greedy(backend):
    plain = backend.generate("abc", max_tokens=8, temperature=0.0)
    cfg = cfg_generate(backend, "abc", negative_prompt="zzz", scale=1.0,
                       max_tokens=8, temperature=0.0)
    assert cfg.token_ids == plain.token_ids  # scale=1 reduces to conditioned logits

    strong = cfg_generate(backend, "abc", negative_prompt="zzz", scale=8.0,
                          max_tokens=8, temperature=0.0)
    assert strong.token_ids != plain.token_ids  # guidance actually moves logits


def test_contrastive_decoding_runs_deterministically(backend):
    a = contrastive_generate(backend, backend, "abc", alpha=0.2, max_tokens=6)
    b = contrastive_generate(backend, backend, "abc", alpha=0.2, max_tokens=6)
    assert a.token_ids == b.token_ids
    assert all(0 <= t < 257 for t in a.token_ids)


def test_steering_changes_and_restores_output(backend, tmp_path):
    vector = SteeringVector.extract_contrastive(
        backend, "test-steer",
        positive_prompts=["happy joyful bright"], negative_prompts=["sad gloomy dark"],
        layer_indices=[0, 1],
    )
    assert set(vector.layers) == {0, 1}

    base = backend.generate("tell me", max_tokens=8, temperature=0.0, processors=[NoEos()])
    vector.apply(backend, strength=40.0)
    steered = backend.generate("tell me", max_tokens=8, temperature=0.0, processors=[NoEos()])
    vector.remove()
    restored = backend.generate("tell me", max_tokens=8, temperature=0.0, processors=[NoEos()])

    assert steered.token_ids != base.token_ids   # steering moved the distribution
    assert restored.token_ids == base.token_ids  # hooks fully removed

    # save/load round-trip has the identical effect
    vector.save(tmp_path / "v.pt")
    loaded = SteeringVector.load(tmp_path / "v.pt")
    loaded.apply(backend, strength=40.0)
    steered2 = backend.generate("tell me", max_tokens=8, temperature=0.0, processors=[NoEos()])
    loaded.remove()
    assert steered2.token_ids == steered.token_ids


def test_lora_finetune_and_hotswap(tmp_path):
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        n_layer=2, n_head=2, n_embd=32, n_positions=256, vocab_size=257,
        bos_token_id=0, eos_token_id=0,
    )
    backend = NativeBackend(transformers.GPT2LMHeadModel(config), ByteTokenizer())
    base = backend.generate("ab", max_tokens=6, temperature=0.0, processors=[NoEos()])

    losses = finetune_lora(
        backend, [Example("ab", "cdcdcd"), Example("ab", "cdcdcd")],
        tmp_path / "adapter", epochs=4, lr=5e-2, target_modules=["c_attn"],
    )
    assert losses[-1] < losses[0]  # it learned something

    manager = AdapterManager(backend)
    adapted = backend.generate("ab", max_tokens=6, temperature=0.0, processors=[NoEos()])
    with manager.disabled():
        bypassed = backend.generate("ab", max_tokens=6, temperature=0.0, processors=[NoEos()])
    assert bypassed.token_ids == base.token_ids   # disable = base behavior
    assert adapted.token_ids != base.token_ids    # adapter changed behavior


async def test_native_skill_bridge_hotswaps_real_adapter(tmp_path):
    """The on-device/edge path: a skill's LoRA adapter hot-swaps in the in-process
    backend and changes behavior — validated on a real (tiny) PEFT adapter."""
    from local_harness.native.skill_exec import generate_with_skill_native
    from local_harness.skills.skill import Skill

    torch.manual_seed(0)
    config = transformers.GPT2Config(
        n_layer=2, n_head=2, n_embd=32, n_positions=256, vocab_size=257,
        bos_token_id=0, eos_token_id=0)
    backend = NativeBackend(transformers.GPT2LMHeadModel(config), ByteTokenizer())
    base = await generate_with_skill_native(
        backend, Skill(name="t"), "ab", max_tokens=6, temperature=0.0)
    finetune_lora(backend, [Example("ab", "cdcdcd"), Example("ab", "cdcdcd")],
                  tmp_path / "a", epochs=6, lr=5e-2, target_modules=["c_attn"],
                  adapter_name="sql_lora")
    skilled = await generate_with_skill_native(
        backend, Skill(name="t", adapter="sql_lora"), "ab",
        adapters=AdapterManager(backend), max_tokens=6, temperature=0.0)
    assert skilled.adapter == "sql_lora"
    assert skilled.text != base.text       # the hot-swapped adapter changed behavior


def test_activation_probe_separates_training_classes(backend):
    from local_harness.native.steering import ActivationProbe

    probe = ActivationProbe.fit_contrastive(
        backend, "honesty",
        positive_prompts=["the sky is blue", "water is wet", "fire is hot"],
        negative_prompts=["the sky is plaid", "water is dry", "fire is cold"],
        layer_indices=[0, 1],
    )
    assert set(probe.directions) == {0, 1}

    pos = probe.score_text(backend, "the sky is blue")
    neg = probe.score_text(backend, "the sky is plaid")
    assert 0.0 <= neg.score <= 1.0 and 0.0 <= pos.score <= 1.0
    assert pos.score > 0.5 > neg.score  # calibrated threshold splits the classes


def test_activation_probe_watches_generation(backend):
    from local_harness.native.steering import ActivationProbe

    probe = ActivationProbe.fit_contrastive(
        backend, "p",
        positive_prompts=["aaa bbb"], negative_prompts=["zzz yyy"],
        layer_indices=[0, 1],
    )
    probe.attach(backend)
    backend.generate("tell me", max_tokens=6, temperature=0.0, processors=[NoEos()])
    readout = probe.readout()
    probe.detach()

    # one observation per forward pass: prompt prefill + each generated token
    assert readout.n_observations >= 6
    assert 0.0 <= readout.score <= 1.0
    assert set(readout.per_layer) == {0, 1}
    d = readout.to_dict()
    assert d["name"] == "p" and "0" in d["per_layer"]

    # detached: hooks are gone, generation records nothing new
    backend.generate("tell me", max_tokens=3, temperature=0.0, processors=[NoEos()])
    assert probe.readout().n_observations == readout.n_observations


def test_activation_probe_save_load_roundtrip(backend, tmp_path):
    from local_harness.native.steering import ActivationProbe

    probe = ActivationProbe.fit_contrastive(
        backend, "rt",
        positive_prompts=["good kind true"], negative_prompts=["bad cruel false"],
        layer_indices=[1],
    )
    before = probe.score_text(backend, "some new text").score
    probe.save(tmp_path / "probe.pt")
    loaded = ActivationProbe.load(tmp_path / "probe.pt")
    assert abs(loaded.score_text(backend, "some new text").score - before) < 1e-9
