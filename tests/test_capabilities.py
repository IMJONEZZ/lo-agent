import httpx

from local_harness.inference.capabilities import Capabilities, probe
from local_harness.inference.client import OpenAICompatClient

from mocks import MockGeneric, MockLlamaCpp, chat_response


def _lmstudio_handler(request: httpx.Request) -> httpx.Response:
    """LM Studio: no logprobs on chat-completions, but yes on /v1/responses."""
    path = request.url.path
    if path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "glm", "owned_by": "lmstudio"}]})
    if path == "/slots":
        return httpx.Response(200, json=[{"id": 0}])
    if path == "/props":
        return httpx.Response(200, json={})
    if path == "/v1/chat/completions":
        return httpx.Response(200, json=chat_response(content="hi", with_logprobs=False))
    if path == "/v1/responses":
        return httpx.Response(200, json={"output": [{"content": [
            {"type": "output_text", "text": "hi",
             "logprobs": [{"token": "hi", "logprob": -0.1}]}]}]})
    return httpx.Response(404)


async def test_probe_recovers_logprobs_from_responses_api():
    client = OpenAICompatClient("http://lm", "glm",
                                transport=httpx.MockTransport(_lmstudio_handler))
    caps = await probe(client)
    assert caps.responses_api is True
    # chat-completions had none, but the harness recovered logprobs via Responses
    assert caps.logprobs is True


async def test_responses_client_call():
    client = OpenAICompatClient("http://lm", "glm",
                                transport=httpx.MockTransport(_lmstudio_handler))
    data = await client.responses("hi", include=["message.output_text.logprobs"])
    assert data["output"][0]["content"][0]["logprobs"][0]["token"] == "hi"


def test_tier_ladder():
    assert Capabilities().tier() == 0
    assert Capabilities(seed=True, logprobs=True).tier() == 1
    assert Capabilities(seed=True, logprobs=True, grammar="gbnf", logit_bias=True).tier() == 2
    assert Capabilities(
        seed=True, logprobs=True, grammar="gbnf", logit_bias=True, kv_snapshot=True
    ).tier() == 3
    assert Capabilities(
        seed=True, logprobs=True, grammar="guided", logit_bias=True, parallel_n=True
    ).tier() == 3
    assert Capabilities(in_process=True).tier() == 4
    # logprobs alone isn't Tier 1, and grammar without seed doesn't skip tiers
    assert Capabilities(logprobs=True, grammar="gbnf", logit_bias=True).tier() == 0


async def test_probe_llamacpp_is_tier_3():
    mock = MockLlamaCpp()
    async with OpenAICompatClient("http://test", "test-model", transport=mock.transport()) as client:
        caps = await probe(client)
    assert caps.server == "llama.cpp"
    assert caps.seed is True          # identical seeded responses
    assert caps.logprobs is True
    assert caps.grammar == "gbnf"
    assert caps.kv_snapshot is True   # /slots responded
    assert caps.raw_completion is True
    assert "dry" in caps.sampler_zoo and "mirostat" in caps.sampler_zoo
    assert caps.tier() == 3


async def test_probe_generic_is_tier_0():
    mock = MockGeneric()
    async with OpenAICompatClient("http://test", "test-model", transport=mock.transport()) as client:
        caps = await probe(client)
    assert caps.server == "generic"
    assert caps.seed is False         # responses differed despite same seed
    assert caps.logprobs is False
    assert caps.grammar is None
    assert caps.kv_snapshot is False
    assert caps.raw_completion is False
    assert caps.tier() == 0


async def test_probe_detects_post_sampling_probs():
    # recent llama.cpp lists post_sampling_probs among the /props default params
    mock = MockLlamaCpp(post_sampling=True)
    async with OpenAICompatClient("http://t", "test-model", transport=mock.transport()) as c:
        caps = await probe(c)
    assert caps.post_sampling_probs is True

    old = MockLlamaCpp()  # older server: key absent from /props
    async with OpenAICompatClient("http://t", "test-model", transport=old.transport()) as c:
        caps = await probe(c)
    assert caps.post_sampling_probs is False


def test_prob_shaped_logprobs_parse_to_log_space():
    import math

    from local_harness.inference.types import GenerationResponse
    from mocks import _to_post_sampling_shape

    resp = _to_post_sampling_shape(chat_response(content="hi"))
    parsed = GenerationResponse.from_chat_response(resp, timing_ms=0.0)
    assert parsed.logprobs is not None
    # mock logprobs are -0.05 / -0.50; prob-shape round-trips back to log-space
    assert abs(parsed.logprobs[0].logprob - (-0.05)) < 1e-9
    assert abs(parsed.logprobs[1].logprob - (-0.50)) < 1e-9
    assert abs(parsed.logprobs[0].top[0][1] - (-0.05)) < 1e-9
