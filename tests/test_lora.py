"""LoRA-as-skills over HTTP: per-request adapter routing for vLLM and llama.cpp.

Live validation (a real adapter on a LoRA-enabled server) happens during the spin;
here we lock the routing contract — the right body keys reach the right server.
"""

import json

import httpx

from local_harness.inference.capabilities import Capabilities
from local_harness.inference.client import OpenAICompatClient
from local_harness.inference.lora import probe_lora, request_overrides
from local_harness.skills.skill import Skill
from local_harness.skills.exec import generate_with_skill

from mocks import chat_response


def test_request_overrides_per_backend():
    vllm = Capabilities(server="vllm", lora_mode="vllm")
    assert request_overrides(vllm, "sql_expert") == {"model": "sql_expert"}
    assert request_overrides(vllm, "sql=/path/to/sql") == {"model": "sql"}  # strips the path

    llama = Capabilities(server="llama.cpp", lora_mode="llamacpp",
                         lora_adapters=[{"id": 0, "path": "base.gguf"},
                                        {"id": 2, "path": "sql_lora.gguf"}])
    assert request_overrides(llama, "2") == {"lora": [{"id": 2, "scale": 1.0}]}      # by id
    assert request_overrides(llama, "sql_lora") == {"lora": [{"id": 2, "scale": 1.0}]}  # by name
    assert request_overrides(llama, "unknown") == {}                                 # not loaded

    assert request_overrides(Capabilities(server="generic"), "x") == {}              # no LoRA


class _Recorder(httpx.AsyncBaseTransport):
    def __init__(self, lora_adapters=None):
        self.bodies = []
        self.loaded = []
        self.lora_adapters = lora_adapters or []

    async def handle_async_request(self, request):
        path = request.url.path
        if path == "/lora-adapters":
            return httpx.Response(200, json=self.lora_adapters)
        if path == "/v1/load_lora_adapter":
            self.loaded.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        body = json.loads(request.content)
        self.bodies.append(body)
        return httpx.Response(200, json=chat_response(content="yes"))


class _LMStudioish(httpx.AsyncBaseTransport):
    """LM Studio: built on llama.cpp (serves /props), but also serves /v1/responses
    and answers /lora-adapters with a non-list body — must NOT be taken for a
    runtime-LoRA llama.cpp server."""
    async def handle_async_request(self, request):
        p = request.url.path
        if p == "/props":
            return httpx.Response(200, json={"default_generation_settings": {}})
        if p == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "nemotron", "owned_by": "lmstudio"}]})
        if p == "/v1/responses":
            return httpx.Response(200, json={"output": []})
        if p == "/lora-adapters":
            return httpx.Response(200, json={"error": "not supported"})  # not a list
        if p in ("/slots",):
            return httpx.Response(404)
        # real LM Studio returns NO chat-completions logprobs (only via /v1/responses)
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}], "usage": {}})


async def test_lmstudio_not_mislabeled_llamacpp_nor_false_lora():
    from local_harness.inference.capabilities import probe
    client = OpenAICompatClient("http://lmstudio", "nemotron", transport=_LMStudioish())
    caps = await probe(client)
    assert caps.server == "lmstudio"      # relabeled off llama.cpp via /v1/responses
    assert caps.lora_mode is None         # no false "llamacpp · preloaded"
    assert caps.responses_api is True


async def test_probe_lora_llamacpp_lists_preloaded_adapters():
    rec = _Recorder(lora_adapters=[{"id": 0, "path": "sql.gguf", "scale": 1.0}])
    client = OpenAICompatClient("http://llama", "m", transport=rec)
    caps = Capabilities(server="llama.cpp")
    await probe_lora(client, caps)
    assert caps.lora_mode == "llamacpp" and caps.lora_adapters[0]["id"] == 0


async def test_skill_routes_to_vllm_adapter_and_loads_it():
    rec = _Recorder()
    client = OpenAICompatClient("http://vllm", "base", transport=rec)
    caps = Capabilities(server="vllm", lora_mode="vllm")
    skill = Skill(name="sqled", adapter="sql_expert=/models/sql")
    await generate_with_skill(client, caps, skill, "SELECT?", max_attempts=1)
    assert rec.loaded and rec.loaded[0]["lora_name"] == "sql_expert"   # runtime-loaded
    assert rec.bodies[-1]["model"] == "sql_expert"                     # request routed to it


class _FakeNative:
    """A NativeBackend stand-in (no torch) recording seeds + active adapter."""
    def __init__(self, text="yes"):
        self.text = text
        self.seeds = []

    def generate(self, prompt, max_tokens=64, temperature=1.0, seed=0, processors=None):
        from local_harness.native.backend import NativeResult
        self.seeds.append(seed)
        return NativeResult(text=self.text, token_ids=[], logprobs=[], rewinds=0)


class _FakeAdapters:
    def __init__(self):
        self.activated = []

    def with_adapter(self, name):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            self.activated.append(name)
            yield
        return _cm()


async def test_native_skill_hotswaps_the_adapter():
    from local_harness.native.skill_exec import generate_with_skill_native
    backend, adapters = _FakeNative("yes"), _FakeAdapters()
    skill = Skill(name="sqled", adapter="sql_lora")
    res = await generate_with_skill_native(backend, skill, "go", adapters=adapters, max_attempts=1)
    assert res.text == "yes" and res.adapter == "sql_lora"
    assert adapters.activated == ["sql_lora"]      # the adapter was swapped in for the call


async def test_native_skill_without_adapter_runs_base():
    from local_harness.native.skill_exec import generate_with_skill_native
    res = await generate_with_skill_native(_FakeNative("hello"), Skill(name="t"), "go", max_attempts=1)
    assert res.text == "hello" and res.adapter is None


async def test_skill_routes_to_llamacpp_adapter_by_name():
    rec = _Recorder(lora_adapters=[{"id": 3, "path": "sql_lora.gguf"}])
    client = OpenAICompatClient("http://llama", "base", transport=rec)
    caps = Capabilities(server="llama.cpp", lora_mode="llamacpp",
                        lora_adapters=[{"id": 3, "path": "sql_lora.gguf"}])
    skill = Skill(name="sqled", adapter="sql_lora")
    await generate_with_skill(client, caps, skill, "SELECT?", max_attempts=1)
    assert rec.bodies[-1]["lora"] == [{"id": 3, "scale": 1.0}]         # per-request adapter
    assert not rec.loaded                                              # llama.cpp preloads, no load call
