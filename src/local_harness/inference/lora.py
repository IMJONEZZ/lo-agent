"""LoRA adapters as hot-swappable skills, over HTTP.

A small base model + a library of tiny LoRA adapters = many specialized skills
without shipping many full models — the pattern that makes a versatile agent fit
on edge hardware, and one a frontier API structurally can't offer (closed weights,
no bring-your-own-adapter). This routes a per-request adapter selection onto each
server's mechanism:

  vLLM       adapters are served as extra model names; select with `model=<name>`,
             and load at runtime via POST /v1/load_lora_adapter (needs
             VLLM_ALLOW_RUNTIME_LORA_UPDATING + --enable-lora).
  llama.cpp  adapters are PRELOADED at server start (--lora a.gguf ...); select
             per-request with `{"lora": [{"id": N, "scale": S}]}`; GET /lora-adapters
             lists them.
  native     in-process PEFT hot-swap (see native/lora.py AdapterManager) — the
             on-device path, no server.

An adapter spec is "name" or "name=/path/or/hf-id" (the path is used to load it on
vLLM; llama.cpp resolves a name/id against its preloaded set).
"""

from __future__ import annotations

import httpx


async def probe_lora(client, caps) -> None:
    """Set caps.lora_mode (+ caps.lora_adapters for llama.cpp) from the server."""
    if caps.server == "vllm":
        caps.lora_mode = "vllm"  # runtime-loadable when the server enabled LoRA
        return
    if caps.server == "llama.cpp":  # NOT lmstudio (it's built on llama.cpp but can't
        try:                        # runtime-swap LoRA — merge-offline only)
            r = await client.get("/lora-adapters")
            data = r.json() if r.status_code == 200 else None
            # The genuine llama.cpp endpoint returns a JSON LIST of adapter objects
            # (possibly empty). Anything else (LM Studio answering 200 with an
            # unrelated body) must not be mistaken for a loaded adapter.
            if isinstance(data, list):
                caps.lora_mode = "llamacpp"
                caps.lora_adapters = [a for a in data if isinstance(a, dict) and "id" in a]
        except (httpx.HTTPError, ValueError):
            pass


def _llama_id(caps, adapter: str) -> int | None:
    """Resolve a llama.cpp adapter spec (id or name/path substring) to its id."""
    if adapter.isdigit():
        return int(adapter)
    for a in caps.lora_adapters or []:
        if adapter in (a.get("path") or "") or adapter == a.get("name"):
            return a.get("id")
    return None


async def ensure_adapter(client, caps, adapter: str) -> None:
    """Make `adapter` available for the next request. vLLM may need a runtime load;
    llama.cpp adapters are preloaded so this is a no-op there."""
    if caps.lora_mode != "vllm":
        return
    name, _, path = adapter.partition("=")
    name = name or adapter
    try:
        if name in await client.list_models():
            return
    except httpx.HTTPError:
        pass
    try:  # best-effort load; a failure surfaces on the actual generation request
        await client.post("/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": path or name})
    except httpx.HTTPError:
        pass


def request_overrides(caps, adapter: str) -> dict:
    """Per-request body keys (merged into sampling.extra) that route this call to
    `adapter`. Empty dict when the server has no LoRA mechanism."""
    if not adapter or not getattr(caps, "lora_mode", None):
        return {}
    if caps.lora_mode == "vllm":
        return {"model": adapter.split("=")[0]}
    if caps.lora_mode == "llamacpp":
        aid = _llama_id(caps, adapter)
        if aid is not None:
            return {"lora": [{"id": aid, "scale": 1.0}]}
    return {}
