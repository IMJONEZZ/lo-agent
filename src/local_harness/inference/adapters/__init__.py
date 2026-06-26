from .base import Adapter, Fingerprint
from .generic import GenericAdapter
from .llamacpp import LlamaCppAdapter
from .vllm import VllmAdapter

# Order matters: first adapter whose `matches()` returns True wins;
# GenericAdapter matches everything and must stay last.
ADAPTERS: list[type[Adapter]] = [LlamaCppAdapter, VllmAdapter, GenericAdapter]

__all__ = ["ADAPTERS", "Adapter", "Fingerprint", "GenericAdapter", "LlamaCppAdapter", "VllmAdapter"]
