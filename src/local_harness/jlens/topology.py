"""Model-topology helpers for the lens — chiefly MTP/NextN awareness.

The Jacobian lens reads the residual stream ``l_out-<il>``. On multi-token-
prediction (MTP / NextN) checkpoints the GGUF's ``block_count`` includes the
NextN draft layers, but only the *base* layers emit ``l_out`` and the readable
final layer is the last base layer — NOT ``block_count - 1``. Getting this
wrong makes the sidecar's self-check report a false ``l_out_ok: false`` and
makes the bridge request activations for a layer that never captured
(KeyError). These helpers centralize the one correct rule.

Discovered live on qwen3.6-27b-mtp (block_count=65, nextn_predict_layers=1 →
64 emitting layers, final readable layer index 63). See
docs/jlens-integration.md.
"""

from __future__ import annotations


def emitting_layers(block_count: int, n_nextn: int) -> int:
    """Number of layers that emit ``l_out`` = base blocks (excludes NextN)."""
    n = int(block_count) - max(0, int(n_nextn))
    if n < 1:
        raise ValueError(f"nonsensical topology: block_count={block_count}, n_nextn={n_nextn}")
    return n


def final_readable_layer(block_count: int, n_nextn: int) -> int:
    """Index of the last layer whose residual is readable (== model output)."""
    return emitting_layers(block_count, n_nextn) - 1


def nextn_layers_from_gguf(reader, arch: str) -> int:
    """Read the NextN/MTP layer count from a gguf reader (0 if absent)."""
    for key in (f"{arch}.nextn_predict_layers", f"{arch}.n_layer_nextn"):
        f = reader.fields.get(key)
        if f is not None:
            try:
                return int(f.contents())
            except Exception:
                pass
    return 0
