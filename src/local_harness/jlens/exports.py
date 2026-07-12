"""Export J-space edits as files the user's EXISTING server loads — no process
changes, no sidecar. Three targets:

- control vector (GGUF): additive steering llama.cpp applies with
  ``--control-vector f.gguf --control-vector-layer-range a b``. Steer/suppress
  only (additive), on a stock llama-server, one relaunch.
- abliterated model (GGUF): orthogonalize the writer matrices (o_proj,
  down_proj, token embeddings) against a concept direction so the model can no
  longer write it — a new GGUF that works on ANY server (LM Studio, Ollama).
- LoRA (GGUF): the same edits as a hot-swappable low-rank adapter.

The control-vector path is the cleanest "steer your existing server" proof; the
abliteration path is the weight-space form of the user's "abliterate" goal.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------- control vector ------- #


def build_control_vector(readout, specs: list[dict], *, layers: list[int] | None = None,
                         norms: dict[int, float] | None = None) -> dict[int, np.ndarray]:
    """Per-layer additive direction from steer specs.

    For each steer spec, adds ``alpha · h_rms · v̂_t`` at each fitted layer in
    range (the same vector the live steer path uses). Negative alpha suppresses.
    Returns ``{layer: direction[d_model]}``.
    """
    fitted = set(readout.lens.source_layers)
    out: dict[int, np.ndarray] = {}
    for spec in specs:
        if spec.get("type") != "steer":
            raise ValueError("control vectors are additive — only 'steer' specs "
                             "(use export_abliterated for ablate/swap)")
        tid = int(spec["token_id"])
        alpha = float(spec.get("alpha", 2.0))
        lr = spec.get("layers")
        sel = sorted(fitted) if lr is None else [l for l in sorted(fitted) if lr[0] <= l <= lr[1]]
        if layers is not None:
            sel = [l for l in sel if l in layers]
        for l in sel:
            hr = (norms or {}).get(l)
            vec = readout.steer_vector(l, tid, alpha, h_rms=hr)
            out[l] = out.get(l, np.zeros_like(vec)) + vec
    return out


def export_control_vector(readout, specs: list[dict], out_path: str, *,
                          layers: list[int] | None = None,
                          norms: dict[int, float] | None = None,
                          model_hint: str = "") -> str:
    """Write a llama.cpp control-vector GGUF (``direction.{layer}`` F32, 1-based).

    llama.cpp indexes control layers 1..n_layer (layer 0 is unused), so a lens
    residual layer ``l`` is written as ``direction.{l+1}``.
    """
    import gguf

    directions = build_control_vector(readout, specs, layers=layers, norms=norms)
    if not directions:
        raise ValueError("no steer specs produced any direction")
    writer = gguf.GGUFWriter(out_path, arch="controlvector")
    writer.add_string("controlvector.model_hint", model_hint or readout.weights.arch)
    writer.add_uint32("controlvector.layer_count", readout.weights.n_readable_layers)
    for l, vec in sorted(directions.items()):
        writer.add_tensor(f"direction.{l + 1}", np.ascontiguousarray(vec, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("wrote control vector with %d layer directions -> %s", len(directions), out_path)
    return out_path


# ----------------------------------------------------- abliteration -------- #


def abliterate_matrix(W: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Remove the ability to WRITE along ``directions`` from a writer matrix.

    ``W`` maps some hidden space to the residual stream ``[d_model, k]`` (columns
    are what gets added to the residual). Projecting each column off the span of
    the (orthonormalized) directions kills the model's ability to emit those
    concepts. ``directions``: ``[d_model, r]`` (unit-ish); returns W'.
    """
    Q, _ = np.linalg.qr(directions)            # orthonormal basis [d_model, r]
    # W' = (I - Q Qᵀ) W  — remove the direction-span component of each column
    return (W - Q @ (Q.T @ W)).astype(W.dtype, copy=False)


def concept_directions(readout, token_ids: list[int]) -> np.ndarray:
    """Final-layer-space unit directions for the given tokens (abliteration
    works in the residual/output space the unembedding reads)."""
    cols = []
    gamma = readout._wu_gamma  # [vocab, d]
    for t in token_ids:
        v = gamma[int(t)].astype(np.float32)
        n = np.linalg.norm(v)
        cols.append(v / n if n > 0 else v)
    return np.stack(cols, axis=1)  # [d_model, r]


def _copy_gguf_metadata(reader, writer) -> None:
    """Copy every real KV field from a reader to a writer, preserving types
    (including arrays), so an abliterated model stays loadable (tokenizer,
    chat template, hparams all survive)."""
    import gguf

    GT = gguf.GGUFValueType
    for key, field in reader.fields.items():
        if not field.types or key.startswith("general.architecture") or key.startswith("GGUF."):
            continue
        vtype = field.types[0]
        try:
            if vtype == GT.ARRAY:
                elem = field.types[1]
                if elem == GT.STRING:
                    vals = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data]
                else:
                    vals = [field.parts[i][0].item() if hasattr(field.parts[i][0], "item")
                            else field.parts[i][0] for i in field.data]
                writer.add_array(key, vals)
            elif vtype == GT.STRING:
                writer.add_string(key, str(field.contents()))
            else:
                writer.add_key_value(key, field.contents(), vtype)
        except Exception as e:  # noqa: BLE001
            logger.debug("skip metadata %s: %s", key, e)


def export_abliterated(model_gguf: str, readout, token_ids: list[int], out_path: str, *,
                       tensors=("attn_output.weight", "ffn_down.weight", "token_embd.weight"),
                       progress: bool = True) -> str:
    """Write a new GGUF with writer matrices orthogonalized against the concept.

    Reads every tensor from ``model_gguf``; for tensors whose name contains one
    of ``tensors`` (the residual writers), removes the concept span; copies the
    rest unchanged. Quantized writers are dequantized→edited→re-quantized to F16
    for the affected tensors (others keep their original quant).
    """
    import gguf
    from gguf.quants import dequantize

    directions = concept_directions(readout, token_ids)  # [d_model, r]
    reader = gguf.GGUFReader(model_gguf)
    arch = str(reader.fields["general.architecture"].contents())
    writer = gguf.GGUFWriter(out_path, arch=arch)

    # copy kv metadata (skip gguf-internal fields the writer manages itself)
    _copy_gguf_metadata(reader, writer)

    d = readout.weights.d_model
    edited = 0
    for t in reader.tensors:
        name = t.name
        shape = [int(x) for x in t.shape]
        is_writer = any(tok in name for tok in tensors)
        if is_writer and shape and (d in shape):
            if t.tensor_type in (gguf.GGMLQuantizationType.F32,):
                W = np.asarray(t.data, dtype=np.float32)
            elif t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
                W = np.asarray(t.data).astype(np.float32)
            else:
                W = dequantize(t.data, t.tensor_type).astype(np.float32)
            W = W.reshape(shape[::-1]) if len(shape) == 2 else W
            # orient so d_model is axis 0 (columns = residual writes)
            if W.ndim == 2 and W.shape[0] != d and W.shape[1] == d:
                Wp = abliterate_matrix(W.T, directions).T
            elif W.ndim == 2 and W.shape[0] == d:
                Wp = abliterate_matrix(W, directions)
            else:
                Wp = W  # 1D or unexpected — leave it
            writer.add_tensor(name, np.ascontiguousarray(Wp, dtype=np.float16))
            edited += 1
            if progress:
                logger.info("abliterated %s %s", name, W.shape)
        else:
            writer.add_tensor(name, t.data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("wrote abliterated model (%d writer tensors edited) -> %s", edited, out_path)
    return out_path
