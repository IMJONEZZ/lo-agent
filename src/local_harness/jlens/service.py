"""The lens service: numpy lens math over a compact HTTP API.

Runs ON THE MODEL BOX next to the sidecar (heavy readout GEMMs + ~GB readout
weights stay near the model). lo clients speak this small JSON/base64 API
remotely. All lens math is numpy here; the sidecar only produces residual
activations and applies generic residual edits.

Adapted from jlens-gguf's browser bridge (Apache-2.0): browser/static-file
serving removed, MTP/NextN topology handled, Starlette front to match lo's
server/proxy conventions. See src/local_harness/jlens/NOTICE.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import OrderedDict

import numpy as np

from local_harness.jlens.model_reader import ReadoutWeights
from local_harness.jlens.native_client import NativeClient
from local_harness.jlens.readout import LensReadout, compute_grid

logger = logging.getLogger(__name__)


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


class LogitsCache:
    """LRU cache of per-(ctx, layer) lens logits, bounded by bytes."""

    def __init__(self, budget_bytes: int = 1 << 30) -> None:
        self.budget = budget_bytes
        self._data: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0

    def get(self, key):
        arr = self._data.get(key)
        if arr is not None:
            self._data.move_to_end(key)
        return arr

    def put(self, key, arr) -> None:
        if key in self._data:
            self._bytes -= self._data.pop(key).nbytes
        self._data[key] = arr
        self._bytes += arr.nbytes
        while self._bytes > self.budget and len(self._data) > 1:
            _, old = self._data.popitem(last=False)
            self._bytes -= old.nbytes

    def drop_ctx(self, ctx_id: str) -> None:
        for key in [k for k in self._data if k[0] == ctx_id]:
            self._bytes -= self._data.pop(key).nbytes


class Context:
    def __init__(self, ctx_id, tokens, n_prompt, n_gen, activations, grid, use_lens, interventions):
        self.ctx_id = ctx_id
        self.tokens = tokens
        self.n_prompt = n_prompt
        self.n_gen = n_gen
        self.activations = activations
        self.grid = grid
        self.use_lens = use_lens
        self.interventions = interventions


class LensService:
    """Owns the readout weights, lens, sidecar client, and context cache."""

    def __init__(self, *, model_path, native_url, lens_path=None, top_n=10,
                 max_contexts=3, logits_cache_mb=1024):
        from local_harness.jlens.lens import JacobianLensGGUF

        self.client = NativeClient(native_url)
        self.props = self.client.props()
        self.weights = ReadoutWeights.from_gguf(model_path)
        # MTP fix: the readable layer count excludes NextN; the sidecar reports
        # its emitting-layer count in /props n_layer (base layers only).
        self.n_layers = self.weights.n_readable_layers
        sidecar_layers = int(self.props.get("n_layer", self.n_layers))
        if sidecar_layers != self.n_layers:
            logger.info("layer counts: sidecar=%d, readable(readout)=%d (MTP=%d) — using %d",
                        sidecar_layers, self.n_layers, self.weights.n_nextn, self.n_layers)

        if lens_path:
            self.lens = JacobianLensGGUF.load(lens_path)
        else:
            self.lens = JacobianLensGGUF.identity(
                d_model=self.weights.d_model,
                layers=list(range(self.n_layers - 1)),
                base_model=model_path,
            )
        self.lens_path = lens_path
        self.readout = LensReadout(self.weights, self.lens)

        self.vocab = self.client.vocab()
        self.vocab_attrs = self.client.vocab_attrs()
        self.vocab_lower = [p.lower() for p in self.vocab]

        self.top_n = top_n
        self.max_tokens = self.props["n_ctx"] - 8
        self.contexts: OrderedDict[str, Context] = OrderedDict()
        self.max_contexts = max_contexts
        self.logits_cache = LogitsCache(logits_cache_mb << 20)
        self.lock = threading.Lock()
        self._ctx_counter = 0

    # ------------------------------------------------------------------ #

    def pieces(self, tokens):
        return [self.vocab[t] if 0 <= t < len(self.vocab) else f"[{t}]" for t in tokens]

    def resolve_tokens(self, body):
        if body.get("tokens"):
            tokens = [int(t) for t in body["tokens"]]
        elif body.get("messages"):
            prompt = self.client.apply_template(body["messages"], add_assistant=True)
            if body.get("assistant_prefill"):
                prompt += body["assistant_prefill"]
            tokens = self.client.tokenize(prompt, add_special=True, parse_special=True)
        elif body.get("prompt") is not None:
            tokens = self.client.tokenize(
                body["prompt"],
                add_special=bool(body.get("add_special", True)),
                parse_special=bool(body.get("parse_special", True)),
            )
        else:
            raise ValueError("provide 'prompt', 'tokens', or 'messages'")
        if not tokens:
            raise ValueError("empty prompt")
        return tokens[: self.max_tokens]

    def readout_layers(self, stride=1):
        layers = self.readout.readout_layers(self.n_layers)
        if stride > 1:
            final = layers[-1]
            layers = layers[::stride]
            if final not in layers:
                layers.append(final)
        return layers

    def layer_norms(self, tokens, layers):
        norms = {l: self.lens.h_rms[l] for l in layers if l in self.lens.h_rms}
        missing = [l for l in layers if l not in norms]
        if not missing:
            return norms
        for ctx in reversed(self.contexts.values()):
            match = (tokens is None or ctx.tokens[: len(tokens)] == tokens
                     or tokens[: len(ctx.tokens)] == ctx.tokens)
            if match and ctx.grid is not None:
                for l in list(missing):
                    if l in ctx.grid.norms:
                        norms[l] = ctx.grid.norms[l]
                        missing.remove(l)
                break
        if missing:
            probe = tokens or self.client.tokenize("The quick brown fox jumps over the lazy dog.")
            fr = self.client.forward(probe, capture_layers=missing, dtype="f16")
            for l in missing:
                norms[l] = float(np.median(np.linalg.norm(fr.activations[l], axis=-1)))
        return norms

    def translate_interventions(self, specs, tokens):
        """UI intervention specs → native add/lowrank edits.

        Position defaults follow the doctrine found live on qwen3.6:
        STEER defaults to prompt-positions-only (all-positions feeds back into
        each generated token → degenerate loops); ABLATE/SWAP default to
        all-positions (projection is self-limiting; prompt-only lets the
        concept re-form at the answer token).
        """
        if not specs:
            return []
        fitted = set(self.lens.source_layers)
        native = []
        n_prompt = len(tokens) if tokens else None

        def layer_range(spec):
            lr = spec.get("layers")
            if lr is None:
                return sorted(fitted)
            l0, l1 = int(lr[0]), int(lr[1])
            return [l for l in sorted(fitted) if l0 <= l <= l1]

        def pos_range(spec, kind):
            pos = spec.get("pos")
            if pos is not None:
                return int(pos[0]), int(pos[1])
            if kind == "steer" and n_prompt is not None:
                return 0, n_prompt      # prompt-only default
            return 0, -1                # ablate/swap: all positions

        steer_layers = sorted({l for s in specs if s.get("type") == "steer" for l in layer_range(s)})
        norms = self.layer_norms(tokens, steer_layers) if steer_layers else {}

        for spec in specs:
            kind = spec.get("type")
            p0, p1 = pos_range(spec, kind)
            if kind == "steer":
                tid = int(spec["token_id"])
                alpha = float(spec.get("alpha", 2.0))
                for l in layer_range(spec):
                    vec = self.readout.steer_vector(l, tid, alpha, h_rms=norms.get(l))
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "add", "vector": vec})
            elif kind == "ablate":
                tid = int(spec["token_id"])
                for l in layer_range(spec):
                    A, B = self.readout.ablate_factors(l, [tid])
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "lowrank", "a": A, "b": B})
            elif kind == "swap":
                ta, tb = int(spec["token_a"]), int(spec["token_b"])
                for l in layer_range(spec):
                    A, B = self.readout.swap_factors(l, ta, tb)
                    native.append({"layer": l, "pos_start": p0, "pos_end": p1, "mode": "lowrank", "a": A, "b": B})
            else:
                raise ValueError(f"unknown intervention type {kind!r}")
        return native

    def lens_logits_for(self, ctx, layer):
        key = (ctx.ctx_id, layer, ctx.use_lens)
        cached = self.logits_cache.get(key)
        if cached is not None:
            return cached
        logits = self.readout.lens_logits(ctx.activations[layer], layer, use_lens=ctx.use_lens)
        self.logits_cache.put(key, logits.astype(np.float32))
        return logits

    # ------------------------------------------------------------------ #
    # API handlers (return plain dicts; the Starlette layer JSON-encodes)
    # ------------------------------------------------------------------ #

    def api_props(self):
        return {
            "model_name": self.weights.model_name or "",
            "arch": self.weights.arch,
            "n_vocab": self.weights.n_vocab,
            "d_model": self.weights.d_model,
            "n_layers": self.n_layers,
            "n_nextn": self.weights.n_nextn,
            "final_layer": self.weights.final_layer,
            "lens": {
                "path": self.lens_path,
                "method": self.lens.fit_method,
                "n_prompts": self.lens.n_prompts,
                "source_layers": self.lens.source_layers,
                "has_bias": bool(self.lens.biases),
            },
            "top_n": self.top_n,
        }

    def api_slice(self, body):
        t0 = time.time()
        tokens = self.resolve_tokens(body)
        # Teacher-force a known completion (e.g. the answer the chat model
        # actually produced) after the prompt: the readouts then cover the real
        # output tokens, and n_prompt still marks the prompt/answer boundary.
        n_forced = 0
        if body.get("completion"):
            comp = self.client.tokenize(body["completion"],
                                        add_special=False, parse_special=True)
            comp = comp[: max(0, self.max_tokens - len(tokens))]
            n_forced = len(comp)
            tokens = tokens + comp
        use_lens = bool(body.get("use_lens", True))
        top_n = int(body.get("top_n", self.top_n))
        stride = max(1, int(body.get("stride", 1)))
        n_predict = int(body.get("n_predict", 0))
        sampling = body.get("sampling") or {"greedy": True}
        ui_specs = body.get("interventions") or []

        layers = self.readout_layers(stride)
        native_ivs = self.translate_interventions(ui_specs, tokens)
        fr = self.client.forward(tokens, capture_layers=layers, dtype="f16",
                                 interventions=native_ivs, n_predict=n_predict, sampling=sampling)
        t_fwd = time.time()
        self._ctx_counter += 1
        ctx_id = f"c{self._ctx_counter}"
        n_prompt = fr.n_prompt - n_forced  # the forced answer is not prompt
        ctx = Context(ctx_id, fr.tokens, n_prompt, fr.n_gen, fr.activations, None, use_lens, ui_specs)
        grid = compute_grid(self.readout, fr.activations, layers, top_n=top_n, use_lens=use_lens,
                            logits_fn=lambda layer: self.lens_logits_for(ctx, layer))
        ctx.grid = grid
        self.contexts[ctx_id] = ctx
        while len(self.contexts) > self.max_contexts:
            old_id, _ = self.contexts.popitem(last=False)
            self.logits_cache.drop_ctx(old_id)
        return {
            "ctx_id": ctx_id, "tokens": ctx.tokens, "pieces": self.pieces(ctx.tokens),
            "n_prompt": ctx.n_prompt, "n_gen": ctx.n_gen, "n_forced": n_forced,
            "generated_text": fr.generated_text,
            "layers": layers, "top_n": top_n, "use_lens": use_lens,
            "top_ids": _b64(grid.top_ids.astype("<i4")),
            "norms": {str(l): v for l, v in grid.norms.items()},
            "vocab_size": self.weights.n_vocab, "interventions": ui_specs,
            "timings": {"forward_ms": round((t_fwd - t0) * 1000, 1), **fr.timings},
        }

    def _ctx(self, body):
        ctx = self.contexts.get(str(body.get("ctx_id")))
        if ctx is None:
            raise ValueError("unknown or expired ctx_id; re-run the slice")
        return ctx

    def api_ranks(self, body):
        ctx = self._ctx(body)
        token_ids = np.asarray([int(t) for t in body["token_ids"]], dtype=np.int64)
        T, Lc = len(ctx.tokens), len(ctx.grid.layers)
        out = np.zeros((T, Lc, len(token_ids)), dtype=np.int32)
        for li, layer in enumerate(ctx.grid.layers):
            out[:, li, :] = LensReadout.ranks_of(self.lens_logits_for(ctx, layer), token_ids)
        return {"ctx_id": ctx.ctx_id, "token_ids": [int(t) for t in token_ids],
                "shape": [T, Lc, len(token_ids)], "ranks": _b64(out.astype("<i4"))}

    def api_readout(self, body):
        ctx = self._ctx(body)
        pos, layer = int(body["pos"]), int(body["layer"])
        top_n = int(body.get("top_n", 40))
        logits = self.lens_logits_for(ctx, layer)[pos]
        ids, vals = LensReadout.topk(logits[None, :], top_n)
        ids, vals = ids[0], vals[0]
        z = logits - logits.max()
        probs = np.exp(z); probs /= probs.sum()
        return {"pos": pos, "layer": layer, "tokens": [
            {"token": int(t), "piece": self.vocab[int(t)], "logit": float(v), "prob": float(probs[int(t)])}
            for t, v in zip(ids, vals)]}

    def api_decompose(self, body):
        ctx = self._ctx(body)
        pos, layer = int(body["pos"]), int(body["layer"])
        k = min(int(body.get("k", 12)), 25)
        if layer not in self.lens.jacobians and layer != self.n_layers - 1:
            raise ValueError(f"layer {layer} not fitted")
        items = self.readout.decompose(ctx.activations[layer][pos], layer, k=k)
        for item in items:
            item["piece"] = self.vocab[item["token"]]
        return {"pos": pos, "layer": layer, "items": items}

    def api_generate(self, body):
        if body.get("ctx_id"):
            tokens = self._ctx(body).tokens[: self._ctx(body).n_prompt]
        else:
            tokens = self.resolve_tokens(body)
        n_predict = int(body.get("n_predict", 32))
        sampling = body.get("sampling") or {"greedy": True}
        ui_specs = body.get("interventions") or []
        native_ivs = self.translate_interventions(ui_specs, tokens)
        fr = self.client.forward(tokens, capture=False, interventions=native_ivs,
                                 n_predict=n_predict, sampling=sampling)
        out = {"steered": {"text": fr.generated_text, "tokens": [g["token"] for g in fr.generated]}}
        if body.get("compare", True) and ui_specs:
            fr0 = self.client.forward(tokens, capture=False, n_predict=n_predict, sampling=sampling)
            out["baseline"] = {"text": fr0.generated_text, "tokens": [g["token"] for g in fr0.generated]}
        return out

    def api_search_tokens(self, query, limit=50):
        q = query.strip()
        results = []
        if q.startswith("#") and q[1:].isdigit():
            tid = int(q[1:])
            if 0 <= tid < len(self.vocab):
                results.append((0, tid))
        ql = q.lower()
        if ql:
            for tid, piece in enumerate(self.vocab_lower):
                s = piece.strip()
                if not s:
                    continue
                if s == ql:
                    results.append((1, tid))
                elif s.startswith(ql):
                    results.append((2, tid))
                elif ql in s:
                    results.append((3, tid))
        results.sort(key=lambda x: (x[0], len(self.vocab[x[1]]), x[1]))
        return {"results": [{"token": tid, "piece": self.vocab[tid]} for _, tid in results[:limit]]}

    # ---- live intervention set (backend mode passthrough) ---- #

    def api_live_push(self, body):
        specs = body.get("interventions") or []
        if not body.get("keep_positions", False):
            specs = [{**s, "pos": [0, -1]} for s in specs]
        native = self.translate_interventions(specs, tokens=None)
        out = self.client.live_interventions_set(native, meta={"ui_specs": specs})
        return {"count": out.get("count", 0), "specs": specs}

    def api_live_clear(self, body):
        return self.client.live_interventions_clear()


def create_app(service: LensService):
    """A Starlette app exposing the compact lens API (house pattern)."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    def _run(fn, body):
        with service.lock:
            return fn(body)

    async def health(request):
        return JSONResponse({"status": "ok", **service.api_props()})

    async def props(request):
        return JSONResponse(service.api_props())

    async def vocab(request):
        return JSONResponse({"pieces": service.vocab, "attrs": service.vocab_attrs})

    async def search_tokens(request):
        q = request.query_params.get("q", "")
        limit = int(request.query_params.get("limit", "50"))
        return JSONResponse(service.api_search_tokens(q, limit))

    def _post(handler, locked=True):
        async def route(request):
            body = await request.json() if await request.body() else {}
            try:
                result = _run(handler, body) if locked else handler(body)
                return JSONResponse(result)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            except Exception as e:  # noqa: BLE001
                logger.exception("lens API error")
                return JSONResponse({"error": str(e)}, status_code=500)
        return route

    routes = [
        Route("/health", health),
        Route("/lens/props", props),
        Route("/lens/vocab", vocab),
        Route("/lens/search_tokens", search_tokens),
        Route("/lens/slice", _post(service.api_slice), methods=["POST"]),
        Route("/lens/ranks", _post(service.api_ranks), methods=["POST"]),
        Route("/lens/readout", _post(service.api_readout), methods=["POST"]),
        Route("/lens/decompose", _post(service.api_decompose), methods=["POST"]),
        Route("/lens/generate", _post(service.api_generate), methods=["POST"]),
        Route("/lens/live/push", _post(service.api_live_push), methods=["POST"]),
        Route("/lens/live/clear", _post(service.api_live_clear), methods=["POST"]),
    ]
    return Starlette(routes=routes)
