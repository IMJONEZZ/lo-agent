"""A hermetic mock jlens-server for tests — no llama.cpp, no real model.

Speaks the JLNS v1 HTTP protocol over a real socket (ThreadingHTTPServer) so
the vendored NativeClient and the lens service can be exercised end to end.
The "model" is a fixed, seeded toy: d_model=D, n_layers=L, vocab=V, with
deterministic per-token/per-layer residuals so tests can assert exact bytes.
"""

from __future__ import annotations

import base64
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

D = 16
L = 4          # emitting layers 0..3; final readable = 3
V = 48
N_NEXTN = 0


def _rng(*seed) -> np.random.Generator:
    return np.random.default_rng(abs(hash(seed)) % (2**32))


class MockModel:
    """Deterministic toy: residual[layer, token] is a fixed function of ids."""

    def __init__(self, d=D, n_layers=L, vocab=V):
        self.d, self.n_layers, self.vocab = d, n_layers, vocab
        # a fixed unembedding so readouts are reproducible
        self.w_unembed = _rng("unembed").standard_normal((vocab, d)).astype(np.float32) * 0.1
        self.norm_weight = np.ones(d, np.float32)

    def residual(self, tokens: list[int], layer: int) -> np.ndarray:
        out = np.empty((len(tokens), self.d), np.float32)
        for i, t in enumerate(tokens):
            v = _rng("resid", t, layer).standard_normal(self.d).astype(np.float32)
            out[i] = v * (1.0 + layer)  # norm grows with depth, like real models
        return out

    def logits(self, h: np.ndarray) -> np.ndarray:
        # final norm (rms) + head
        scale = 1.0 / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-5)
        return (h * scale * self.norm_weight) @ self.w_unembed.T

    def piece(self, t: int) -> str:
        return f"tok{t}"


class _Handler(BaseHTTPRequestHandler):
    model: MockModel = None
    live_ivs: list = []
    live_meta: dict = {}
    last_completion: dict = {}

    def log_message(self, *a):  # silence
        pass

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        m = self.model
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/props":
            return self._json({
                "model_path": "/mock/toy.gguf", "model_desc": "mock toy",
                "n_layer": m.n_layers, "n_embd": m.d, "n_vocab": m.vocab,
                "n_ctx": 512, "chunk": 128, "l_out_ok": True,
                "bos": 1, "eos": 2, "add_bos": False, "has_chat_template": False,
            })
        if self.path == "/vocab":
            return self._json({"n_vocab": m.vocab,
                               "pieces": [m.piece(t) for t in range(m.vocab)],
                               "attrs": [0] * m.vocab})
        if self.path == "/jlens/interventions":
            return self._json({"count": len(self.__class__.live_ivs),
                               "meta": self.__class__.live_meta})
        if self.path == "/jlens/last_completion":
            return self._json(self.__class__.last_completion or {"id": 0})
        return self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        if self.path == "/jlens/interventions":
            self.__class__.live_ivs = []
            self.__class__.live_meta = {}
            return self._json({"count": 0})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        m = self.model
        body = self._body()
        if self.path == "/tokenize":
            # deterministic: words -> ids by hash, prepend bos-ish
            content = body.get("content", "")
            toks = [1] + [3 + (abs(hash(w)) % (m.vocab - 3)) for w in content.split()]
            if body.get("add_special", True) is False:
                toks = toks[1:]
            return self._json({"tokens": toks, "pieces": [m.piece(t) for t in toks]})
        if self.path == "/detokenize":
            return self._json({"content": " ".join(m.piece(t) for t in body["tokens"])})
        if self.path == "/apply_template":
            text = " ".join(msg["content"] for msg in body.get("messages", []))
            return self._json({"prompt": text})
        if self.path == "/jlens/interventions":
            self.__class__.live_ivs = body.get("interventions", [])
            self.__class__.live_meta = body.get("meta", {})
            return self._json({"count": len(self.__class__.live_ivs)})
        if self.path == "/jlens/forward":
            return self._forward(body)
        return self._json({"error": "not found"}, 404)

    def _forward(self, body: dict):
        m = self.model
        tokens = [int(t) for t in body["tokens"]]
        capture_layers = body.get("capture_layers")
        if capture_layers is None:
            capture_layers = list(range(m.n_layers)) if body.get("capture", True) else []
        dtype = body.get("dtype", "f16")
        np_dt = np.float16 if dtype == "f16" else np.float32
        n_predict = int(body.get("n_predict", 0))

        # apply any 'add'/'set' interventions to captured residuals (enough for
        # tests to see an effect; lowrank left as identity in the mock)
        ivs = body.get("interventions", [])

        def resid(layer):
            h = m.residual(tokens, layer).copy()
            for iv in ivs:
                if iv["layer"] != layer:
                    continue
                p0 = iv.get("pos_start", 0)
                p1 = iv.get("pos_end", -1)
                p1 = len(tokens) if p1 < 0 else min(p1, len(tokens))
                if iv["mode"] in ("add", "set"):
                    vec = np.frombuffer(base64.b64decode(iv["data"]), np.float32)
                    if iv["mode"] == "add":
                        h[p0:p1] += vec
                    else:
                        h[p0:p1] = vec
            return h

        activations = {l: resid(l) for l in capture_layers}

        # greedy "generation": argmax of the final-layer readout, appended.
        # add/set edits that cover the current (final) position are applied to
        # the final-layer residual so interventions can change generated tokens.
        def _edit_final(h_row, pos):
            for iv in ivs:
                if iv["layer"] != m.n_layers - 1:
                    continue
                p1 = iv.get("pos_end", -1)
                if not (iv.get("pos_start", 0) <= pos and (p1 < 0 or pos < p1)):
                    continue
                if iv["mode"] in ("add", "set"):
                    vec = np.frombuffer(base64.b64decode(iv["data"]), np.float32)
                    h_row = h_row + vec if iv["mode"] == "add" else vec.copy()
            return h_row

        generated = []
        cur = list(tokens)
        for _ in range(n_predict):
            pos = len(cur)
            h = _edit_final(m.residual(cur, m.n_layers - 1)[-1], pos)[None, :]
            nxt = int(m.logits(h)[0].argmax())
            generated.append({"token": nxt, "piece": m.piece(nxt)})
            cur.append(nxt)

        # assemble JLNS frame
        payload = bytearray()
        act_hdr = []
        for l, h in activations.items():
            arr = h.astype(np_dt)
            off = len(payload)
            payload += arr.tobytes()
            act_hdr.append({"layer": l, "dtype": dtype,
                            "shape": list(arr.shape), "offset": off,
                            "nbytes": arr.nbytes})
        header = {
            "tokens": tokens, "n_prompt": len(tokens), "n_gen": len(generated),
            "generated": generated, "activations": act_hdr, "logits": [],
            "timings": {"prompt_ms": 0.1},
        }
        if n_predict:
            self.__class__.last_completion = {
                "id": self.__class__.last_completion.get("id", 0) + 1,
                "tokens": cur, "n_prompt": len(tokens), "n_gen": len(generated),
                "text": "".join(g["piece"] for g in generated),
                "interventions_active": len(ivs),
            }
        hjson = json.dumps(header).encode()
        frame = b"JLNS" + struct.pack("<II", 1, len(hjson)) + hjson + bytes(payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)


class MockSidecar:
    """Context manager that runs the mock on a real localhost socket."""

    def __init__(self, model: MockModel | None = None):
        self.model = model or MockModel()

    def __enter__(self):
        # fresh subclass per instance so class-level live state doesn't leak
        handler = type("H", (_Handler,), {
            "model": self.model, "live_ivs": [], "live_meta": {}, "last_completion": {},
        })
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
