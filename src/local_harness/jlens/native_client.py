# Vendored from igorbarshteyn/jlens-gguf (Apache-2.0) at pinned commit,
# ported to httpx and adapted for lo. See src/local_harness/jlens/NOTICE.
"""HTTP client for jlens-server (the llama.cpp activation sidecar).

Speaks the JLNS v1 binary protocol: /jlens/forward returns a "JLNS" framed
response (JSON header + raw f16/f32 activation blocks). Uses httpx to match
the rest of lo (the upstream reference used requests).
"""

from __future__ import annotations

import base64
import json
import struct
import threading
from dataclasses import dataclass, field

import httpx
import numpy as np


def _piece(p) -> str:
    """Decode a piece that may be {"b64": ...} (raw non-UTF8 bytes)."""
    if isinstance(p, dict):
        return base64.b64decode(p["b64"]).decode("utf-8", errors="replace")
    return p


@dataclass
class ForwardResult:
    """Parsed /jlens/forward response."""

    tokens: list[int]
    n_prompt: int
    n_gen: int
    generated: list[dict]                     # [{"token": id, "piece": str}]
    activations: dict[int, np.ndarray]        # layer -> [n_pos, d] float32
    logits: dict[int, np.ndarray]             # pos -> [n_vocab] float32
    timings: dict = field(default_factory=dict)

    @property
    def generated_text(self) -> str:
        return "".join(g["piece"] for g in self.generated)


class NativeClient:
    """Thin wrapper over jlens-server's HTTP API (httpx, thread-safe)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8091", timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # httpx.Client is not documented thread-safe for concurrent requests on
        # one connection; give each thread its own (the bridge calls this from
        # a ThreadingHTTPServer / asyncio executor).
        self._local = threading.local()
        self._vocab: list[str] | None = None
        self._vocab_attrs: list[int] | None = None

    @property
    def _client(self) -> httpx.Client:
        c = getattr(self._local, "client", None)
        if c is None:
            c = self._local.client = httpx.Client(timeout=self.timeout)
        return c

    # ------------------------------------------------------------------ #

    def _get(self, path: str) -> dict:
        r = self._client.get(self.base_url + path)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = self._client.post(self.base_url + path, json=body)
        if r.status_code != 200:
            try:
                raise RuntimeError(f"{path}: {r.json().get('error', r.text)}")
            except (ValueError, KeyError):
                raise RuntimeError(f"{path}: HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    # ------------------------------------------------------------------ #

    def health(self) -> bool:
        try:
            return self._get("/health").get("status") == "ok"
        except httpx.HTTPError:
            return False

    def props(self) -> dict:
        return self._get("/props")

    def vocab(self) -> list[str]:
        """All token pieces, decoded for display. Cached."""
        if self._vocab is None:
            data = self._get("/vocab")
            self._vocab = [_piece(p) for p in data["pieces"]]
            self._vocab_attrs = data.get("attrs")
        return self._vocab

    def vocab_attrs(self) -> list[int]:
        self.vocab()
        return self._vocab_attrs or []

    def tokenize(self, content: str, *, add_special: bool = True, parse_special: bool = True) -> list[int]:
        return self._post(
            "/tokenize",
            {"content": content, "add_special": add_special, "parse_special": parse_special},
        )["tokens"]

    def tokenize_with_pieces(self, content: str, **kw) -> tuple[list[int], list[str]]:
        out = self._post("/tokenize", {"content": content, **kw})
        return out["tokens"], [_piece(p) for p in out["pieces"]]

    def detokenize(self, tokens: list[int]) -> str:
        return _piece(self._post("/detokenize", {"tokens": tokens})["content"])

    def apply_template(self, messages: list[dict], *, add_assistant: bool = True) -> str:
        return self._post("/apply_template", {"messages": messages, "add_assistant": add_assistant})["prompt"]

    # ------------------------------------------------------------------ #
    # backend mode: live intervention set + last completion
    # ------------------------------------------------------------------ #

    def live_interventions_get(self) -> dict:
        return self._get("/jlens/interventions")

    def live_interventions_set(self, interventions: list[dict], *, meta=None) -> dict:
        return self._post(
            "/jlens/interventions",
            {"interventions": [self._encode_iv(iv) for iv in interventions],
             "meta": meta if meta is not None else {}},
        )

    def live_interventions_clear(self) -> dict:
        r = self._client.delete(self.base_url + "/jlens/interventions")
        r.raise_for_status()
        return r.json()

    def last_completion(self) -> dict:
        out = self._get("/jlens/last_completion")
        if "text" in out:
            out["text"] = _piece(out["text"])
        return out

    # ------------------------------------------------------------------ #

    def forward(
        self,
        tokens: list[int],
        *,
        capture_layers: list[int] | None = None,
        capture: bool = True,
        dtype: str = "f16",
        interventions: list[dict] | None = None,
        n_predict: int = 0,
        sampling: dict | None = None,
        logits_positions: list[int] | None = None,
    ) -> ForwardResult:
        """Run the model over ``tokens``; return activations and metadata."""
        body: dict = {"tokens": [int(t) for t in tokens], "dtype": dtype, "capture": capture}
        if capture_layers is not None:
            body["capture_layers"] = [int(l) for l in capture_layers]
        if interventions:
            body["interventions"] = [self._encode_iv(iv) for iv in interventions]
        if n_predict:
            body["n_predict"] = int(n_predict)
        if sampling:
            body["sampling"] = sampling
        if logits_positions:
            body["logits_positions"] = [int(p) for p in logits_positions]

        r = self._client.post(self.base_url + "/jlens/forward", json=body)
        if r.status_code != 200:
            try:
                raise RuntimeError(f"/jlens/forward: {r.json().get('error', r.text)}")
            except ValueError:
                raise RuntimeError(f"/jlens/forward: HTTP {r.status_code}")
        return self._parse_forward(r.content)

    @staticmethod
    def _encode_iv(iv: dict) -> dict:
        out = {
            "layer": int(iv["layer"]),
            "pos_start": int(iv.get("pos_start", 0)),
            "pos_end": int(iv.get("pos_end", -1)),
            "mode": iv.get("mode", "add"),
        }
        if out["mode"] in ("add", "set"):
            vec = np.ascontiguousarray(iv["vector"], dtype=np.float32)
            out["data"] = base64.b64encode(vec.tobytes()).decode()
        elif out["mode"] == "lowrank":
            a = np.ascontiguousarray(iv["a"], dtype=np.float32)
            b = np.ascontiguousarray(iv["b"], dtype=np.float32)
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                raise ValueError(f"lowrank shapes must be A[d,k], B[k,d]; got {a.shape}, {b.shape}")
            out["k"] = int(a.shape[1])
            out["data"] = base64.b64encode(a.tobytes() + b.tobytes()).decode()
        else:
            raise ValueError(f"unknown intervention mode {out['mode']!r}")
        return out

    @staticmethod
    def _parse_forward(buf: bytes) -> ForwardResult:
        if buf[:4] != b"JLNS":
            raise RuntimeError(f"bad response magic: {buf[:16]!r}")
        version, hlen = struct.unpack_from("<II", buf, 4)
        if version != 1:
            raise RuntimeError(f"unsupported response version {version}")
        header = json.loads(buf[12 : 12 + hlen])
        payload = memoryview(buf)[12 + hlen :]

        activations: dict[int, np.ndarray] = {}
        for a in header["activations"]:
            dt = np.float16 if a["dtype"] == "f16" else np.float32
            n_pos, d = a["shape"]
            arr = np.frombuffer(payload, dtype=dt, count=n_pos * d, offset=a["offset"])
            activations[a["layer"]] = arr.reshape(n_pos, d).astype(np.float32)

        logits: dict[int, np.ndarray] = {}
        for l in header["logits"]:
            logits[l["pos"]] = np.frombuffer(
                payload, dtype=np.float32, count=l["nbytes"] // 4, offset=l["offset"]
            ).copy()

        generated = [
            {"token": g["token"], "piece": _piece(g["piece"])} for g in header["generated"]
        ]
        return ForwardResult(
            tokens=header["tokens"],
            n_prompt=header["n_prompt"],
            n_gen=header["n_gen"],
            generated=generated,
            activations=activations,
            logits=logits,
            timings=header.get("timings", {}),
        )
