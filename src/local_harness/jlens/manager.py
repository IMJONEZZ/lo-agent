"""Sidecar lifecycle + model-file resolution for `lo lens`.

The C++ activation sidecar (`jlens-server`) is NOT shipped in the wheel; it is
cloned + built on the model box on demand (capability-ladder pattern). This
module: finds/builds the binary (with the toolchain fixes the spike surfaced),
spawns/health-checks it, resolves which GGUF a given provider is serving, and
refuses a model/sidecar mismatch.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

JLENS_REPO = "https://github.com/igorbarshteyn/jlens-gguf.git"
JLENS_PIN = os.environ.get("LO_JLENS_COMMIT", "")  # empty = default branch
LO_JLENS_HOME = Path(os.environ.get("LO_JLENS_HOME", Path.home() / ".lo" / "jlens"))


# ---------------------------------------------------------------- binary --


def find_sidecar_bin(explicit: str | None = None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("LO_JLENS_BIN"),
        str(LO_JLENS_HOME / "jlens-gguf" / "native" / "jlens-server"),
        shutil.which("jlens-server"),
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def _pick_cxx() -> str | None:
    """A working C++ compiler. Order matches the spike: system g++, then a
    conda-forge/pixi cxx-compiler (self-contained sysroot, dodges glibc header
    clashes), then clang++."""
    for c in ("g++", "c++",
              str(Path.home() / ".pixi" / "envs" / "cxx-compiler" / "bin" / "g++"),
              "clang++"):
        if shutil.which(c) or Path(c).is_file():
            return c
    return None


def build_sidecar(*, llama_dir: str | None = None, jobs: int | None = None) -> str:
    """Clone (pinned) + build jlens-server under LO_JLENS_HOME. Returns the path.

    Applies the two link fixes the spike found vs upstream build.sh:
    ``-lggml-cpu`` and (on Linux) ``-lgomp``.
    """
    LO_JLENS_HOME.mkdir(parents=True, exist_ok=True)
    repo = LO_JLENS_HOME / "jlens-gguf"
    if not repo.exists():
        logger.info("cloning jlens-gguf into %s", repo)
        subprocess.run(["git", "clone", "--recursive", "--depth", "1", JLENS_REPO, str(repo)],
                       check=True)
        if JLENS_PIN:
            subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", JLENS_PIN], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", JLENS_PIN], check=True)
            subprocess.run(["git", "-C", str(repo), "submodule", "update", "--init", "--depth", "1"], check=True)

    cxx = _pick_cxx()
    if cxx is None:
        raise RuntimeError(
            "no C++ compiler found. Install one:\n"
            "  Fedora/RHEL:  sudo dnf install gcc-c++\n"
            "  Debian:       sudo apt install g++\n"
            "  macOS:        xcode-select --install\n"
            "  no-sudo:      pixi global install cxx-compiler")
    env = dict(os.environ, CXX=cxx, CC=cxx.replace("g++", "gcc").replace("clang++", "clang"))
    if llama_dir:
        env["LLAMA_DIR"] = llama_dir
    if jobs:
        env["JOBS"] = str(jobs)

    build = repo / "native" / "build.sh"
    logger.info("building jlens-server (CXX=%s) — first build takes a few minutes", cxx)
    r = subprocess.run(["bash", str(build)], env=env, cwd=str(repo),
                       capture_output=True, text=True)
    binp = repo / "native" / "jlens-server"
    if binp.is_file() and os.access(binp, os.X_OK):
        return str(binp)

    # upstream build.sh link step can miss ggml-cpu / gomp — retry with the fix
    logger.warning("build.sh did not produce the binary; retrying link with -lggml-cpu -lgomp")
    lib = repo / "llama.cpp" / "build" / "bin"
    httplib_o = repo / "native" / "httplib.o"
    if not httplib_o.exists():
        subprocess.run([cxx, "-O2", "-std=c++17", "-pthread",
                        "-I", str(repo / "native" / "vendor" / "cpp-httplib"),
                        "-c", str(repo / "native" / "vendor" / "cpp-httplib" / "httplib.cpp"),
                        "-o", str(httplib_o)], check=True)
    extra = ["-lggml-cpu"] + (["-lgomp"] if os.uname().sysname == "Linux" else [])
    link = [cxx, "-O2", "-std=c++17", "-pthread",
            "-I", str(repo / "llama.cpp" / "include"),
            "-I", str(repo / "llama.cpp" / "ggml" / "include"),
            "-I", str(repo / "native" / "vendor"),
            "-o", str(binp), str(repo / "native" / "jlens_server.cpp"), str(httplib_o),
            "-L", str(lib), "-lllama", "-lggml", *extra, "-lggml-base",
            f"-Wl,-rpath,{lib}"]
    res = subprocess.run(link, capture_output=True, text=True)
    if not binp.is_file():
        raise RuntimeError(f"jlens-server build failed:\n{r.stderr[-1500:]}\n{res.stderr[-1500:]}")
    return str(binp)


# ---------------------------------------------------------------- spawn --


def spawn_sidecar(bin_path: str, model: str, *, port: int, ctx_size: int = 4096,
                  chunk: int = 256, n_gpu_layers: int = 0, threads: int = 0,
                  wait_s: float = 600.0) -> subprocess.Popen:
    """Start jlens-server and wait for /health. Registers atexit terminate."""
    cmd = [bin_path, "-m", model, "--port", str(port), "-c", str(ctx_size), "--chunk", str(chunk)]
    if n_gpu_layers:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]
    if threads:
        cmd += ["-t", str(threads)]
    logger.info("starting sidecar: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    atexit.register(proc.terminate)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode(errors="replace")[-800:] if proc.stderr else ""
            raise RuntimeError(f"sidecar exited ({proc.returncode}):\n{err}")
        try:
            if httpx.get(url + "/health", timeout=2).json().get("status") == "ok":
                return proc
        except Exception:
            pass
        time.sleep(1.0)
    proc.terminate()
    raise RuntimeError(f"sidecar did not become healthy within {wait_s:.0f}s")


def sidecar_props(port: int) -> dict | None:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/props", timeout=5).json()
    except Exception:
        return None


def check_l_out_ok(props: dict, weights) -> tuple[bool, str]:
    """Interpret the sidecar's l_out self-check with MTP awareness.

    The sidecar's own check counts captures against its total layer count and
    can false-negative on MTP models (NextN layers emit no l_out). If the
    number of emitting layers matches our readable-layer count, capture works.
    """
    if props.get("l_out_ok"):
        return True, "sidecar self-check passed"
    sidecar_n = int(props.get("n_layer", 0))
    if sidecar_n == weights.n_readable_layers:
        return True, (f"sidecar reported l_out_ok=false but its {sidecar_n} base layers "
                      f"match the readable-layer count (MTP={weights.n_nextn}); capture works")
    return False, (f"architecture may not expose l_out (sidecar n_layer={sidecar_n}, "
                   f"readable={weights.n_readable_layers})")


# ------------------------------------------------------- GGUF resolution --


def model_from_llama_server(url: str) -> str | None:
    """The GGUF path a llama-server / jlens-server serves, from /props."""
    for path in ("/props", "/v1/props"):
        try:
            data = httpx.get(url.rstrip("/") + path, timeout=5).json()
        except Exception:
            continue
        mp = data.get("model_path") or (
            data.get("default_generation_settings", {}) or {}).get("model", {}).get("path")
        if mp and mp != "none" and os.path.exists(mp):
            return mp
    return None


def _scan(dirs, needle: str) -> str | None:
    needle = needle.lower()
    for d in dirs:
        p = Path(d).expanduser()
        if not p.is_dir():
            continue
        cands = sorted(g for g in p.rglob("*.gguf")
                       if needle in g.name.lower() and "mmproj" not in g.name.lower())
        # multi-shard: take the first shard
        firsts = [c for c in cands if "00001-of-" in c.name] or cands
        if firsts:
            return str(firsts[0])
    return None


def resolve_model_gguf(*, model: str | None = None, llama_server: str | None = None,
                       model_name: str | None = None) -> str | None:
    """Best-effort GGUF path from an explicit path, a running server, or a
    provider's on-disk layout (LM Studio / Ollama)."""
    if model and os.path.exists(model):
        return model
    if llama_server:
        m = model_from_llama_server(llama_server)
        if m:
            return m
    if model_name:
        # LM Studio hides the path in its API; scan the conventional stores.
        m = _scan(["~/.lmstudio/models", "~/.cache/lm-studio/models"], model_name)
        if m:
            return m
    return None


def same_gguf(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return Path(a).resolve() == Path(b).resolve()


# ------------------------------------------- LD_PRELOAD shim (Phase 6) ----- #

_SHIM_SRC = Path(__file__).parent / "native" / "lo_jlens_shim.cpp"


def build_shim(*, llama_dir: str, llama_build: str | None = None,
               out: str | None = None) -> str:
    """Compile the LD_PRELOAD steering shim against a llama.cpp checkout.

    Links only libllama/libggml-base public API (like the sidecar). Returns the
    .so path. `llama_build` defaults to <llama_dir>/build/bin.
    """
    import subprocess

    lib = llama_build or str(Path(llama_dir) / "build" / "bin")
    out = out or str(LO_JLENS_HOME / "lo_jlens_shim.so")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cxx = _pick_cxx()
    if cxx is None:
        raise RuntimeError("no C++ compiler (see `lo lens up` guidance)")
    cmd = [cxx, "-O2", "-std=c++17", "-fPIC", "-shared",
           "-I", str(Path(llama_dir) / "include"),
           "-I", str(Path(llama_dir) / "ggml" / "include"),
           "-o", out, str(_SHIM_SRC),
           "-L", lib, "-lllama", "-lggml-base", "-ldl",
           f"-Wl,-rpath,{lib}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not Path(out).is_file():
        raise RuntimeError(f"shim build failed:\n{r.stderr[-1500:]}")
    return out


def write_steer_file(edits: list[dict], path: str) -> str:
    """Write a steer.bin the shim reads live. Each edit: {layer, pos_start,
    pos_end, vector:[float]}. Magic LJS1 | u32 n | (i32 layer,ps,pe,d, d*f32)."""
    import struct

    import numpy as np

    with open(path, "wb") as f:
        f.write(b"LJS1")
        f.write(struct.pack("<I", len(edits)))
        for e in edits:
            vec = np.ascontiguousarray(e["vector"], dtype="<f4")
            f.write(struct.pack("<iiii", int(e["layer"]), int(e.get("pos_start", 0)),
                                int(e.get("pos_end", -1)), int(vec.size)))
            f.write(vec.tobytes())
    return path


def detect_llama_linkage(binary: str) -> tuple[bool, str]:
    """Is a llama-server binary dynamically linked to libllama (shim-able)?"""
    import subprocess

    if not Path(binary).is_file():
        return False, f"{binary} not found"
    try:
        out = subprocess.run(["ldd", binary], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return False, "ldd unavailable"
    if "libllama" in out:
        return True, "dynamically linked to libllama — LD_PRELOAD shim applies"
    if "not a dynamic executable" in out:
        return False, "statically linked — rebuild llama.cpp with -DBUILD_SHARED_LIBS=ON to shim it"
    return False, "libllama not in ldd output (static build?) — use the sidecar or exports instead"
