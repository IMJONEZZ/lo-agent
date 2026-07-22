"""`lo lens ...` — bring up / diagnose / fit / inspect the lens service.

Box-side commands (`up`, `fit`) run where the model file lives. Client-side
commands (`doctor`, `status`) talk to a configured lens_url. See
docs/jlens-integration.md and the plan.
"""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger("lo.lens")

DEFAULT_SIDECAR_PORT = 8091
DEFAULT_SERVICE_PORT = 8092


def _warn_remote_load(target: str | None) -> None:
    if target and not any(h in target for h in ("127.0.0.1", "localhost", "0.0.0.0")):
        logger.warning(
            "target %s is not localhost — capture/fit generates real load on that box. "
            "Confirm it is free before proceeding (shared-fleet rule).", target)


def cmd_up(args) -> None:
    """Build (if needed) + start the sidecar and lens service on THIS box."""
    from local_harness.jlens import manager
    from local_harness.jlens.service import LensService, create_app

    model = manager.resolve_model_gguf(model=args.model, llama_server=args.llama_server,
                                       model_name=args.model_name)
    if not model:
        sys.exit("error: could not resolve a local GGUF. Pass --model PATH, or "
                 "--llama-server URL to read it from a running server's /props.")
    logger.info("model: %s", model)

    binp = manager.find_sidecar_bin(args.sidecar_bin)
    if binp is None:
        if args.no_build:
            sys.exit("error: no jlens-server binary and --no-build set. Run `lo lens up` "
                     "without --no-build, or set LO_JLENS_BIN.")
        binp = manager.build_sidecar(llama_dir=args.llama_dir, jobs=args.jobs)
    logger.info("sidecar binary: %s", binp)

    # reuse a running sidecar on the port, else spawn one
    proc = None  # set only if WE spawned it — we never reap someone else's
    props = manager.sidecar_props(args.sidecar_port)
    if props is None:
        proc = manager.spawn_sidecar(binp, model, port=args.sidecar_port,
                                     ctx_size=args.ctx_size, chunk=args.chunk,
                                     n_gpu_layers=args.n_gpu_layers, threads=args.threads)
        props = manager.sidecar_props(args.sidecar_port)
    else:
        served = props.get("model_path", "")
        if served and not manager.same_gguf(served, model):
            sys.exit(f"error: sidecar on :{args.sidecar_port} serves {served!r}, not {model!r}. "
                     "Stop it or pass a matching --model.")

    from local_harness.jlens.model_reader import ReadoutWeights
    weights = ReadoutWeights.from_gguf(model)
    ok, why = manager.check_l_out_ok(props, weights)
    (logger.info if ok else logger.error)("l_out check: %s", why)
    if not ok:
        sys.exit("error: this model/arch does not expose a readable residual stream.")

    lens = args.lens or _auto_lens(model)
    if lens:
        logger.info("lens: %s", lens)
    service = LensService(model_path=model, native_url=f"http://127.0.0.1:{args.sidecar_port}",
                          lens_path=lens)
    app = create_app(service)

    import uvicorn
    url = f"http://{args.host}:{args.service_port}"

    # Record what we're running so `lo lens down` can reap it from another
    # shell — including after a SIGKILL, when no atexit hook of ours survives.
    manager.write_state(service_pid=os.getpid(), service_host=args.host,
                        service_port=args.service_port,
                        sidecar_pid=(proc.pid if proc else None),
                        sidecar_port=args.sidecar_port,
                        sidecar_owned=proc is not None,
                        model=model, started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    print(f"\n  ── lo lens service ready: {url} ──")
    print(f"  point a client at it:  lo config set lens_url {url}")
    print("  take it down:          lo lens down\n", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.service_port, log_level="warning")
    finally:
        # Runs on Ctrl-C and on uvicorn's SIGTERM handling alike. Only the
        # sidecar we spawned is ours to stop; a reused one keeps running.
        if proc is not None:
            logger.info("stopping sidecar (pid %d) ...", proc.pid)
            if not manager.stop_sidecar(proc):
                logger.error("sidecar pid %d survived SIGKILL — stop it by hand", proc.pid)
        manager.clear_state()


def _auto_lens(model: str) -> str | None:
    from pathlib import Path
    m = Path(model)
    for c in (m.with_suffix(".jlens.gguf"), m.parent / f"{m.stem}-jlens.gguf",
              Path("lenses") / f"{m.stem}.gguf"):
        if c.is_file():
            return str(c)
    return None


def cmd_down(args) -> None:
    """Stop the lens service + sidecar started by `lo lens up` on THIS box.

    Prefers the run-state file written by `up`; falls back to whoever is
    listening on the ports, so an orphan from a crashed/killed run (or from a
    build predating the state file) is still reapable.
    """
    from local_harness.jlens import manager

    state = manager.read_state()
    service_port = args.service_port or state.get("service_port") or DEFAULT_SERVICE_PORT
    sidecar_port = args.sidecar_port or state.get("sidecar_port") or DEFAULT_SIDECAR_PORT

    # Service first: its own cleanup may take the sidecar with it.
    targets, skipped = [], False
    svc_pid = state.get("service_pid") or manager.pid_on_port(service_port)
    if svc_pid and manager.pid_alive(svc_pid):
        targets.append(("lens service", svc_pid, service_port))

    side_pid = state.get("sidecar_pid") or manager.pid_on_port(sidecar_port)
    if side_pid and manager.pid_alive(side_pid):
        # Refuse to reap a sidecar `up` explicitly reused, unless asked.
        if state and state.get("sidecar_owned") is False and not args.force:
            print(f"  ! sidecar on :{sidecar_port} (pid {side_pid}) was not started by "
                  f"`lo lens up` — leaving it alone (--force to stop it anyway)")
            skipped = True
        else:
            targets.append(("sidecar", side_pid, sidecar_port))

    if not targets:
        if not skipped:
            print(f"nothing to stop (no lens service on :{service_port}, "
                  f"no sidecar on :{sidecar_port})")
        manager.clear_state()
        return

    failed = []
    for label, pid, port in targets:
        print(f"  stopping {label} (pid {pid}, :{port}) ...", end=" ", flush=True)
        if manager.stop_pid(pid, timeout=args.timeout, label=label):
            print("stopped")
        else:
            print("FAILED")
            failed.append((label, pid))

    manager.clear_state()
    if failed:
        sys.exit("error: could not stop "
                 + ", ".join(f"{label} (pid {pid})" for label, pid in failed)
                 + " — you may need `sudo kill -9`, or it belongs to another user.")
    print("\n  lens is down.")


def cmd_doctor(args) -> None:
    """Client-side: explain what to run where; check a configured lens_url."""
    import httpx
    mark = lambda b: "✓" if b else "✗"  # noqa: E731
    lens_url = args.lens_url
    print("lo lens — readiness\n")
    print("  The lens needs a sidecar + lens service running ON THE MODEL BOX")
    print("  (activations can't be read over a stock server's API). On that box:")
    print("     lo lens up --llama-server http://127.0.0.1:8080")
    print("  and to stop it again (frees the model's RAM/VRAM and the ports):")
    print("     lo lens down\n")
    if not lens_url:
        print(f"  {mark(False)} lens_url not configured — set it once the box-side service is up:")
        print("       lo config set lens_url http://<model-box>:8092")
        return
    try:
        h = httpx.get(lens_url.rstrip("/") + "/health", timeout=5).json()
        ok = h.get("status") == "ok"
        print(f"  {mark(ok)} lens service @ {lens_url}")
        if ok:
            lm = (h.get("lens") or {})
            print(f"       model={h.get('model_name','?')} arch={h.get('arch','?')} "
                  f"d_model={h.get('d_model')} layers={h.get('n_layers')} "
                  f"(MTP={h.get('n_nextn',0)})")
            print(f"       lens={lm.get('method','identity')} "
                  f"fitted_layers={len(lm.get('source_layers') or [])}")
    except Exception as e:  # noqa: BLE001
        print(f"  {mark(False)} lens service @ {lens_url} unreachable: {e}")


def cmd_status(args) -> None:
    import httpx
    if not args.lens_url:
        sys.exit("no lens_url configured (lo config set lens_url http://box:8092)")
    try:
        print(httpx.get(args.lens_url.rstrip("/") + "/lens/props", timeout=5).text)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"unreachable: {e}")


def cmd_fit(args) -> None:
    """Fit a lens ON THIS box — GGUF-native regression (default) or, with
    --hf, the exact causal lens on a torch/safetensors checkpoint (vLLM path)."""
    if getattr(args, "hf", None):
        return _cmd_fit_hf(args)
    import numpy as np

    from local_harness.jlens import manager
    from local_harness.jlens.fitting import fit_regression, load_corpus
    from local_harness.jlens.native_client import NativeClient
    from local_harness.jlens.model_reader import ReadoutWeights

    url = f"http://127.0.0.1:{args.sidecar_port}"
    _warn_remote_load(url)
    client = NativeClient(url)
    if not client.health():
        sys.exit(f"no sidecar on {url}. Start one: lo lens up --model {args.model or 'MODEL.gguf'}")

    # MTP-aware final layer
    props = client.props()
    weights = ReadoutWeights.from_gguf(args.model) if args.model else None
    target = args.target_layer
    if target is None and weights is not None:
        target = weights.final_layer

    prompts = load_corpus(args.corpus, n_prompts=args.n_prompts)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    logger.info("fitting on %d prompts (target_layer=%s) ...", len(prompts), target)
    lens = fit_regression(client, prompts, source_layers=layers, target_layer=target,
                          max_seq_len=args.max_seq_len, ridge=args.ridge,
                          gram_dtype=np.float32 if args.gram_dtype == "float32" else np.float64,
                          base_model=args.model or "")
    lens.save(args.output)
    print(f"saved {lens!r} -> {args.output}")


def _cmd_fit_hf(args) -> None:
    """Exact causal lens on an HF/safetensors checkpoint (needs the native extra)."""
    try:
        import transformers
    except ImportError:
        sys.exit("--hf needs the native extra: uv sync --extra native")
    from local_harness.jlens.fit_torch import fit_causal_lens
    from local_harness.jlens.fitting import load_corpus

    logger.info("loading %s (torch) ...", args.hf)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.hf)
    tok = transformers.AutoTokenizer.from_pretrained(args.hf)
    prompts = load_corpus(args.corpus, n_prompts=args.n_prompts)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    lens = fit_causal_lens(model, tok, prompts, source_layers=layers,
                           target_layer=args.target_layer, max_seq_len=args.max_seq_len,
                           device=args.device)
    lens.save(args.output)
    print(f"saved exact causal lens {lens!r} -> {args.output}")


def cmd_inspect(args) -> None:
    from local_harness.jlens.lens import JacobianLensGGUF
    lens = JacobianLensGGUF.load(args.path)
    print(lens)
    print(f"  target_layer: {lens.target_layer}")
    print(f"  method:       {lens.fit_method}  n_prompts={lens.n_prompts}")
    print(f"  biases:       {'yes' if lens.biases else 'no'}")
    if lens.h_rms:
        vals = ", ".join(f"L{l}={v:.1f}" for l, v in sorted(lens.h_rms.items())[:8])
        print(f"  h_rms:        {vals}{' ...' if len(lens.h_rms) > 8 else ''}")


def cmd_gen(args) -> None:
    """One-shot A/B: generate a continuation with an intervention vs baseline.

    Shows, not tells — the two continuations print side by side so the concept
    edit's effect is visible. steer/ablate a token, or swap two.
    """
    import httpx

    url = args.lens_url
    if not url:
        sys.exit("no lens_url (lo config set lens_url http://box:8092)")

    def _resolve(piece):
        res = httpx.get(url.rstrip("/") + "/lens/search_tokens",
                        params={"q": piece.strip(), "limit": 20}, timeout=30).json()["results"]
        exact = [r for r in res if r["piece"] == piece] or [r for r in res if r["piece"].strip() == piece.strip()]
        if not exact:
            sys.exit(f"no token matches {piece!r} (try a leading space, e.g. ' yen')")
        return exact[0]["token"], exact[0]["piece"]

    ivs = []
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    if args.steer:
        tid, p = _resolve(args.steer)
        ivs.append({"type": "steer", "token_id": tid, "alpha": args.alpha, "layers": layers})
        print(f"steer {p!r} α={args.alpha}")
    if args.ablate:
        tid, p = _resolve(args.ablate)
        ivs.append({"type": "ablate", "token_id": tid, "layers": layers})
        print(f"ablate {p!r}")
    if args.swap:
        a, b = args.swap.split("/", 1)
        ta, pa = _resolve(a)
        tb, pb = _resolve(b)
        ivs.append({"type": "swap", "token_a": ta, "token_b": tb, "layers": layers})
        print(f"swap {pa!r} ⇄ {pb!r}")
    if not ivs:
        sys.exit("pass at least one of --steer/--ablate/--swap")

    out = httpx.post(url.rstrip("/") + "/lens/generate", json={
        "prompt": args.prompt, "n_predict": args.n_predict,
        "interventions": ivs, "compare": True}, timeout=1200).json()
    print("\n  baseline:", repr(out.get("baseline", {}).get("text", "")))
    print("  steered: ", repr(out["steered"]["text"]))


def cmd_export(args) -> None:
    """Export a J-space edit as a file the user's EXISTING server loads."""
    from local_harness.jlens.lens import JacobianLensGGUF
    from local_harness.jlens.model_reader import ReadoutWeights
    from local_harness.jlens.readout import LensReadout
    from local_harness.jlens import exports

    weights = ReadoutWeights.from_gguf(args.model)
    lens = JacobianLensGGUF.load(args.lens) if args.lens else JacobianLensGGUF.identity(
        d_model=weights.d_model, layers=list(range(weights.n_readable_layers - 1)))
    readout = LensReadout(weights, lens)

    def _resolve(piece):
        from local_harness.jlens.native_client import NativeClient
        # resolve via the sidecar if present, else fail with guidance
        import httpx
        try:
            res = httpx.get(f"http://127.0.0.1:{args.sidecar_port}/vocab", timeout=30).json()
        except Exception:
            sys.exit("token resolution needs the sidecar (lo lens up) or numeric #id")
        pieces = res["pieces"]
        for tid, p in enumerate(pieces):
            pp = p["b64"] if isinstance(p, dict) else p
            if pp == piece or (isinstance(pp, str) and pp.strip() == piece.strip()):
                return tid
        sys.exit(f"no token {piece!r}")

    def _tid(tok):
        return int(tok[1:]) if tok.startswith("#") else _resolve(tok)

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    if args.kind == "cvec":
        specs = [{"type": "steer", "token_id": _tid(args.steer), "alpha": args.alpha,
                  "layers": layers}]
        exports.export_control_vector(readout, specs, args.output, model_hint=weights.arch)
        print(f"wrote control vector -> {args.output}")
        print(f"  use it on your stock llama-server:\n"
              f"    llama-server -m {args.model} --control-vector {args.output}"
              + (f" --control-vector-layer-range {layers[0]} {layers[1]}" if layers else ""))
    elif args.kind == "abliterate":
        tids = [_tid(t) for t in args.token.split(",")]
        exports.export_abliterated(args.model, readout, tids, args.output)
        print(f"wrote abliterated model -> {args.output}")
        print(f"  load it on any server (llama.cpp / LM Studio / Ollama).")


def cmd_shim(args) -> None:
    """Build the LD_PRELOAD steering shim for the user's OWN llama-server."""
    from local_harness.jlens import manager

    if not args.llama_dir:
        sys.exit("--llama-dir is required (the llama.cpp checkout your server is built from)")
    out = manager.build_shim(llama_dir=args.llama_dir, llama_build=args.llama_build,
                             out=args.output)
    print(f"built shim -> {out}\n")
    if args.server_bin:
        ok, why = manager.detect_llama_linkage(args.server_bin)
        print(f"  {'✓' if ok else '✗'} {args.server_bin}: {why}\n")
    print("  live-steer your own llama-server (one restart):")
    print(f"    LD_PRELOAD={out} LO_JLENS_STEER=/path/steer.bin \\")
    print("        llama-server -m model.gguf ...    # your usual flags")
    print("  write steer.bin with `lo lens export` directions (or the Python API).")


# subcommands whose imports need the lens extra (numpy/gguf); doctor/status/gen
# talk HTTP only and shim just compiles C++.
_NEEDS_LENS_EXTRA = {"up", "fit", "inspect", "export"}


def run(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.lens_action in _NEEDS_LENS_EXTRA:
        from local_harness.jlens import _EXTRA_MSG, missing_lens_deps

        missing = missing_lens_deps()
        if missing:
            sys.exit(f"error: `lo lens {args.lens_action}` needs "
                     f"{' and '.join(missing)} — {_EXTRA_MSG}")
    {"up": cmd_up, "down": cmd_down, "doctor": cmd_doctor, "status": cmd_status,
     "fit": cmd_fit, "inspect": cmd_inspect, "gen": cmd_gen, "export": cmd_export,
     "shim": cmd_shim}[args.lens_action](args)
