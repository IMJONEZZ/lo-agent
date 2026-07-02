"""CLI: lo probe | run | resume | replay | runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx

from ..agent.loop import Agent
from ..agent.tools import ToolRegistry, builtin_tools
from ..events.log import EventLog, MODEL_CALL, POLICY_TRIGGERED
from ..events.replay import replay_run
from ..inference.capabilities import probe
from ..inference.client import OpenAICompatClient
from ..skills.skill import SkillNotFound


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--url", default=os.environ.get("HARNESS_BASE_URL", "http://localhost:8080")
    )
    p.add_argument("--model", default=os.environ.get("HARNESS_MODEL", ""))
    p.add_argument("--db", default=os.environ.get("HARNESS_DB", "harness.db"))


async def _client(args) -> OpenAICompatClient:
    client = OpenAICompatClient(args.url, args.model)
    if not client.model:
        models = await client.list_models()
        if not models:
            print("error: no models on server and --model not given", file=sys.stderr)
            raise SystemExit(1)
        client.model = models[0]
    return client


async def cmd_probe(args) -> None:
    async with await _client(args) as client:
        caps = await probe(client)
        if args.json:
            print(json.dumps(caps.to_dict(), indent=2))
        else:
            print(caps.summary())


async def cmd_lora(args) -> None:
    async with await _client(args) as client:
        caps = await probe(client)
        if caps.lora_mode is None:
            print(f"LoRA hot-swap: not available on this server ({caps.server})")
            print(
                "  vLLM (with --enable-lora) and llama.cpp (--lora) support per-request "
                "adapters; native runs them in-process."
            )
            return
        print(f"LoRA hot-swap: {caps.lora_mode}")
        if caps.lora_mode == "llamacpp":
            if not caps.lora_adapters:
                print(
                    "  no adapters preloaded — start llama-server with --lora <file.gguf>"
                )
            for a in caps.lora_adapters:
                print(
                    f"  id {a.get('id')}  scale {a.get('scale', 1.0)}  {a.get('path', '')}"
                )
        elif caps.lora_mode == "vllm":
            print("  served models/adapters (request model=<name> to use one):")
            for m in await client.list_models():
                print(f"    {m}")
            print(
                "  load a new one at runtime via POST /v1/load_lora_adapter "
                "(needs VLLM_ALLOW_RUNTIME_LORA_UPDATING)"
            )


def cmd_sandbox(args) -> None:
    """Install / check the microVM sandbox that ships with the harness."""
    import shutil
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "install-sandbox.sh"
    if args.action == "install":
        if not script.exists():
            print(f"missing {script}")
            return
        raise SystemExit(subprocess.run(["bash", str(script)]).returncode)

    # doctor: report readiness without touching anything
    kvm = Path("/dev/kvm").exists()
    msb = shutil.which("msb")
    try:
        import microsandbox  # noqa: F401

        sdk = True
    except Exception:
        sdk = False
    mark = lambda b: "✓" if b else "✗"  # noqa: E731
    print("microVM sandbox readiness")
    print(
        f"  {mark(kvm)} KVM (/dev/kvm)        {'present' if kvm else 'MISSING — enable virtualization'}"
    )
    print(f"  {mark(bool(msb))} msb runtime          {msb or 'not installed'}")
    print(
        f"  {mark(sdk)} microsandbox SDK     {'importable' if sdk else 'not installed (uv sync --extra sandbox)'}"
    )
    ready = kvm and msb and sdk
    print(
        f"\n  {'READY — bash/file tools can run in a microVM' if ready else 'NOT READY — run: lo sandbox install'}"
    )


async def cmd_bench(args) -> None:
    from ..bench import run_bench, format_report

    async with await _client(args) as client:
        if not client.model:
            models = await client.list_models()
            client.model = models[0] if models else ""
        caps = await probe(client)
        results = await run_bench(
            client,
            caps,
            skills_dir=args.skills_dir,
            n=args.n,
            batch_invariance=not args.no_batch_invariance,
        )
        print(
            format_report(
                results, model=client.model, server=caps.server, tier=caps.tier()
            )
        )


def _guardrails_factory(args, tools: ToolRegistry):
    from ..guardrails.guardrails import Guardrails

    if args.no_guardrails:
        return None
    from ..agent.codemode import RUN_CODE_NAME
    from ..agent.tools import TOOL_SEARCH_NAME
    from ..server.coordinator import SEND_MESSAGE_NAME, SPAWN_AGENTS_NAME

    required = [s for s in (args.required_steps or "").split(",") if s]
    terminal = frozenset(s for s in (args.terminal_tool or "").split(",") if s)
    names = [s["function"]["name"] for s in tools.schemas()] + [
        TOOL_SEARCH_NAME,
        SPAWN_AGENTS_NAME,
        SEND_MESSAGE_NAME,
        RUN_CODE_NAME,
    ]
    return lambda: Guardrails(
        tool_names=names, required_steps=required, terminal_tools=terminal
    )


def _make_agent(args, client, caps, log, sandbox=None, on_compact=None) -> Agent:
    tools = ToolRegistry(builtin_tools(sandbox=sandbox))
    return Agent(
        client,
        tools,
        log,
        capabilities=caps,
        max_steps=args.max_steps,
        guardrails_factory=_guardrails_factory(args, tools),
        context_budget=args.context_budget,
        compact_fraction=getattr(args, "compact_fraction", 0.85),
        code_mode=getattr(args, "code_mode", True),
        sandbox=sandbox,
        on_compact=on_compact,
    )


_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _cli_status(log, run_id, caps, t0, *, tool=None, spin=None):
    """A status renderable for `lo run`: separator + the harness status bar
    (preset · tier · ctx% · $ saved · determinism), with an optional live
    spinner/elapsed/running-tool head."""
    import time

    from rich.console import Group
    from rich.text import Text

    from ..tui import render

    calls = log.events(run_id, type=MODEL_CALL)
    ctx = None
    if calls:
        body = calls[-1].payload.get("request_body") or {}
        used = render.context_used(body.get("messages") or [], body.get("tools"))
        w = caps.context_window
        ctx = (
            (f"{round(used / w * 100)}%", used / w) if w else (render._ktok(used), None)
        )
    tin = tout = 0
    for ev in calls:
        a, b = render.tokens_of(ev.payload)
        tin += a
        tout += b
    bar = render.status_bar(
        preset="run",
        tier=caps.tier(),
        glyphs=render.ability_glyphs(caps),
        saved=render.frontier_saved(tin, tout),
        deterministic=caps.seed,
        learn="off",
        ctx=ctx,
    )
    parts = [Text("─" * 46, style=render.C_DIM)]
    if spin is not None:
        head = Text.assemble(
            (_SPIN[spin % len(_SPIN)] + " ", render.C_RESAMPLE),
            (f"{time.time() - t0:.0f}s", render.C_DIM),
        )
        if tool:
            head.append(f"  ⚙ {tool}…", style=render.C_TOOL)
        parts.append(head)
    parts.append(bar)
    return Group(*parts)


async def cmd_run(args) -> None:
    import asyncio as aio
    import os
    import sys
    import time

    from ..sandbox import make_sandbox, SandboxUnavailable

    try:
        sandbox = make_sandbox(getattr(args, "sandbox", "host"), os.getcwd())
    except SandboxUnavailable as e:
        raise SystemExit(f"✗ {e}")
    if sandbox.kind != "host":
        print(
            f"⛨ tools run inside a {sandbox.kind} sandbox (workdir mounted, host isolated)"
        )
    result = None
    log = caps = run_id = t0 = None
    try:
        from .progress import CompactionProgressBar

        async with await _client(args) as client:
            caps = await probe(client)
            log = EventLog(args.db)
            run_id = log.create_run(args.task)  # pre-create so the live bar can read it
            agent = _make_agent(
                args,
                client,
                caps,
                log,
                sandbox=sandbox,
                on_compact=CompactionProgressBar(),
            )
            state = {"tool": None}
            agent.on_tool = lambda name, phase: state.__setitem__(
                "tool", name if phase == "start" else None
            )
            if agent.auto_budget:
                print(
                    f"⛁ auto-compact armed at {agent.context_budget:,} tokens "
                    f"({int(agent.compact_fraction * 100)}% of {caps.context_window:,}-token "
                    f"context window)"
                )
            t0 = time.time()
            if sys.stdout.isatty():  # live bottom status bar while the run works
                from rich.console import Console
                from rich.live import Live

                with Live(
                    console=Console(), refresh_per_second=8, transient=True
                ) as live:

                    async def _refresh(spin=0):
                        while True:
                            live.update(
                                _cli_status(
                                    log, run_id, caps, t0, tool=state["tool"], spin=spin
                                )
                            )
                            spin += 1
                            await aio.sleep(0.15)

                    ref = aio.ensure_future(_refresh())
                    try:
                        result = await agent.run(args.task, run_id=run_id)
                    finally:
                        ref.cancel()
            else:
                result = await agent.run(args.task, run_id=run_id)
    finally:
        await sandbox.aclose()
    if result is not None:
        print(result.answer)
        if sys.stdout.isatty():  # a final, persistent status line
            from rich.console import Console

            Console().print(_cli_status(log, run_id, caps, t0))
        print(
            f"[run {result.run_id}] {result.status} · {result.model_calls} model calls "
            f"· tier {caps.tier()}"
        )


async def cmd_resume(args) -> None:
    async with await _client(args) as client:
        caps = await probe(client)
        agent = _make_agent(args, client, caps, EventLog(args.db))
        result = await agent.resume(args.run_id)
        print(
            f"[run {result.run_id}] {result.status} after {result.model_calls} model calls"
        )
        print(result.answer)


async def cmd_replay(args) -> None:
    async with await _client(args) as client:
        log = EventLog(args.db)
        report = await replay_run(log, args.run_id, client)
        print(report.summary())
        raise SystemExit(0 if report.identical else 2)


async def cmd_replay_tuned(args) -> None:
    """Counterfactual replay: re-run a logged answer step under an intervention
    (a grammar/guidance skill and/or an optimized instruction) so it produces
    something different — deterministically, with the evidence held fixed."""
    from ..tuned_replay import Intervention, replay_tuned

    if not args.skill and not args.instruction:
        raise SystemExit("give --skill (grammar) and/or --instruction (prompt-opt)")
    async with await _client(args) as client:
        caps = await probe(client)
        log = EventLog(args.db)
        skill = None
        label = []
        if args.skill:
            from ..skills.skill import SkillRegistry

            skill = SkillRegistry(args.skills_dir).get(args.skill)
            label.append(f"grammar:{args.skill}")
        if args.instruction:
            label.append("prompt-opt")
        iv = Intervention(
            label=" + ".join(label),
            skill=skill,
            system_prompt=args.instruction,
            seed=args.seed,
        )
        report = await replay_tuned(
            log, args.run_id, client, caps, iv, fork_index=args.fork
        )
        print(report.summary())


async def cmd_skill(args) -> None:
    from ..skills.skill import SkillRegistry
    from ..skills.exec import generate_with_skill

    registry = SkillRegistry(args.skills_dir)
    if args.skill_name == "list":
        names = registry.names()
        if not names:
            print(f"no skills found in {registry.skill_dir}")
            print(
                "add .toml skill files there, or point --skills-dir / HARNESS_SKILLS elsewhere"
            )
            return
        for name in names:
            print(f"{name:<20} {registry.get(name).description}")
        return
    skill = registry.get(args.skill_name)  # validate before touching the network
    async with await _client(args) as client:
        caps = await probe(client)
        result = await generate_with_skill(
            client, caps, skill, args.prompt or "", seed=args.seed
        )
        grammar_status = result.plan.status_of("grammar")
        print(
            f"[{skill.name}] valid={result.valid} attempts={result.attempts} "
            f"grammar={grammar_status.value if grammar_status else 'n/a'} (tier {caps.tier()})"
        )
        print(result.text)


async def cmd_background(args) -> None:
    from ..agent.memory import Memory
    from ..background import auto_skills, consolidate, induce_skills, reflect

    log = EventLog(args.db)
    memory = Memory(args.memory_db)
    async with await _client(args) as client:
        from ..inference.capabilities import probe

        caps = await probe(client)
        n_episodes = await consolidate(log, memory, client, limit=args.limit)
        n_lessons = await reflect(
            log,
            memory,
            client,
            limit=args.limit,
            caps=caps,
            min_agreement=args.min_agreement,
        )
        skill_docs = await auto_skills(
            log, client, args.drafts_dir, memory=memory, limit=args.limit
        )
        proposed = []
        if getattr(args, "autonomous_actions", False):
            from ..background import propose_actions

            proposed = await propose_actions(
                log, client, args.drafts_dir, limit=args.limit
            )
    drafts = induce_skills(log, args.drafts_dir)
    print(
        f"consolidated {n_episodes} episodes, {n_lessons} lessons, {len(skill_docs)} skill docs"
        + (f", {len(proposed)} proposed actions" if proposed else "")
        + f"; memory now holds {memory.count()} entries"
    )
    for d in [*skill_docs, *drafts, *proposed]:
        print(f"wrote: {d}")


def cmd_recall(args) -> None:
    from ..agent.memory import Memory

    memory = Memory(args.memory_db)
    for entry in memory.recall(args.query, limit=10):
        print(f"[{entry.kind}] {entry.text}")


def cmd_proxy(args) -> None:
    import json as _json

    import uvicorn

    from ..proxy.app import create_app
    from ..proxy.config import ProxyConfig
    from ..proxy.engine import ProxyEngine

    cfg = ProxyConfig(
        upstream_url=args.url,
        model=args.model,
        db=args.db,
        skills_dir=args.skills_dir,
        skill=args.skill,
        samplers=_json.loads(args.samplers) if args.samplers else {},
        bias_profile=args.bias_profile,
        banned_phrases=[p for p in (args.banned_phrases or "").split(",") if p],
        think_budget=args.think_budget,
        rescue=not args.no_rescue,
        max_internal_retries=args.max_internal_retries,
    )
    app = create_app(ProxyEngine(cfg))
    print(f"lo proxy: {args.host}:{args.port} -> {cfg.upstream_url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_tui(args) -> None:
    from ..integrations.load import load_config
    from ..tui.app import HarnessApp

    client = OpenAICompatClient(args.url, args.model)

    # By default the TUI is a CLIENT of a headless session server: if --server
    # wasn't given and --in-process wasn't requested, start one embedded on
    # localhost and connect to it. The same args.db backs both, so the TUI shares
    # it (full history, no event duplication) and only consumes the SSE stream for
    # live token/tool deltas. --in-process is the escape hatch to the old path.
    embedded = None
    server = getattr(args, "server", None)
    shared_db = False
    if server is None and not getattr(args, "in_process", False):
        server, embedded = _start_embedded_server(args)
        shared_db = True
        print(f"lo tui: embedded server at {server} (tail it from another terminal)")

    app = HarnessApp(
        client,
        args.db,
        max_steps=args.max_steps,
        use_guardrails=not args.no_guardrails,
        required_steps=[s for s in (args.required_steps or "").split(",") if s],
        terminal_tools=frozenset(s for s in (args.terminal_tool or "").split(",") if s),
        context_budget=args.context_budget,
        skills_dir=args.skills_dir,
        tools_config=load_config(args.tools),
        resample_threshold=args.resample_threshold,
        memory_dir=args.memory_dir,
        allow_all=args.allow_all,
        preset=args.preset,
        background=args.background,
        sandbox=getattr(args, "sandbox", "host"),
        server=server,
        shared_db=shared_db,
    )
    try:
        app.run()
    finally:
        if embedded is not None:
            embedded.should_exit = True


def cmd_runs(args) -> None:
    log = EventLog(args.db)
    for r in log.runs():
        n = len(log.events(r.run_id))
        print(f"{r.run_id}  {r.status:<10} {n:>4} events  {r.task[:60]}")


def _aggregate_stats(log: EventLog) -> dict:
    """Tokens, frontier-equivalent saving, calls and resamples across all runs."""
    from ..tui.render import frontier_saved, tokens_of

    tin = tout = calls = resamples = 0
    for r in log.runs():
        for ev in log.events(r.run_id):
            if ev.type == MODEL_CALL:
                calls += 1
                a, b = tokens_of(ev.payload)
                tin += a
                tout += b
            elif ev.type == POLICY_TRIGGERED:
                resamples += 1
    return {
        "calls": calls,
        "tokens": tin + tout,
        "saved": frontier_saved(tin, tout),
        "resamples": resamples,
    }


def cmd_cost(args) -> None:
    from ..tui.render import _money

    st = _aggregate_stats(EventLog(args.db))
    print(
        f"$0.00 spent  ·  ~{_money(st['saved'])} the same {st['tokens']:,} tokens would cost "
        f"on a frontier API  ·  {st['calls']} call(s)"
    )


def cmd_usage(args) -> None:
    from ..tui.render import _money

    st = _aggregate_stats(EventLog(args.db))
    print(f"  calls      {st['calls']}")
    print(f"  tokens     {st['tokens']:,}")
    print("  spent      $0.00  (local marginal cost is zero)")
    print(f"  saved      ~{_money(st['saved'])}  vs a frontier API")
    print(f"  resamples  {st['resamples']}")


def cmd_export(args) -> None:
    from ..events.export import transcript_markdown

    log = EventLog(args.db)
    run_id = args.run_id
    if not run_id:
        runs = log.runs()
        if not runs:
            raise SystemExit("no runs to export")
        run_id = runs[-1].run_id
    if log.run(run_id) is None:
        raise SystemExit(f"no such run: {run_id}")
    md = transcript_markdown(log, run_id)
    if args.stdout:
        print(md)
        return
    path = f"run-{run_id}.md"
    with open(path, "w") as f:
        f.write(md)
    print(f"wrote {path}  ({md.count(chr(10))} lines)")


def cmd_rewind(args) -> None:
    """Roll a run back to an earlier point. Without --seq, list the rewind points;
    with --seq, archive the tail and remove it (lossless)."""
    log = EventLog(args.db)
    if log.run(args.run_id) is None:
        raise SystemExit(f"no such run: {args.run_id}")
    if args.seq is None:
        points = log.rewind_points(args.run_id)
        if not points:
            raise SystemExit("nothing to rewind in this run yet")
        print(
            f"rewind points for {args.run_id[:8]} (use: lo rewind {args.run_id} --seq <N>):"
        )
        for seq, kind, preview in points:
            label = (
                "the original answer" if kind == "answer" else f"follow-up: {preview}"
            )
            print(f"  seq {seq:>3}  ✂ {label}")
        return
    archive_id = log.rewind(args.run_id, args.seq)
    if archive_id is None:
        raise SystemExit(f"nothing at or after seq {args.seq}")
    print(
        f"rewound {args.run_id[:8]} to before seq {args.seq}; tail archived as {archive_id}"
    )


async def cmd_context(args) -> None:
    """Show what's in a run's context window — a segmented token breakdown
    (system · task · assistant · tool I/O · free) against the probed window."""
    from rich.console import Console

    from ..tui import render

    log = EventLog(args.db)
    run_id = args.run_id
    if not run_id:
        runs = log.runs()
        if not runs:
            raise SystemExit("no runs in the log")
        run_id = runs[-1].run_id
    calls = log.events(run_id, type=MODEL_CALL)
    if not calls:
        raise SystemExit(f"run {run_id[:8]} has no model calls yet")
    body = calls[-1].payload.get("request_body") or {}
    breakdown = render.context_breakdown(body.get("messages") or [], body.get("tools"))
    window = None
    try:  # probe the live server for its context window (optional — shows % when present)
        async with await _client(args) as client:
            window = (await probe(client)).context_window
    except Exception:
        pass
    Console().print(render.context_panel(breakdown, window=window))


def _build_session_app(args, sandbox, *, interactive_permissions: bool = False):
    """Construct the session-server Starlette app + manager, shared by `lo
    serve` and the TUI's embedded server. Startup probes the upstream resiliently
    (a failed probe leaves /health degraded rather than crashing the server).
    `interactive_permissions` routes the ask tier to the client over the bus."""
    from ..events.bus import EventBus
    from ..server.app import create_server_app
    from ..server.sessions import SessionManager

    log = EventLog(args.db)
    bus = EventBus(log)
    state: dict = {"url": args.url, "model": args.model}

    from ..agent.presets import get_preset

    async def _auto_allow(_tool, _args):  # server has no interactive client yet:
        return True  # auto-approve the ask tier (deny still denies)

    def factory(on_token, on_tool, on_notice, preset=None):
        # Apply the per-session preset so plan/explore are actually enforced
        # server-side (system prompt + read-only exposed toolset + deny list),
        # not silently ignored. Default comes from --preset. The "ask" tier is
        # auto-approved (no over-the-wire approval UI yet); "deny" is honoured, so
        # plan mode genuinely can't write/run, and build keeps full tools.
        p = get_preset(preset or getattr(args, "preset", "build"))
        tools = ToolRegistry(builtin_tools(sandbox=sandbox))
        tools.permissions = p.permissions(approver=_auto_allow)
        if state.get("client") is None:
            detail = f" ({state['error']})" if state.get("error") else ""
            raise RuntimeError(
                f"upstream {state['url']} unreachable{detail} — "
                "press ^o in the TUI (or POST /connect) to point at your model server"
            )
        return Agent(
            state["client"],
            tools,
            log,
            capabilities=state["caps"],
            system_prompt=p.system_prompt,
            sampling=p.sampling,
            exposed_tools=p.exposed(),
            max_steps=args.max_steps,
            guardrails_factory=_guardrails_factory(args, tools),
            context_budget=args.context_budget,
            compact_fraction=getattr(args, "compact_fraction", 0.85),
            code_mode=getattr(args, "code_mode", True),
            sandbox=sandbox,
            on_token=on_token,
            on_tool=on_tool,
            on_notice=on_notice,
        )

    async def connect_upstream(url: str | None = None, model: str | None = None):
        """(Re)build the upstream client and re-probe. Called at startup and by
        POST /connect — an empty body just re-probes the current upstream."""
        if url:
            state["url"] = url.rstrip("/")
        if model is not None:
            state["model"] = model
        old = state.pop("client", None)
        state.pop("caps", None)
        state.pop("error", None)
        if old is not None:
            try:
                await old.aclose()
            except Exception:
                pass
        try:
            client = OpenAICompatClient(state["url"], state["model"])
            # Reachability first: probe() tolerates missing endpoints by design,
            # so a dead host would otherwise "probe" fine as a generic tier.
            await client.get("/v1/models")
            if not client.model:
                models = await client.list_models()
                if not models:
                    raise RuntimeError("no models on server and none configured")
                client.model = models[0]
            state["client"] = client
            state["caps"] = await probe(client)
        except Exception as e:
            state["error"] = str(e)
        return health()

    async def startup():
        await connect_upstream()

    def health():
        caps = state.get("caps")
        return {
            "status": "ok" if caps else "degraded",
            "upstream": state["url"],
            "model": state["client"].model if state.get("client") else state["model"],
            "capabilities": caps.to_dict() if caps else None,
            "error": state.get("error"),
        }

    manager = SessionManager(
        bus, factory, interactive_permissions=interactive_permissions
    )
    return create_server_app(
        manager, health=health, on_startup=startup, connect=connect_upstream
    )


def cmd_serve(args) -> None:
    """Headless session server — the OpenCode-style bus over HTTP+SSE. Clients
    (the TUI, `lo tail`, a future web view) all observe one live session."""
    import os
    import uvicorn

    from ..sandbox import SandboxUnavailable, make_sandbox

    try:
        sandbox = make_sandbox(getattr(args, "sandbox", "host"), os.getcwd())
    except SandboxUnavailable as e:
        raise SystemExit(f"✗ {e}")

    app = _build_session_app(
        args,
        sandbox,
        interactive_permissions=getattr(args, "approval", "auto") == "prompt",
    )
    print(
        f"lo serve: http://{args.host}:{args.port}  (open it in a browser for the web client)"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_embedded_server(args):
    """Start a `lo serve` in a daemon thread on localhost; return
    (url, server). The TUI connects to it as a client — so `lo tui` is the
    OpenCode-style client/server architecture out of the box, no separate process,
    and the same session is tail-able from another terminal."""
    import os
    import threading
    import time

    import uvicorn

    from ..sandbox import SandboxUnavailable, make_sandbox

    try:
        sandbox = make_sandbox(getattr(args, "sandbox", "host"), os.getcwd())
    except SandboxUnavailable as e:
        raise SystemExit(f"✗ {e}")

    port = getattr(args, "port", None) or _free_port()
    # The TUI's embedded server prompts for the ask tier through the bus (so the
    # PermissionModal works in server mode too), unless the user passed --allow-all.
    app = _build_session_app(
        args, sandbox, interactive_permissions=not getattr(args, "allow_all", False)
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    # uvicorn skips signal handlers off the main thread automatically.
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(300):  # wait until the socket is accepting (≤30s)
        if server.started:
            break
        time.sleep(0.1)
    return f"http://127.0.0.1:{port}", server


def cmd_tail(args) -> None:
    """Follow a session's live event stream from a running `lo serve` — a
    thin read-only client, proving many clients can observe one session."""
    import httpx

    _C = {"token_delta": "", "reasoning_delta": "\033[2m"}  # inline, dim for reasoning

    async def _tail():
        base = args.server.rstrip("/")
        async with httpx.AsyncClient(timeout=None) as c:
            run_id = args.run_id
            if args.task:
                r = await c.post(f"{base}/session", json={"task": args.task})
                r.raise_for_status()
                run_id = r.json()["run_id"]
                print(f"[{run_id}] {args.task}\n")
            if not run_id:
                raise SystemExit("give a run_id to follow, or --task to start one")
            etype = None
            async with c.stream(
                "GET", f"{base}/session/{run_id}/events?replay=1&once=1"
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        etype = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        _emit(etype, data)
        print()

    def _emit(etype, data):
        payload = data.get("payload", {})
        if etype in ("token_delta", "reasoning_delta"):
            pre = "\033[2m" if etype == "reasoning_delta" else ""
            post = "\033[0m" if pre else ""
            print(f"{pre}{payload.get('text', '')}{post}", end="", flush=True)
        elif etype == "tool_progress":
            if payload.get("phase") == "start":
                print(f"\n  ⚙ {payload.get('name')}…", flush=True)
        elif etype == "tool_call":
            print(
                f"\n  ↳ {payload.get('name')} → {str(payload.get('result'))[:80]}",
                flush=True,
            )
        elif etype == "run_completed":
            print(f"\n✓ {payload.get('answer', '')[:200]}", flush=True)
        elif etype == "run_failed":
            print(f"\n✗ {payload.get('error', '')[:200]}", flush=True)

    asyncio.run(_tail())


async def _probe_local_servers(
    extra: list[str] | None = None,
) -> tuple[str, str, str] | None:
    """Find the first reachable local OpenAI-compatible server, trying the common
    ports. Returns (url, server_name, first_model_id), or None if nothing answers."""
    import httpx

    candidates = [(u.rstrip("/"), "server") for u in (extra or [])] + [
        ("http://localhost:8080", "llama.cpp"),
        ("http://localhost:8000", "vLLM"),
        ("http://localhost:1234", "LM Studio"),
        ("http://localhost:11434", "Ollama"),
    ]
    async with httpx.AsyncClient(timeout=2.0) as c:
        for url, name in candidates:
            try:
                r = await c.get(f"{url}/v1/models")
                if r.status_code == 200:
                    data = r.json().get("data") or []
                    if data:
                        return url, name, data[0].get("id", "")
            except Exception:
                continue
    return None


def cmd_quickstart(args) -> None:
    """Auto-find a local server and drop straight into the TUI against it."""
    extra = [args.url] if getattr(args, "url", None) else []
    found = asyncio.run(_probe_local_servers(extra))
    if found is None:
        where = f"at {args.url} or " if extra else ""
        print(
            f"no server found {where}on localhost :8080 (llama.cpp), :8000 (vLLM), "
            ":1234 (LM Studio), or :11434 (Ollama)."
        )
        print(
            "start one, then re-run `lo quickstart` — for a server on another "
            "machine: lo quickstart --url http://<host>:<port>"
        )
        raise SystemExit(1)
    url, name, model = found
    print(f"✓ {name} at {url}  ·  {model}")
    print("starting the TUI…  (^c to quit)")
    # Re-use the real `tui` subparser so the app gets all its proper defaults;
    # leave the model blank so the TUI resolves the served model itself.
    cmd_tui(build_parser().parse_args(["tui", "--url", url]))


_DAEMON_SESSION = "harness-serve"


def cmd_daemon(args) -> None:
    """Run `lo serve` in a detached tmux session: start/stop/status/logs/attach."""
    import shlex
    import shutil
    import subprocess
    import sys

    if shutil.which("tmux") is None:
        raise SystemExit("tmux not found — install tmux, or run `lo serve` directly.")
    s = _DAEMON_SESSION
    alive = (
        subprocess.run(["tmux", "has-session", "-t", s], capture_output=True).returncode
        == 0
    )

    if args.action == "start":
        if alive:
            print(f"daemon already running (tmux '{s}') — attach: tmux attach -t {s}")
            return
        cmd = [
            sys.executable,
            "-m",
            "local_harness.cli.main",
            "serve",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--db",
            args.db,
            "--url",
            args.url,
            "--approval",
            args.approval,
        ]
        if args.model:
            cmd += ["--model", args.model]
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                s,
                " ".join(shlex.quote(c) for c in cmd),
            ],
            check=True,
        )
        print(f"lo daemon started (tmux '{s}') → http://{args.host}:{args.port}")
        print(
            f"  attach: tmux attach -t {s}  ·  logs: lo daemon logs  ·  stop: lo daemon stop"
        )
    elif args.action == "stop":
        r = subprocess.run(["tmux", "kill-session", "-t", s], capture_output=True)
        print("daemon stopped." if r.returncode == 0 else f"no running daemon ('{s}').")
    elif args.action == "status":
        if not alive:
            print("daemon: not running")
            return
        print(f"daemon: running (tmux '{s}') at http://{args.host}:{args.port}")
        try:
            h = httpx.get(f"http://{args.host}:{args.port}/health", timeout=3).json()
            caps = h.get("capabilities") or {}
            print(
                f"  health: {h.get('status')} · model {h.get('model')} · tier {caps.get('tier')}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  (server not answering /health yet: {e})")
    elif args.action == "logs":
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", s, "-p", "-S", "-200"],
            capture_output=True,
            text=True,
        )
        print(r.stdout if r.returncode == 0 else f"no running daemon ('{s}').")
    elif args.action == "attach":
        print(f"run:  tmux attach -t {s}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lo")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("quickstart", help="auto-find a local server and launch the TUI")
    p.add_argument("--url", help="also try this URL first (e.g. a LAN model server)")

    p = sub.add_parser("daemon", help="run the session server in the background (tmux)")
    p.add_argument("action", choices=["start", "stop", "status", "logs", "attach"])
    _add_common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--approval", default="auto", choices=["auto", "prompt"])

    p = sub.add_parser("probe", help="report the server's capability tier")
    _add_common(p)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("bench", help="measure lo-agent's advantages vs frontier APIs")
    _add_common(p)
    p.add_argument("--skills-dir", default=os.environ.get("HARNESS_SKILLS"))
    p.add_argument("--n", type=int, default=8, help="samples per bench (default 8)")
    p.add_argument(
        "--no-batch-invariance",
        action="store_true",
        help="skip the batch-invariance bench (it issues concurrent load; "
        "skip it on a shared/busy server)",
    )

    p = sub.add_parser("sandbox", help="install / check the microVM tool sandbox")
    p.add_argument("action", nargs="?", choices=["doctor", "install"], default="doctor")

    p = sub.add_parser("lora", help="list hot-swappable LoRA adapters on the server")
    _add_common(p)

    def _add_agent_flags(p):
        p.add_argument("--max-steps", type=int, default=20)
        p.add_argument(
            "--required-steps",
            help="comma-separated tools that must run before finishing",
        )
        p.add_argument(
            "--terminal-tool", help="comma-separated tools that may end the workflow"
        )
        p.add_argument(
            "--no-guardrails",
            action="store_true",
            help="disable rescue parsing, nudges, and step enforcement",
        )
        p.add_argument(
            "--context-budget",
            type=int,
            default=None,
            help="approx token budget that triggers compaction; overrides the "
            "auto budget derived from the model's context window",
        )
        p.add_argument(
            "--compact-fraction",
            type=float,
            default=0.85,
            help="auto-compact trigger as a fraction of the probed context window "
            "(default 0.85); ignored if --context-budget is given",
        )
        p.add_argument(
            "--sandbox",
            default="host",
            choices=["host", "microvm"],
            help="run bash inside a microVM (workdir mounted, host isolated); "
            "default host (unsandboxed). 'microvm' needs: lo sandbox install",
        )
        p.add_argument(
            "--no-code-mode",
            dest="code_mode",
            action="store_false",
            help="disable code-mode (default on): use classic per-tool calling "
            "instead of having the model write code that calls tools",
        )
        p.set_defaults(code_mode=True)

    p = sub.add_parser("run", help="run an agent task")
    _add_common(p)
    p.add_argument("task")
    _add_agent_flags(p)

    p = sub.add_parser("resume", help="resume an interrupted run from its event log")
    _add_common(p)
    p.add_argument("run_id")
    _add_agent_flags(p)

    p = sub.add_parser("tui", help="interactive TUI: live run viewer + task launcher")
    _add_common(p)
    _add_agent_flags(p)
    p.add_argument("--skills-dir", default=os.environ.get("HARNESS_SKILLS"))
    p.add_argument(
        "--tools",
        default=os.environ.get("HARNESS_TOOLS", "tools.json"),
        help="JSON config of UTCP manuals / MCP servers to load",
    )
    p.add_argument(
        "--resample-threshold",
        type=float,
        default=None,
        help="resample (ghost-retype) when mean logprob falls below this, e.g. -1.2",
    )
    p.add_argument(
        "--memory-dir",
        default=os.environ.get("HARNESS_MEMORY_DIR", ".harness/memory"),
        help="dir for self-editing memory (MEMORY.md / USER.md)",
    )
    p.add_argument(
        "--allow-all",
        action="store_true",
        help="skip tool-permission prompts (allow every tool)",
    )
    p.add_argument(
        "--preset",
        default="build",
        help="agent preset: build | plan | explore | general",
    )
    p.add_argument(
        "--background",
        action="store_true",
        help="overnight apprentice: learn (consolidate/reflect/skills) after each idle run",
    )
    p.add_argument(
        "--server",
        default=None,
        help="connect to an EXISTING `lo serve` at this URL instead of "
        "starting an embedded one",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="port for the embedded server (default: a free port)",
    )
    p.add_argument(
        "--in-process",
        action="store_true",
        help="escape hatch: drive the agent in-process (the pre-server path) "
        "instead of starting/using a session server",
    )

    p = sub.add_parser(
        "serve", help="headless session server (OpenCode-style bus over HTTP+SSE)"
    )
    _add_common(p)
    _add_agent_flags(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument(
        "--preset",
        default="build",
        help="default agent preset when a session request omits one "
        "(build | plan | explore | general)",
    )
    p.add_argument(
        "--approval",
        default="auto",
        choices=["auto", "prompt"],
        help="ask-tier tools: 'auto' approves them (headless), 'prompt' asks the "
        "connected client over the bus (default auto)",
    )

    p = sub.add_parser(
        "tail", help="follow a session's live stream from a running `lo serve`"
    )
    p.add_argument(
        "run_id", nargs="?", help="run to follow (omit with --task to start a new one)"
    )
    p.add_argument("--server", default="http://127.0.0.1:8099", help="lo serve URL")
    p.add_argument("--task", help="start a new session with this task and follow it")

    p = sub.add_parser(
        "replay", help="re-issue a run's model calls and verify determinism"
    )
    _add_common(p)
    p.add_argument("run_id")

    p = sub.add_parser(
        "replay-tuned",
        help="counterfactual replay: re-run a step under a grammar or optimized prompt",
    )
    _add_common(p)
    p.add_argument("run_id")
    p.add_argument("--skill", help="grammar/guidance skill to constrain the answer")
    p.add_argument(
        "--instruction", help="override/optimized system instruction (prompt-opt)"
    )
    p.add_argument(
        "--fork",
        type=int,
        default=None,
        help="model-call index to fork at (default: the last/answer call)",
    )
    p.add_argument("--seed", type=int, default=None, help="override the seed")
    p.add_argument("--skills-dir", default=os.environ.get("HARNESS_SKILLS"))

    p = sub.add_parser("runs", help="list runs in the event log")
    _add_common(p)

    p = sub.add_parser(
        "context", help="show a run's context-window usage (token breakdown)"
    )
    _add_common(p)
    p.add_argument(
        "run_id", nargs="?", help="run to inspect (default: the most recent)"
    )

    p = sub.add_parser(
        "rewind", help="roll a run back to an earlier point (lossless; tail archived)"
    )
    p.add_argument("--db", default=os.environ.get("HARNESS_DB", "harness.db"))
    p.add_argument("run_id")
    p.add_argument(
        "--seq",
        type=int,
        default=None,
        help="remove events at/after this seq (omit to list the rewind points)",
    )

    p = sub.add_parser("cost", help="$ saved vs a frontier API across logged runs")
    p.add_argument("--db", default=os.environ.get("HARNESS_DB", "harness.db"))

    p = sub.add_parser(
        "usage", help="token / cost / resample summary across logged runs"
    )
    p.add_argument("--db", default=os.environ.get("HARNESS_DB", "harness.db"))

    p = sub.add_parser("export", help="write a run's transcript to run-<id>.md")
    p.add_argument("--db", default=os.environ.get("HARNESS_DB", "harness.db"))
    p.add_argument("run_id", nargs="?", help="run to export (default: the most recent)")
    p.add_argument(
        "--stdout",
        action="store_true",
        help="print Markdown to stdout instead of a file",
    )

    p = sub.add_parser(
        "skill", help="generate with a grammar skill ('skill list' to enumerate)"
    )
    _add_common(p)
    p.add_argument("skill_name")
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--skills-dir", default=os.environ.get("HARNESS_SKILLS"))
    p.add_argument("--seed", type=int, default=1)

    p = sub.add_parser(
        "background", help="run background cognition once: consolidate, reflect, induce"
    )
    _add_common(p)
    p.add_argument("--memory-db", default=os.environ.get("HARNESS_MEMORY", "memory.db"))
    p.add_argument("--drafts-dir", default="skills/drafts")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--min-agreement",
        type=float,
        default=0.5,
        help="keep a lesson only if at least this fraction of resamples agree "
        "on it (sample-consistency gate; default 0.5)",
    )
    p.add_argument(
        "--autonomous-actions",
        action="store_true",
        help="also draft PROPOSED next actions for stalled runs (never executed; "
        "written to the drafts dir for human review)",
    )

    p = sub.add_parser("recall", help="query the FTS5 memory")
    p.add_argument("query")
    p.add_argument("--memory-db", default=os.environ.get("HARNESS_MEMORY", "memory.db"))

    p = sub.add_parser(
        "proxy", help="guardrails + logit-pipeline proxy (OpenAI + Anthropic APIs)"
    )
    p.add_argument(
        "--url",
        default=os.environ.get("HARNESS_BASE_URL", "http://localhost:8080"),
        help="upstream model server",
    )
    p.add_argument("--model", default=os.environ.get("HARNESS_MODEL", ""))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--db", default="proxy.db")
    p.add_argument("--skills-dir", default=os.environ.get("HARNESS_SKILLS"))
    p.add_argument("--skill", help="default grammar skill applied to every request")
    p.add_argument(
        "--samplers", help='JSON sampler settings, e.g. \'{"min_p": 0.05, "dry": {}}\''
    )
    p.add_argument("--bias-profile")
    p.add_argument("--banned-phrases", help="comma-separated anti-slop phrase list")
    p.add_argument("--think-budget", type=int)
    p.add_argument("--no-rescue", action="store_true")
    p.add_argument("--max-internal-retries", type=int, default=2)

    return parser


_HANDLERS = {
    "quickstart": cmd_quickstart,
    "daemon": cmd_daemon,
    "probe": cmd_probe,
    "bench": cmd_bench,
    "sandbox": cmd_sandbox,
    "lora": cmd_lora,
    "run": cmd_run,
    "resume": cmd_resume,
    "replay": cmd_replay,
    "replay-tuned": cmd_replay_tuned,
    "runs": cmd_runs,
    "context": cmd_context,
    "rewind": cmd_rewind,
    "cost": cmd_cost,
    "usage": cmd_usage,
    "export": cmd_export,
    "skill": cmd_skill,
    "background": cmd_background,
    "recall": cmd_recall,
    "proxy": cmd_proxy,
    "tui": cmd_tui,
    "serve": cmd_serve,
    "tail": cmd_tail,
}


def main() -> None:
    args = build_parser().parse_args()
    handler = _HANDLERS[args.command]
    try:
        if asyncio.iscoroutinefunction(handler):
            asyncio.run(handler(args))
        else:
            handler(args)
    except SkillNotFound as e:
        import difflib

        msg = f"✗ unknown skill {e.name!r}"
        hint = difflib.get_close_matches(e.name, [*e.available, "list"], n=1)
        if hint:
            msg += f" — did you mean {hint[0]!r}?"
        msg += "\n  available: " + (", ".join(e.available) or "(none)")
        msg += "\n  ('lo skill list' to enumerate; --skills-dir / HARNESS_SKILLS to change where skills load from)"
        print(msg, file=sys.stderr)
        raise SystemExit(2)
    except httpx.ConnectError:
        url = getattr(args, "url", "the server")
        raise SystemExit(
            f"✗ can't reach {url} — is the server running? "
            "Check --url / HARNESS_BASE_URL."
        )
    except httpx.HTTPError as e:
        raise SystemExit(f"✗ server error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
