# local_harness

An agent harness built around the advantages local and self-hosted LLMs have over
frontier APIs: determinism & exact replay, spec-driven grammar skills, a token-level
logit pipeline, uncertainty-aware control flow, KV-cache-aware tree search, and
free-tokens compute scaling.

Plan: `thoughts/shared/plans/2026-06-12-local-llm-agent-harness-v2.md`
Status: **all five phases implemented.**

## Installation

### uv (recommended)

```bash
uv tool install "git+https://github.com/IMJONEZZ/lo-agent"
```

### Homebrew

```bash
brew tap IMJONEZZ/lo-agent
brew install lo-agent
```

### Build from source

```bash
git clone https://github.com/IMJONEZZ/lo-agent
cd lo-agent
uv pip install -e .
```

## Quickstart

```bash
# Probe your server's capabilities
lo probe --url http://localhost:8080

# Run an agent task
lo run "Use the calculator tool to compute 17*23 plus 100."

# Start the TUI
lo tui

# Start the proxy (OpenAI + Anthropic compatible)
lo proxy --url http://localhost:8080 --port 8088
```

See the [Usage](#usage) section below for the full command reference.

1. Substrate — client, capability prober, event log, bit-identical replay, crash resume
2. Spec-driven skills (grammar IR → GBNF/Lark/JSON-schema/validate-retry), logit
   pipeline (sampler zoo, bias profiles, think-budget forcing), logprob signals + step policies
3. Tree state with fork(), slot snapshots, best-of-N with verifiers, beam search,
   anti-slop phrase bans with backtracking
4. FTS5 memory, bootstrap-few-shot + instruction-search optimizers (DSPy optional:
   `--extra dspy`), background cognition (consolidate / reflect / induce)
5. Native in-process backend (`--extra native`): custom sampling loop with arbitrary
   logit processors, exact anti-slop via KV-cache rewind, classifier-free guidance with
   negative prompts, contrastive decoding, CAA activation steering, LoRA hot-swap,
   weight-level fine-tuning
6. Guardrails (after [forge](https://github.com/antoinezambelli/forge), MIT): rescue
   parsing of tool calls from free text, corrective nudges with channel separation
   (format errors → user, tool errors → tool), error budgets, required-step /
   terminal-tool / prerequisite enforcement with escalating nudges, and priority-based
   context compaction. On by default in `lo run`; tune with `--required-steps`,
   `--terminal-tool`, `--context-budget`, or `--no-guardrails`.
7. **Proxy mode** — the front door for any client. `lo proxy` serves both the
    OpenAI chat-completions API and the Anthropic Messages API (`/v1/messages`, so
    Claude Code works) in front of any upstream, applying the logit pipeline
    (grammar skills, sampler zoo, bias profiles, think budgets, anti-slop) and
    guardrails (rescue parsing, internal retry nudges, schema validate-and-retry)
    transparently. Every proxied call is event-logged and `lo replay`-able.
8. **TUI** — `lo tui` (Textual). A live runs table, a transcript view rendered
    straight from the event log (assistant panels with reasoning, tool calls,
    per-call seed/latency/confidence, guardrail rescues and nudges), and a task
    launcher. The TUI is a pure read-side consumer of the log, so runs started
    here, via `lo run` in another terminal, or through the proxy
    (`lo tui --db proxy.db`) all stream in live. `ctrl+r` replays the
    selected run against the server and reports whether it's bit-identical.

## TUI

```bash
lo tui                  # watch + launch agent runs (harness.db)
lo tui --db proxy.db    # watch live proxy traffic
```

Type a task in the bottom input and press Enter to launch an event-sourced
agent run; the transcript follows it live. Agent flags (`--required-steps`,
`--terminal-tool`, `--no-guardrails`, `--context-budget`, `--max-steps`) work
exactly as they do for `lo run`.

## Proxy

```bash
lo proxy --url http://localhost:8080 --port 8088
# then point opencode/aider/Continue at http://localhost:8088/v1
# or Claude Code / Anthropic SDKs at http://localhost:8088 (/v1/messages)
```

Server-wide defaults via flags (`--skill`, `--samplers '{"min_p":0.05,"dry":{}}'`,
`--bias-profile`, `--banned-phrases delve,tapestry`, `--think-budget 300`); any
request can override per-call with a `harness` extension object:

```json
{"messages": [...],
 "harness": {"skill": "sql_select", "samplers": {"min_p": 0.05},
             "think_budget": 200, "banned_phrases": ["delve"]}}
```

`GET /health` reports the upstream's probed capability tier. Buffered SSE is
emitted for `stream: true` in both dialects.

## Architecture: capability tiers

One OpenAI-compatible client + a capability prober + thin per-server adapters
(`llama.cpp`, `vLLM`, generic). Features unlock by verified tier:

| Tier | Requires | Unlocks |
|------|----------|---------|
| 0 | any OpenAI-compat endpoint | agent loop, event traces, validate-and-retry structure |
| 1 | + seed (verified live), logprobs | bit-identical replay, uncertainty signals |
| 2 | + grammar, logit_bias, sampler params, raw completion | CFG skills, sampler zoo, think-budget control |
| 3 | + KV/slot snapshots or parallel n | cheap tree search, fork/backtrack |
| 4 | in-process model (Phase 5) | steering, LoRA, arbitrary logit processors |

## Usage

```bash
uv sync

# What can this server do? (probes seed determinism with live test requests)
lo probe --url http://localhost:8080

# Run an agent task (event-sourced; every model call logged with its seed)
lo run "Use the calculator tool to compute 17*23 plus 100."

# List runs, resume a crashed run, or verify a run replays bit-identically
lo runs
lo resume <run-id>
lo replay <run-id>

# Grammar skills: guaranteed-valid output, server-constrained where possible
lo skill list
lo skill sql_select "names of users older than 30; table users(name, age)"

# Background cognition: summarize runs into memory, reflect on failures,
# induce draft skills from recurring output shapes; then query memory
lo background
lo recall "sql users"
```

Tip: start llama.cpp with `--slot-save-path /some/dir` to unlock true KV-state
snapshots (tree forks restore exactly instead of relying on prefix cache).

`--url/--model/--db` or `HARNESS_BASE_URL`/`HARNESS_MODEL`/`HARNESS_DB` select the
endpoint and event-log database (default `harness.db`).

## Layout

```
src/local_harness/
├── inference/   # OpenAI-compat client, capability prober, server adapters
├── events/      # append-only SQLite event log, deterministic replay
├── skills/      # grammar IR (EBNF -> GBNF/Lark/validator), TOML skills, execution
├── logits/      # pipeline stages: samplers, bias, grammar, think-budget,
│                # anti-slop (HTTP emulation), CFG/contrastive guidance (native)
├── guardrails/  # rescue parsing, nudges, step enforcement, error budgets
├── signals/     # logprob confidence metrics, step policies (resample/escalate/ask)
├── tree/        # conversation tree + fork, slot snapshots, best-of-N, beam
├── agent/       # event-sourced agent loop (resume from any crash), tools, FTS5 memory
├── optimize/    # bootstrap few-shot, instruction search, LoRA fine-tuning, DSPy adapter
├── background/  # idle-time cognition: consolidate, reflect, induce skills
├── native/      # Tier-4 in-process backend, activation steering, LoRA hot-swap
├── proxy/       # OpenAI + Anthropic API front door with pipeline + guardrails
├── tui/         # Textual app: live run viewer + task launcher over the event log
└── cli/         # lo probe|run|resume|replay|runs|skill|background|recall|proxy|tui
```

## Tests

```bash
uv run pytest
```

Unit tests run against mock servers (no GPU needed). Live verification against a
running llama.cpp server: `lo probe`, then `run` + `replay` — replay exits 0
only if the transcript hash matches.
