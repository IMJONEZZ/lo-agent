# Gap Analysis: local_harness vs Claude Code & opencode

**Repository:** `/home/imjonezz/Desktop/local_harness` (`lo-agent` v0.1.21)  
**Compared against:** Claude Code (via `ccunpacked.dev`, leaked GitHub source) and OpenCode (v1.17.x, `opencode.ai`)  
**Date:** 2026-07-08  
**Author:** Comprehensive codebase exploration + competitor research

---

## 1. Overview

`local_harness` (shipped as `lo-agent`) is a Python-based agent harness for local and self-hosted LLMs. It is built around capabilities that frontier-API clients structurally cannot offer: deterministic exact replay, token-level logit control, grammar-constrained decoding, KV-cache-aware tree search, microVM sandboxing, live capability probing with 5-tier graceful degradation, and free-token economics.

The project lives at `/home/imjonezz/Desktop/local_harness` with source under `src/local_harness/`. It is distributed as a CLI tool (`lo`), a Textual TUI (`lo tui`), a headless session server (`lo serve`), an OpenAI+Anthropic dual-protocol proxy (`lo proxy`), and a library.

**This analysis compares local_harness against Claude Code (the dominant frontier-API coding agent) and OpenCode (the leading open-source alternative).** It builds on the project's own excellent gap analysis documents at `docs/ccunpacked-comparison.md`, `docs/opencode-hermes-gap-analysis.md`, and `docs/gap-closure-plan.md`, adding new findings from direct competitor research and deeper codebase inspection.

---

## 2. Unique Strengths (Our Moat)

These are capabilities that neither Claude Code nor OpenCode has — structurally impossible for a frontier-API client (Claude Code) or architecturally absent (OpenCode). **This moat is intact and not threatened by any gap below.**

### 2.1 Deterministic Inference & Exact Replay
- **File:** `events/replay.py`, `events/log.py`, `agent/loop.py:472-479`
- Seeded generation with `seed = base_seed + call_index + 1000*attempt`
- `replay_run` re-issues every MODEL_CALL and hash-verifies transcript identity (SHA-256)
- `replay-tuned` forks a run at step K and re-generates under altered sampling/grammar/prompt
- `probe_batch_invariance` tests determinism under concurrent load — a Thinking Machines 2025 finding
- **Neither competitor can do this.** Claude Code uses cache-as-replay (never regenerates). OpenCode passes seed but never verifies.

### 2.2 Logit Pipeline
- **File:** `logits/` — `samplers.py`, `bias.py`, `grammar_stage.py`, `antislop.py`, `budget.py`, `pipeline.py`, `guidance.py`
- Full sampler zoo (min-p, XTC, DRY, typical, mirostat, et al.)
- Anti-slop with KV-cache rewind (llama.cpp `/v1/completions` raw path)
- Think-budget forcing (EOS suppression during reasoning, forced after budget)
- `logit_bias` profiles (concise, verbose, technical)
- Classifier-free guidance, contrastive decoding (native Tier-4 backend)
- **Neither competitor touches the logit layer.** It's impossible over a remote API.

### 2.3 Grammar Skills / Constrained Decoding
- **File:** `skills/` — `ir.py` (EBNF IR), `skill.py` (TOML skills), `exec.py` (generation), `structured/builder.py` (guidance/outlines-parity)
- Grammar IR → GBNF (llama.cpp native), Lark (Python validation), or JSON schema validate-and-retry
- Skills are tool-shaped primitives: `lo skill list`, `lo skill sql_select "names of users"`
- Guaranteed-valid structured output from models that would otherwise produce malformed JSON
- **Neither competitor has any grammar path.** OpenCode confirmed absent by grep.

### 2.4 Tree Search Over Agent Steps
- **File:** `tree/` — `state.py` (ConversationTree, fork, SlotSnapshots), `search/best_of_n.py`, `search/beam.py`, `search/self_consistency.py`, `search/plan_search.py`
- Fork agent state at any step, run N parallel rollouts
- Verifier-guided selection (logprob, grammar-validity, LLM-judge)
- Self-consistency: majority vote across N samples as confidence signal
- Plan search: fork N plans, rank by safety/clarity judge
- KV-cache-aware: near-free on local backends (prefix cache reuse, slot snapshots)
- **Neither competitor offers this.** Prohibitive on metered APIs.

### 2.5 microVM Sandboxing
- **File:** `sandbox.py` — `MicrosandboxSandbox` via libkrun
- Hardware-isolated execution with its own kernel (Linux KVM or macOS Apple Silicon)
- Workdir bind-mounted read-write at `/workspace`; everything else invisible
- Fail-closed: requesting microVM when unavailable raises `SandboxUnavailable`
- File ops (read/write/list_dir) also confined to the sandbox
- **Claude Code** has no sandbox; **OpenCode** uses Bun shell (no isolation).
- **Hermes Agent** uses Docker/SSH — shared kernel isolation, weaker than microVM.

### 2.6 Live Capability Probing & 5-Tier Degradation
- **File:** `inference/capabilities.py`, `inference/client.py`, `inference/adapters/`
- Probes seed determinism, logprobs, grammar, KV snapshots, LoRA, context window live
- 5-tier ladder: T0 (generic) → T1 (replay) → T2 (skills) → T3 (tree search) → T4 (native)
- Every feature degrades explicitly per probed tier; `_send` even latches off `logprobs`/`seed` mid-run
- Auto-budget: compact at 85% of the *probed* context window (not hardcoded)
- **Neither competitor adapts to arbitrary backends.** OpenCode has a static model catalog; Claude Code has one known backend.

### 2.7 Free-Token Economics
- **File:** `tui/render.py:frontier_saved()`, `cli/main.py:cmd_cost`
- `$0.00 spent · ~$X saved vs a frontier API` counter in the TUI status bar
- Background cognition (consolidate/reflect/induce) runs on every idle moment at zero marginal cost
- **OpenCode Zen is a *paid* model gateway** — the philosophical opposite.
- **Claude Code** bills per token.

---

## 3. Critical Gaps (Must-Have)

These are the features that users migrating from Claude Code or OpenCode will notice first. They represent the biggest capability gaps and the highest-impact engineering targets.

### 3.1 LSP Integration

**Competitor:** OpenCode integrates 30+ language servers (pyright, typescript, rust-analyzer, clangd, gopls, etc.) auto-detected per project. Diagnostics are fed into the agent's context after every edit. An experimental `lsp` tool provides `goToDefinition`, `findReferences`, `hover`, `documentSymbol`, `workspaceSymbol`, `goToImplementation`, `prepareCallHierarchy`, and more. (Source: `opencode.ai/docs/lsp/`)

Claude Code's `FileEdit` also routes through LSP diagnostics.

**What we have:** `grep` (regex text search), `glob` (file pattern match), `read_file` (full file read). No semantic code intelligence whatsoever.

**What needs to be built:**

1. **LSP manager module** at `integrations/lsp/manager.py`:
   - Auto-detect project type (pyproject.toml → pyright, Cargo.toml → rust-analyzer, package.json → typescript)
   - Start LSP server subprocesses via stdio JSON-RPC
   - Maintain per-file diagnostics state
   - Expose `diagnostics_for_file(path) -> list[Diagnostic]`

2. **Auto-diagnostics injection** at `agent/loop.py`:
   - After every `write_file`/`edit_file`/`apply_patch` tool call, collect diagnostics for the changed file
   - Inject as a system notice or tool-result suffix (configurable)
   - Gate behind a capability flag (LSP-available servers)

3. **LSP tool** at `agent/tools.py`:
   - `lsp_go_to_definition(file, line, col) -> location`
   - `lsp_find_references(file, line, col) -> locations`
   - `lsp_hover(file, line, col) -> markup_content`
   - `lsp_workspace_symbols(query) -> symbols`
   - Each returns structured data the model can use for precise code navigation

4. **Configuration** at `.lo/config.json` / `lo.json`:
   - `"lsp": true` to enable all auto-detected servers
   - `"lsp": {"typescript": {"disabled": true}}` to disable specific servers
   - `"lsp": {"custom": {"command": ["my-lsp"], "extensions": [".ext"]}}` for custom servers
   - `"lsp_diagnostics": "after_edit" | "always" | "never"`

**Effort:** L (>1 week). This is the single biggest coding-agent capability gap. Without it, the agent navigates code purely by text patterns (grep/glob) — functional but far less efficient than semantic navigation.

**Priority:** CRITICAL

### 3.2 OAuth MCP Support

**Competitor:** OpenCode has full OAuth for remote MCP servers: Dynamic Client Registration (RFC 7591), browser-based authorization, secure token storage at `~/.local/share/opencode/mcp-auth.json`, `opencode mcp auth/list/logout` commands, and automatic flow detection on 401. (Source: `opencode.ai/docs/mcp-servers/#oauth`)

Claude Code has `McpAuth` tool + credential management.

**What we have:** `integrations/mcp.py:http_transport()` supports static bearer tokens (line 82-102) via `token`/`token_env` config fields. On 401, raises `MCPError` with a message about config — no browser redirect, no token caching, no auth flow.

**What needs to be built:**

1. **OAuth authorization code flow** at `integrations/mcp_auth.py`:
   - On `401` + `WWW-Authenticate` header, initiate OAuth discovery via `/.well-known/oauth-authorization-server`
   - Open browser for user authorization (redirect to localhost callback)
   - Exchange authorization code for tokens
   - Support Dynamic Client Registration (RFC 7591) when no client_id is configured
   - Support pre-registered client credentials from config

2. **Credential store** at `~/.harness/mcp-credentials.json`:
   - Encrypt tokens at rest (keyring or file-based encryption)
   - Automatically refresh expired tokens using refresh_token
   - Support token revocation (`opencode logout` equivalent)

3. **MCP auth commands** at `cli/main.py` and `tui/app.py`:
   - `lo mcp auth <server>` — trigger auth flow for a named server
   - `lo mcp list` — show servers + auth status
   - `lo mcp logout <server>` — revoke + remove tokens
   - `/mcp auth <server>` — TUI slash command equivalent

**Effort:** M (3-5 days). Unlocks an entire class of SaaS MCP servers (Sentry, GitHub, Linear, etc.)

**Priority:** HIGH

### 3.3 Session Sharing / Multiplayer

**Competitor:** OpenCode's signature feature: `/share` → public `opncd.ai/s/<id>` — any session becomes a shareable read-only link your team can view, comment on, or use for debugging. (Source: `opencode.ai/docs/share/`)

**What we have:**
- `events/export.py` has `transcript_markdown(log, run_id)` — exports a run as Markdown
- `events/replay.py` can replay any run from the log
- `server/app.py` serves `GET /session/{id}/events` as SSE (streaming)
- We have the full substrate for sharing (event log + replay + export) but **no productized `/share` surface**

**What needs to be built:**

1. **Share endpoint** at `server/share.py`:
   - `POST /share` → creates a short unique id, stores the run_id mapping, returns a URL
   - `GET /share/<id>` → serves a read-only replay of the event log as HTML+JS (single-page viewer)
   - Optional: `GET /share/<id>/raw` → raw JSON event log for programmatic consumption
   - Optional: `GET /share/<id>.md` → markdown transcript (reuses `transcript_markdown`)

2. **Share viewer HTML** at `server/templates/share.html`:
   - Vanilla JS SPA that subscribes to the SSE stream (or reads a snapshot)
   - Renders the transcript in read-only mode
   - Shows tool calls, reasoning, token counts, confidence overlay
   - No edit/run capability (purely observational, like OpenCode)

3. **Share hosting** — two options:
   - **Local:** `/share` generates a URL on the running `lo serve` instance (LAN only)
   - **Cloud:** Optional hosted service at (e.g.) `lo.agent/share` for public URLs. The event log is already replayable — a hosted share service just stores the run_id → event mapping and serves the viewer.

4. **Slash command** at `tui/app.py`:
   - `/share` → creates share link, copies to clipboard
   - `/share <run_id>` → shares a specific past run

**Effort:** M (3-5 days for local sharing + viewer). L (>1 week with hosted service).

**Priority:** HIGH

### 3.4 GitHub CI / PR Automation

**Competitor:** OpenCode has `/opencode` on any PR — the agent runs as a GitHub Action, reviews the diff, posts comments, suggests fixes, and can even auto-fix PRs. (Source: `opencode.ai/docs/github/`)

Claude Code has similar CI integration (gated).

**What we have:** The full substrate:
- Review preset at `agent/presets.py:91` with `_REVIEW_PROMPT`
- Security-review preset at `agent/presets.py:96` with `_SECURITY_PROMPT`
- Grammar skills that constrain review output to valid findings lists (`tui/app.py:_findings_skill` at line 130)
- `diff_artifact()` in `tui/render.py`
- The review flow exists in the TUI but **has no CI/PR surface**

**What needs to be built:**

1. **GitHub Action** at `.github/actions/lo-review/action.yml`:
   - Trigger on `pull_request` and `pull_request_review` events
   - Run `lo review --diff` on the PR diff (compute from `git diff origin/main...HEAD`)
   - Post structured findings as PR comments (reuse the findings grammar skill)
   - Optional: `lo fix` mode that suggests or applies fixes

2. **GitLab CI** integration following the same pattern.

3. **`lo ci` subcommand** at `cli/main.py`:
   - `lo ci review` — review the CI diff (works in any CI, not just GitHub Actions)
   - `lo ci check` — run all configured checks (review + security-review + lint)
   - Outputs findings in a machine-readable format for CI annotation

4. **Auto-fix (optional):** Allow the agent to push fix commits to a PR branch, gated behind a `"fix": "ask"` permission.

**Effort:** L (>1 week for full Action + CI surface). M for the basic review-action.

**Priority:** HIGH

### 3.5 Vision / Image Input

**Competitor:** OpenCode accepts drag-and-drop images in the terminal — they're added to the message context for vision models. Claude Code supports image input through its API.

**What we have:** No image/multimodal support at all. The message types in `inference/types.py` don't have an image content variant. The proxy at `proxy/anthropic.py` and `proxy/engine.py` doesn't handle image parts in the message body.

**What needs to be built:**

1. **Image content type** at `inference/types.py`:
   - Add `ContentPart` union type: `TextContent(text: str)` | `ImageContent(url: str, detail: str)`
   - Extend `Message.content` to accept `list[ContentPart]` (OpenAI format) in addition to plain string
   - Extend Anthropic conversion at `proxy/anthropic.py` to handle image blocks

2. **Vision capability detection** at `inference/capabilities.py`:
   - Probe whether the model supports vision (check model name for "vision" or probe with a test image)
   - Add `self.vision: bool = False` to the `Capabilities` dataclass
   - Gate image sending behind this flag

3. **Image input in TUI** at `tui/app.py`:
   - Accept drag-and-drop of image files onto the terminal (Textual supports this via `on_file_drop`)
   - Convert images to base64 data URIs for the API
   - Show a thumbnail in the prompt area before submission

4. **Image input in CLI** at `cli/main.py`:
   - Accept `--image` or `@image.png` in the task description
   - Read and encode the image

**Effort:** M (3-5 days). The protocol support is straightforward; the TUI drag-and-drop is the harder part.

**Priority:** HIGH

### 3.6 Plugin Hooks System

**Competitor:** OpenCode has a full JS/TS plugin system with `tool.execute.before` and `tool.execute.after` hooks, config lifecycle hooks, and startup/shutdown hooks. Plugins are distributed as npm packages. (Source: `opencode.ai/docs/plugins/`)

**What we have:** No plugin system. All behavior is hardcoded or configured via the `ToolRegistry`, `Guardrails`, and agent presets. MCP/UTCP provide external tool loading but no code-level hooks.

**What needs to be built:**

1. **Plugin registry** at `agent/plugin.py`:
   ```python
   class Plugin:
       name: str
       version: str
       hooks: dict[str, Callable]

   class PluginRegistry:
       def register(self, plugin: Plugin) -> None
       def unload(self, name: str) -> None
       def trigger(self, hook: str, context: dict) -> dict
   ```

2. **Hook points** throughout the agent lifecycle:
   - `before_tool_exec(tool_name, arguments, confidence)` → can modify arguments, deny execution, or inject side effects
   - `after_tool_exec(tool_name, arguments, result)` → can modify result or trigger side effects
   - `before_model_call(messages, sampling_params)` → can modify messages/params
   - `after_model_call(request, response)` → can process or log the response
   - `on_startup()`, `on_shutdown()` — lifecycle
   - `on_config_loaded(config)` — validate or augment config
   - `on_run_completed(run_id, result)` — post-run processing

3. **Plugin loading** from `.lo/plugins/*.py`:
   - Each `.py` file exports a `plugin: Plugin` object
   - Sandboxed execution (plugins run with access to a restricted API)
   - Can also load from `~/.lo/plugins/` (global) and `~/.config/lo/plugins/`

4. **Plugin management commands**:
   - `lo plugin list` / `/plugins`
   - `lo plugin install <path|url>` / `/plugins install`
   - `lo plugin unload <name>` / `/plugins unload`

**Effort:** M (3-5 days for hook system + registry). L (>1 week with full lifecycle + sandboxing).

**Priority:** HIGH

### 3.7 Unified Configuration System

**Competitor:** OpenCode has a single `opencode.json` (or `opencode.jsonc`) in the project root with a published JSON Schema at `https://opencode.ai/config.json`. It covers providers, models, MCP servers, agents, tools, permissions, LSP, themes, keybinds, formatters, rules, instructions, policies, and hooks. (Source: `opencode.ai/docs/config/`)

**What we have:** Configuration is scattered across:
- CLI flags with env-var overrides (`HARNESS_BASE_URL`, `HARNESS_MODEL`, `HARNESS_DB`, etc.) — `cli/main.py`
- `tools.json` for MCP/UTCP — `integrations/load.py`
- `~/.harness/config.json` for TUI theme/vim — `tui/app.py:_CONFIG_PATH`
- Markdown files in `.lo/commands/` and `.lo/agents/` — `agent/commands.py`, `agent/presets.py`
- The `profiles/` directory for bias profiles — `logits/bias.py`
- No single source of truth, no schema, no validation

**What needs to be built:**

1. **Config schema** at `src/local_harness/config/schema.py` (or a standalone JSON Schema file at `config-schema.json`):
   ```python
   @dataclass
   class LoConfig:
       # Provider
       url: str = "http://localhost:8080"
       model: str = ""
       api_key: str | None = None

       # MCP Servers
       mcp: dict[str, MCPServerConfig] = field(default_factory=dict)

       # Agents / Presets
       agent: dict[str, AgentConfig] = field(default_factory=dict)
       default_agent: str = "build"

       # Tools & Permissions
       permission: dict[str, str | dict] = field(default_factory=dict)

       # LSP
       lsp: bool | dict = False

       # Memory
       memory_dir: str = ".harness/memory"

       # Sandbox
       sandbox: str = "host"  # "host" | "microvm"

       # Theme
       theme: str = "osaka-jade"

       # Keybinds
       keybinds: dict[str, str] = field(default_factory=dict)

       # Plugins
       plugins: list[str] = field(default_factory=list)

       # Instructions / rules files
       instructions: list[str] = field(default_factory=list)
   ```

2. **Config loader** at `src/local_harness/config/loader.py`:
   - Search order: `./lo.json` → `./lo.jsonc` → `./.lo/config.json` → `~/.lo/config.json`
   - Deep-merge with CLI flag overrides (CLI wins)
   - Validate against JSON Schema
   - Expose a singleton `config` module that any module can import

3. **Migration path**:
   - Support reading old env vars alongside new config for backward compatibility
   - `lo init` command that generates a starter `lo.json`

4. **JSON Schema** published at a URL (e.g., `https://raw.githubusercontent.com/IMJONEZZ/lo-agent/main/config-schema.json`) for IDE autocompletion.

**Effort:** M (3-5 days). This is foundational infrastructure that many other gaps depend on.

**Priority:** CRITICAL (blocking dependency for other config-based gaps)

### 3.8 Documentation Site

**Competitor:** OpenCode has a full documentation site at `opencode.ai/docs` — beautiful, searchable, versioned, with pages for every feature, config option, CLI command, and integration. Claude Code has `docs.anthropic.com`. Both are essential for adoption.

**What we have:** 15+ markdown files in `docs/` — detailed and accurate but unrendered, unsearchable, unbrowsable. No documentation site.

**What needs to be built:**

1. **Static site** using MkDocs (Material theme) or Astro (same stack as OpenCode):
   - **Getting Started:** Install, Quickstart, Configuration, First Run
   - **Architecture:** Event log, agent loop, capability tiers, proxy
   - **CLI Reference:** Every `lo` subcommand with flags, examples, env vars
   - **Configuration:** All config options, JSON Schema reference, env vars
   - **Agents:** Built-in presets, file-authored agents, permissions
   - **Tools:** Built-in tools, MCP, UTCP, custom tools
   - **MCP:** Local/remote servers, OAuth, resources
   - **Permissions:** allow/ask/deny, bash globs, doom_loop
   - **Sandboxing:** host vs microVM, installation, security model
   - **Proxy:** OpenAI API, Anthropic API, pipeline, logit control
   - **TUI:** Commands, keybinds, status bar, themes, modes
   - **Determinism & Replay:** replay, replay-tuned, batch invariance
   - **Memory:** MEMORY.md/USER.md/PROJECT.md, recall, structured facts
   - **Background Cognition:** consolidate, reflect, induce skills
   - **Tree Search:** beam, best-of-N, self-consistency, plan search
   - **Troubleshooting:** Connection issues, capability tiers, MCP auth, sandbox
   - **FAQ:** Common questions, comparisons

2. **CI/CD** — auto-build and deploy on push to `main`

3. **Versioned docs** — `docs/` folder mirrors `docs/v0.1/`, `docs/latest/` redirects

**Effort:** M (3-5 days for initial site from existing docs). L (>1 week for polished site with all pages).

**Priority:** HIGH

---

## 4. Feature-Level Gaps

### 4.1 CLI/UX

| Gap | Competitor | What We Have | Need To Build | File References | Effort |
|-----|-----------|-------------|---------------|-----------------|--------|
| Inline diff visualization after edits | Both show diffs before applying | `edit_file` returns `"edited {path}"`; `diff_artifact()` exists but unwired | Wire `diff_artifact()` into tool-result render path in TUI | `tui/render.py:diff_artifact`, `tui/app.py:chat_tool()` | S |
| `/undo` / `/redo` | Both: multi-step undo/redo | `delete_events_from()` + `reopen()` exist; CLI `cmd_rewind` exists | Add `/undo` `/redo` slash commands in TUI | `events/log.py:delete_events_from`, `cli/main.py:584`, `tui/app.py:_handle_slash` | S |
| `/context` visualizer | Claude Code: colored grid | `estimate_tokens()`, `context_breakdown()`, `cmd_context` exist | Wire `context_panel()` into `/context` slash command | `agent/compaction.py:82`, `tui/render.py:context_panel`, `cli/main.py:611` | S |
| `/model` switch | Both: `/model <name>` | `ModelPickerScreen` exists; `POST /connect` exists | Add `/model` slash command | `tui/app.py:251`, `server/app.py:109` | S |
| `/fast` / `/effort` | Claude Code | `_EFFORT` dict exists at `tui/app.py:151` | Add `/fast` `/effort <level>` slash commands | `tui/app.py:151-163` | S |
| Shell mode (`!`) | Both: `!` prefix | Plan C of 2026-07-08 plan just shipped | Verify and polish | `tui/app.py:PromptInput`, plan at `thoughts/.../2026-07-08-...` | S |
| `/clear` / `/new` | Both | No equivalent | Add command to reset conversation | `tui/app.py:_handle_slash` | S |
| Theme switching | Both: rich theme system | 7 themes + picker + live preview | Persist choice, add auto light/dark detection, route palette through app vars | `tui/app.py:ThemePickerScreen`, `tui/render.py:Palette` | S |
| Cost/usage dashboards | Both | `usage_panel()`, `frontier_saved()` exist | `/cost` and `/usage` slash commands (wrapping existing data) | `tui/render.py:usage_panel`, `cli/main.py:541` | S |
| `/agents` list | OpenCode: `/agents` | `all_preset_names()` exists | List active agents + spawned children | `agent/presets.py:186`, `server/sessions.py:sessions()` | S |
| `/tools` / `/mcp` list | Both | MCP config loaded from JSON | List loaded tools/servers with auth status | `integrations/load.py`, `agent/tools.py:ToolRegistry` | S |

### 4.2 Agent System

| Gap | Competitor | What We Have | Need To Build | File References | Effort |
|-----|-----------|-------------|---------------|-----------------|--------|
| Plan→Implement→Verify lifecycle | Claude Code: `EnterPlanMode`/`ExitPlanMode`/`VerifyPlanExecution` | `plan` preset, `plan_search`, `/approve` system command stub | Complete the flow: plan→review→approve→implement→verify | `agent/presets.py:85`, `tree/search/plan_search.py`, `tui/app.py:667-672,815` | M |
| Scout agent (read-only dependency researcher) | OpenCode: `@scout` subagent | `explore` preset exists | Add `scout` preset that works in temp dir | `agent/presets.py` | M |
| ToolSearch lazy loading | Claude Code: 50+ tools, lazy schemas | Deferral threshold at `agent/loop.py:45`; `_run_tool_search` at line 542 | Complete the promote-on-search flow; verify all MCP/UTCP tools deferrable | `agent/loop.py:186-206`, `agent/tools.py:22-43` | M |
| Non-blocking subagents | OpenCode: background subagents | `spawn_agents` blocks (awaits children) | Add `wait=false` variant + `collect_results` tool | `server/coordinator.py:27` | M |
| UltraPlan / long-horizon | Claude Code (gated): 30-min planning | `max_steps=20`, no long-plan preset | Add `ultraplan` preset with relaxed limits + periodic compaction | `agent/presets.py`, `agent/loop.py:99` | S |
| Per-agent model override | Both | `AgentPreset.model` field exists but unwired (comment: "not yet wired") | Wire model switching into agent factory + proxy | `agent/presets.py:65`, `tui/app.py:_apply_preset` | M |

### 4.3 MCP / Plugin Ecosystem

| Gap | Competitor | What We Have | Need To Build | File References | Effort |
|-----|-----------|-------------|---------------|-----------------|--------|
| Interactive OAuth for remote MCP | OpenCode: full OAuth flow | Bearer token only; 401 raises unhelpful error | OAuth auth code + PKCE + token store + CLI commands | `integrations/mcp.py:82-102` | M |
| MCP resources support | Both: resources + prompts | Only `tools/list` and `tools/call` implemented | Add `resources/list`, `resources/read`, `prompts/list`, `prompts/get` | `integrations/mcp.py:43-48` | M |
| MCP timeout configuration | OpenCode: per-server timeout (ms) | Hardcoded 30s at `integrations/mcp.py:93` | Pass timeout from config | `integrations/mcp.py:93`, `integrations/load.py:37-45` | S |
| Plugin hook system | OpenCode: JS/TS hooks | None | Plugin registry + lifecycle hooks + loading from `.lo/plugins/` | New: `agent/plugin.py` | M |
| `.claude/skills/` interop | Both: SKILL.md standard | Only `.toml` skills format | Read `.claude/skills/*.md` alongside `.harness/skills/*.toml` | `skills/skill.py` | M |
| `opencode.json` interop | OpenCode: its own config | Read `.opencode/commands/` and `.opencode/agents/` already | Also read MCP/agent definitions from `opencode.json` | `agent/commands.py:44-50`, `agent/presets.py:113-115` | M |

### 4.4 File Editing

| Gap | Competitor | What We Have | Need To Build | File References | Effort |
|-----|-----------|-------------|---------------|-----------------|--------|
| `apply_patch` (unified diff) | Both: patch tool | Only `edit_file` (string-replace) | Add `apply_patch(patch_text)` tool with parser | `agent/tools.py:260-273` | M |
| Line-range file read | OpenCode: `read` with line range | `read_file` returns entire file (capped 64KB) | Add optional `start_line`, `end_line` params | `agent/tools.py:166-167` | S |
| Post-edit auto-formatting | Both: auto-format after edit | No formatting step | Detect formatter from project config, run on changed file | `agent/tools.py` (new hook) | M |
| Batch file operations | Both (gated) | Not needed (code-mode chains ops) | Defer — code-mode already solves this | `agent/codemode.py` | - |
| Notebook edit improvements | Both: NotebookEdit | `notebook_edit` exists at `agent/tools.py:305-351` | Add to TUI artifact renderer | `tui/render.py:notebook_artifact` | S |
| `repl` tool (persistent Python) | Both: REPL | `repl` tool exists at `agent/tools.py:393-394` | Verify replay-correct design (cells logged → rebuilt on resume) | `agent/tools.py:354-394`, `agent/loop.py:_replay_repl:284` | M |

### 4.5 Configuration

| Gap | Competitor | What We Have | Need To Build | File References | Effort |
|-----|-----------|-------------|---------------|-----------------|--------|
| Single config file | Both: single JSON/JSONC file | CLI flags + env vars + tools.json + ~/.harness/config.json | Design `lo.json` schema + loader + merger | New: `config/` module | M |
| JSON Schema for config | OpenCode: published schema | No schema | Write + publish JSON Schema | New: `config-schema.json` | S |
| Config validation | Both | None | Validate on load, report errors with line numbers | `config/loader.py` | S |
| Per-agent model override | Both | Field exists, unwired | Wire into agent factory + proxy | `agent/presets.py:65` | M |
| `.ignore` file support | OpenCode: `.ignore` file | Hardcoded `_GREP_IGNORE` set | Read `.loignore` / `.gitignore` | `agent/tools.py:276` | S |
| Bash-command glob permissions | OpenCode: per-command bash perms | `bash` is a single tool name | Add bash sub-command routing to Permissions | `agent/permissions.py:35`, `agent/tools.py:bash_fn` | M |

### 4.6 Security / Sandboxing

| Gap | Competitor | What We Have | Need To Build | File References | Effort |
|-----|-----------|-------------|---------------|-----------------|--------|
| doom_loop guard | Both: repeated-call detection | Error budgets + step enforcement only | Add `(tool_name, args_truncated)` fingerprint tracking over N steps; inject nudge on >M repeats | `guardrails/steps.py` | M |
| Bash command glob allowlists | OpenCode: `"git *": "ask"` | Tool-level allow/ask/deny only | Parse first word of bash command, match against per-command permission patterns | `agent/permissions.py:35`, `agent/tools.py:181-196` | M |
| Permission prompt in CLI mode | Both (TTY fallback) | Auto-approve lambda in server factory | For `lo run` on TTY, fall back to stdin prompt; for non-TTY, deny | `cli/main.py:656-657`, `agent/permissions.py:approver` | M |
| External path sandboxing | OpenCode: `external_directory` permission | `HostSandbox._host_path()` at `sandbox.py:89` rejects escapes | Make escape-rejection default even in host mode | `sandbox.py:68-107` | S |

### 4.7 Documentation

| Gap | Competitor | What We Have | Need To Build | File References | Effort |
|-----|-----------|-------------|---------------|-----------------|--------|
| Documentation site | Both: hosted, searchable, versioned | 15+ markdown files in `docs/` | Build MkDocs/Astro site, auto-deploy on push | `docs/` directory | M |
| OpenAPI spec + SDK | OpenCode: `/doc` → OpenAPI 3.1 + TypeScript SDK | Hand-rolled Starlette, no spec | Annotate `server/app.py` routes, emit spec, ship Python client | `server/app.py` | M |
| Troubleshooting guide | Both | Error messages in code but no organized guide | Write `docs/troubleshooting.md` | New: `docs/troubleshooting.md` | M |
| Configuration reference | Both | Spread across CLI --help + env vars + docs/ | Create single config reference page | `docs/configuration.md` | M |
| Changelog | OpenCode: `/changelog` | Git log only | Create `CHANGELOG.md` | Root | S |

---

## 5. Files Needing Work

This section lists every source file that needs modification or creation, grouped by area. Files are ranked by how much change they need.

### Hot Zone (heavy changes needed)

| File | Lines | Issues | Priority |
|------|-------|--------|----------|
| `tui/app.py` | ~2500 | Monolithic; missing 6+ slash commands (undo, redo, context, model, effort, mcp, agents, share, plugins, cost, usage, clear). The TUI is the user's primary interface and needs the most feature work. | CRITICAL |
| `integrations/mcp.py` | 121 | No OAuth, no resources, no prompts, no timeout config. Must grow significantly (3-5×). | CRITICAL |
| `agent/tools.py` | 620 | Missing `apply_patch`, line-range read, file-format hooks, bash sub-permissions, formatter hooks. | HIGH |
| `agent/permissions.py` | 77 | No doom_loop, no bash-command globs, no non-TTY approver, no `external_directory` concept. Currently the thinnest permissions system of the three tools. | HIGH |
| `cli/main.py` | 1372 | Missing commands: `mcp auth/list/logout`, `share`, `plugin`, `ask`. Growing monolithic—should refactor into separate command modules. | HIGH |
| `server/app.py` | 142 | No OpenAPI spec, minimal web UI, no share endpoint, no plugin hooks. Needs to grow into a real API surface. | HIGH |

### New files needed

| File | Purpose | Priority |
|------|---------|----------|
| `integrations/lsp/manager.py` | LSP server lifecycle + diagnostics | CRITICAL |
| `integrations/lsp/types.py` | LSP type definitions | CRITICAL |
| `integrations/lsp/tools.py` | LSP tool implementations | CRITICAL |
| `integrations/mcp_auth.py` | OAuth authorization code flow | HIGH |
| `config/schema.py` | `lo.json` schema dataclasses | CRITICAL |
| `config/loader.py` | Config file discovery + parsing + merging | CRITICAL |
| `config/validators.py` | Config validation helpers | CRITICAL |
| `agent/plugin.py` | Plugin registry + hook system | HIGH |
| `server/share.py` | Session sharing endpoints | HIGH |
| `server/templates/share.html` | Share viewer SPA | HIGH |
| `agent/policies.py` | doom_loop detection (or extend guardrails/) | MEDIUM |
| `docs/troubleshooting.md` | Troubleshooting guide | MEDIUM |
| `config-schema.json` | Published JSON Schema | MEDIUM |

### Existing files needing moderate changes

| File | Lines | Changes Needed | Priority |
|------|-------|---------------|----------|
| `agent/presets.py` | 194 | Wire `model` field, add `scout` preset, `ultraplan` preset | MEDIUM |
| `agent/frontmatter.py` | 156 | Already done for 2026-07-08 plan; ensure OpenCode frontmatter interop | MEDIUM |
| `agent/commands.py` | 116 | Already done for 2026-07-08 plan; ensure `!`cmd`` sandbox routing clean | MEDIUM |
| `agent/notebook.py` | 195 | AGENTS.md interop already done; verify server-mode injection | MEDIUM |
| `guardrails/steps.py` | ~80 | Add doom_loop fingerprint tracking | MEDIUM |
| `guardrails/errors.py` | ~60 | Add doom_loop error budget variant | MEDIUM |
| `agent/loop.py` | 654 | Add diagnostics injection after tool calls; add plugin hooks | MEDIUM |
| `proxy/engine.py` | 270 | Add vision support (image parts in body) | MEDIUM |
| `proxy/anthropic.py` | ~200 | Add image block conversion | MEDIUM |
| `inference/types.py` | ~200 | Add ContentPart union type for multimodality | MEDIUM |
| `inference/capabilities.py` | ~300 | Add `vision`, `lsp` capability probes | MEDIUM |
| `events/export.py` | ~150 | Add JSON and HTML export formats for sharing | LOW |
| `skills/skill.py` | ~200 | Add `.md` skill file reading (SKILL.md interop) | LOW |
| `integrations/load.py` | 55 | Pass MCP timeout from config | LOW |
| `tui/render.py` | ~800 | Add notebook_artifact(), polish diff_artifact() | LOW |
| `server/webui.py` | ~200 | Upgrade to proper web app | LOW |

### Files safe from major changes (our moat)

| File | Why |
|------|-----|
| `events/log.py` | Core append-only log — stable, correct, don't touch |
| `events/replay.py` | Hash-verified replay — unique advantage |
| `events/bus.py` | SSE pub/sub — correct separation of persistent/ephemeral |
| `logits/*` | Full pipeline — competitor-inaccessible |
| `skills/ir.py`, `skills/exec.py` | Grammar IR → GBNF/validate — unique |
| `tree/*` | State + search — unique |
| `sandbox.py` | microVM isolation — unique |
| `native/*` | In-process backend — unique |
| `signals/*` | Logprob confidence — competitor-inaccessible |
| `inference/adapters/*` | Backend-specific logic — stable bridge |
| `proxy/app.py` | Dual-protocol surface — unique wrapper capability |
| `server/sessions.py` | Session manager — correct substrate for orchestration |
| `server/coordinator.py` | Sub-agent tools — already built |

---

## 6. Phased Build Roadmap

### Phase 1: Quick Wins (Week 1)

**Theme:** Maximum perceived improvement, minimum risk. Pure UX work on existing substrate.

| # | Item | Files | Effort | Depends On |
|---|------|-------|--------|------------|
| 1 | `/context` visualizer overlay | `tui/render.py`, `tui/app.py:_handle_slash` | S | — |
| 2 | `/undo` `/redo` slash commands | `tui/app.py:_handle_slash`, `events/log.py:delete_events_from` | S | — |
| 3 | `/model` `/fast` `/effort` slash commands | `tui/app.py:_handle_slash`, `tui/app.py:_EFFORT` | S | — |
| 4 | `/clear` `/new` slash commands | `tui/app.py:_handle_slash` | S | — |
| 5 | `/cost` `/usage` slash commands | `tui/app.py:_handle_slash`, `tui/render.py:usage_panel` | S | — |
| 6 | `/agents` `/tools` `/mcp` list commands | `tui/app.py:_handle_slash`, `agent/presets.py:186`, `agent/tools.py:ToolRegistry` | S | — |
| 7 | Persist theme choice across sessions | `tui/app.py:_load_config`, `~/.harness/config.json` | S | — |
| 8 | `lo quickstart` polish | `cli/main.py:912-932` | S | — |
| 9 | `lo daemon` (tmux wrapper) | `cli/main.py:938-1014` | S | — |
| 10 | Diff visualization after edits | `tui/render.py:diff_artifact`, `tui/app.py:chat_tool()` | S | — |

**Deliverable:** TUI feels significantly more complete. 10 new slash commands. Theme persists. Diff shown after edits.

### Phase 2: Core Infrastructure (Weeks 2-3)

**Theme:** Foundation for everything else. Config system, LSP, MCP OAuth, plugin hooks.

| # | Item | Files | Effort | Depends On |
|---|------|-------|--------|------------|
| 1 | **`lo.json` config system** | `config/schema.py`, `config/loader.py`, `config/validators.py`, `config-schema.json` | M | — |
| 2 | **LSP integration — manager + diagnostics** | `integrations/lsp/manager.py`, `integrations/lsp/types.py` | L | — |
| 3 | LSP auto-diagnostics after edits | `agent/loop.py`, `integrations/lsp/manager.py` | M | #2 |
| 4 | **MCP OAuth flow** | `integrations/mcp_auth.py`, `integrations/mcp.py` | M | — |
| 5 | MCP resources support | `integrations/mcp.py` (extend MCPClient) | M | — |
| 6 | **Plugin hook system** | `agent/plugin.py` | M | — |
| 7 | doom_loop guard | `guardrails/steps.py`, `guardrails/errors.py` | M | — |
| 8 | Bash-command glob permissions | `agent/permissions.py`, `agent/tools.py:bash_fn` | M | — |
| 9 | `.ignore` file support | `agent/tools.py:grep`, `agent/tools.py:glob` | S | — |

**Deliverable:** Config is unified under `lo.json`. LSP feeds diagnostics to the agent after edits. MCP OAuth unlocks remote SaaS servers. Plugin hooks exist. Security tightened.

### Phase 3: Feature Parity (Weeks 4-6)

**Theme:** Closing the visible feature gaps that matter most to users.

| # | Item | Files | Effort | Depends On |
|---|------|-------|--------|------------|
| 1 | `apply_patch` tool | `agent/tools.py` | M | — |
| 2 | Line-range file read | `agent/tools.py:166` | S | — |
| 3 | Post-edit auto-formatting | `agent/tools.py` (new hook) | M | — |
| 4 | LSP go-to-definition/find-refs tool | `integrations/lsp/tools.py`, `agent/tools.py` | M | Phase 2 #2 |
| 5 | Plan→Implement→Verify lifecycle | `agent/presets.py`, `tui/app.py:667-815` | M | — |
| 6 | Scout agent preset | `agent/presets.py` | S | — |
| 7 | UltraPlan preset | `agent/presets.py` | S | — |
| 8 | Per-agent model override (wire it) | `agent/presets.py:65`, `tui/app.py:_apply_preset` | M | — |
| 9 | Non-blocking subagents | `server/coordinator.py:27` | M | — |
| 10 | Vision / image input | `inference/types.py`, `proxy/engine.py`, `tui/app.py` | M | — |
| 11 | Permission prompt in CLI mode | `cli/main.py:656-657`, `agent/permissions.py` | M | — |
| 12 | External path sandboxing (default) | `sandbox.py:68-107` | S | — |
| 13 | ToolSearch lazy loading (complete) | `agent/loop.py:186-206`, `agent/tools.py:22-43` | M | — |
| 14 | `.claude/skills/` interop | `skills/skill.py` | M | — |
| 15 | `opencode.json` interop | `agent/presets.py`, `integrations/load.py` | M | Phase 2 #1 |

**Deliverable:** Feature parity with both competitors on the coding-agent surface. Patch tool, LSP navigation, vision, plan lifecycle, per-agent models.

### Phase 4: Differentiation (Weeks 7-10)

**Theme:** Ecosystem reach, sharing, CI, documentation, and polish that competitors have and we now match.

| # | Item | Files | Effort | Depends On |
|---|------|-------|--------|------------|
| 1 | **Documentation site** (MkDocs/Astro) | New `docs-site/`, CI workflow | M | — |
| 2 | **Session sharing** (local) | `server/share.py`, `server/templates/share.html`, `tui/app.py:` /share | M | — |
| 3 | **GitHub Action** (`lo review`) | `.github/actions/lo-review/` | L | Phase 3 #5 |
| 4 | **OpenAPI spec** + Python SDK | `server/app.py` → Annotate routes, emit spec | M | — |
| 5 | **Session sharing** (hosted) | Optional cloud service | L | #2 |
| 6 | **Troubleshooting guide** | `docs/troubleshooting.md` | M | — |
| 7 | **Configuration reference** | `docs/configuration.md` | M | Phase 2 #1 |
| 8 | **Changelog** | `CHANGELOG.md` | S | — |
| 9 | **Web UI upgrade** (Electron/desktop) | `server/webui.py` → Static app | L | Phase 2 #1 |
| 10 | **Plugin system** — lifecycle hooks | `agent/plugin.py` (complete) | M | Phase 2 #6 |
| 11 | **SKILL.md ecosystem** | `skills/skill.py` | S | Phase 3 #14 |
| 12 | **`lo ask`** lightweight mode | `cli/main.py`: new command | S | — |
| 13 | **Auto light/dark theme** detection | `tui/app.py:on_mount` | S | Phase 1 #7 |

**Deliverable:** Full ecosystem tool. Shareable sessions, CI integration, published docs, OpenAPI spec, plugin system. Users can switch from Claude Code or OpenCode and find everything they need, plus the local-model advantages none of them have.

---

## Summary of Effort

| Phase | Items | S | M | L | Total Effort |
|-------|-------|---|---|---|-------------|
| Phase 1: Quick Wins | 10 | 10 | 0 | 0 | ~5 days |
| Phase 2: Core Infrastructure | 9 | 2 | 5 | 2 | ~15 days |
| Phase 3: Feature Parity | 14 | 6 | 7 | 1 | ~18 days |
| Phase 4: Differentiation | 12 | 4 | 5 | 3 | ~18 days |
| **Total** | **45** | **22** | **17** | **6** | **~56 days (≈3 months)** |

**The moat remains intact throughout.** Every phase is additive; nothing requires changing the event-sourced spine, logit pipeline, grammar skills, tree search, sandboxing, or capability probing. The work is surface, breadth, and ecosystem — not engine.

---

*This analysis was produced from live codebase exploration of `/home/imjonezz/Desktop/local_harness` and live research of `ccunpacked.dev` and `opencode.ai` docs on 2026-07-08. All file references are relative to the repository root.*
