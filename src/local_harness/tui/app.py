"""Chat-first Textual TUI — a familiar conversation you drive, with the
harness's advantages surfaced as inline badges, live token streaming, and a
working indicator that sits where the next output appears.

Committed turns live in a RichLog (read from the event log, so runs from
`lo run` or the proxy stream in too). The in-progress turn renders live in
a Static beneath it, fed token-by-token from the agent's on_token callback.

A persistent status bar (OpenCode-style) sits above the input and always shows
the harness's edge — preset · tier + ability glyphs · $0.00 spent / ~$ saved vs a
frontier API · ⟲ determinism · ⚛ background-learning state. A compact collapsible
banner (Hermes-style) lists what the server unlocks, plus loaded tools/skills/
memory. ^u opens the full usage & advantages overlay; ^g forks N candidate plans.

Keys:  ^t history · ^u usage · ^r replay · ^g plan-fork · ^o connect · ^c quit
Slash: /new or /clear to start a fresh conversation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time

from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Collapsible,
    DataTable,
    Footer,
    Input,
    Label,
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from ..agent.loop import Agent
from ..agent.tools import ToolRegistry, builtin_tools
from ..events.log import (
    AGENT_SPAWNED,
    CONTEXT_COMPACTED,
    MESSAGE_SNIPPED,
    MODEL_CALL,
    POLICY_TRIGGERED,
    RUN_COMPLETED,
    RUN_FAILED,
    TOOL_CALL,
    USER_MESSAGE,
    EventLog,
)
from ..events.replay import replay_run
from ..inference.capabilities import Capabilities, probe
from ..inference.client import OpenAICompatClient
from . import render

# Color constants (render.C_OK, render.C_DIM, …) are read as `render.C_*` so a /theme switch,
# which calls render.set_palette(), recolors app-side chrome too (live area,
# menus, modals) — not just the transcript. Don't import them by value here.
from .render import (
    ability_glyphs,
    banner_body,
    chat_render_event,
    frontier_saved,
    status_bar,
    status_text,
    tokens_of,
    usage_panel,
    welcome_panel,
)

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _theme_from_palette(p: render.Palette) -> Theme:
    """Build a Textual Theme (the chrome: bg/surface/panel/accent) from a render
    Palette, so the chrome and the transcript hues stay one coherent skin."""
    return Theme(
        name=p.name,
        primary=p.jade,
        secondary=p.sakura,
        accent=p.gold,
        foreground=p.cream,
        background=p.bg,
        surface=p.surface,
        panel=p.panel,
        success=p.jade,
        warning=p.gold,
        error=p.rose,
        dark=p.dark,
        variables={
            "border": p.jade_deep,
            "block-cursor-background": p.jade,
            "block-cursor-foreground": p.ink,
            "input-cursor-background": p.jade,
            "input-selection-background": p.jade_deep + " 35%",
            "footer-key-foreground": p.gold,
            "footer-description-foreground": p.cream,
            "scrollbar": p.jade_deep,
            "scrollbar-hover": p.jade,
        },
    )


# Where the chosen theme is remembered across sessions.
_CONFIG_PATH = os.path.expanduser("~/.harness/config.json")

_PRESET_BLURB = {
    "build": "full toolset; writes/shell/web ask, reads free",
    "plan": "read & search only; produces a plan, edits denied",
    "explore": "read & search to understand; nothing else",
    "general": "like build, a bit more exploratory",
}


def _preset_blurb(name: str) -> str:
    return _PRESET_BLURB.get(name, "")


def _findings_skill(kind: str):
    """A grammar skill that constrains review output to a valid findings list:
    `<file>:<line> — <severity> — <issue>` per line, or exactly 'No findings.'."""
    from ..structured import choice, lit, one_or_more, regex, seq, zero_or_more

    sevs = (
        ["critical", "high", "medium", "low"]
        if kind == "security-review"
        else ["blocker", "major", "minor", "nit"]
    )
    path = one_or_more(regex(r"[^\s:]"))
    num = one_or_more(regex(r"[0-9]"))
    desc = one_or_more(regex(r"[^\n]"))
    finding = seq(path, lit(":"), num, lit(" — "), choice(*sevs), lit(" — "), desc)
    listing = seq(finding, zero_or_more(seq(lit("\n"), finding)))
    return choice(lit("No findings."), listing).skill(f"{kind}-findings")


# /effort tunes two knobs for subsequent turns: how much the model reasons
# (steered by a system-prompt suffix) and whether low-confidence steps are
# resampled (a StepPolicy threshold; None = off). /fast == low.
_EFFORT = {
    "low": {
        "resample": None,
        "blurb": "brief thinking, single pass",
        "think": "Keep your reasoning brief and reach the answer quickly.",
    },
    "medium": {"resample": None, "blurb": "balanced (the default)", "think": ""},
    "high": {
        "resample": -1.0,
        "blurb": "think thoroughly, resample unsure steps",
        "think": "Think carefully and thoroughly, and double-check before you answer.",
    },
}


# In plan mode we steer behaviour through the *user* turn (not just the preset's
# system prompt) so it holds whether the agent runs in-process or behind the
# embedded session server: produce a plan, don't implement yet.
_PLAN_INSTRUCTION = (
    "Produce a detailed, step-by-step implementation plan for the task below. "
    "Think it through carefully, but DO NOT implement anything yet — output only "
    "the plan as Markdown: numbered steps, files to create/modify, and the key "
    "decisions and trade-offs.\n\nTask: "
)

# Live advantage demos surfaced as slash-commands (see tui/advantages.py). Kept as a
# literal so dispatch needs no import; the worker imports the implementations lazily.
_ADVANTAGE_NAMES = frozenset(
    {
        "grammar",
        "samplers",
        "antislop",
        "overlay",
        "consistency",
        "escalate",
        "bestof",
        "thinkbudget",
    }
)


def _caps_from_health(health: dict) -> Capabilities | None:
    """Reconstruct Capabilities from a `lo serve` /health payload so the
    status bar / banner show the server's real tier and feature flags."""
    caps_d = (health or {}).get("capabilities") or {}
    fields = {f.name for f in dataclasses.fields(Capabilities)}
    kw = {k: v for k, v in caps_d.items() if k in fields}
    if isinstance(kw.get("sampler_zoo"), list):
        kw["sampler_zoo"] = set(kw["sampler_zoo"])
    return Capabilities(**kw) if kw else None


class ConnectScreen(ModalScreen[tuple[str, str, str] | None]):
    """Modal to point the harness at a different provider/endpoint."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, url: str, model: str) -> None:
        super().__init__()
        self._url, self._model = url, model

    def compose(self) -> ComposeResult:
        with Vertical(id="connectbox"):
            yield Label("Connect to a provider", id="connecttitle")
            yield Input(
                value=self._url,
                placeholder="base URL (e.g. https://api.openai.com)",
                id="c_url",
            )
            yield Input(
                value=self._model,
                placeholder="model — leave blank to pick from the server's list",
                id="c_model",
            )
            yield Input(
                placeholder="API key — optional (for hosted providers)",
                password=True,
                id="c_key",
            )
            yield Label(
                "Enter to connect (then pick a model) · Esc to cancel", id="connecthint"
            )

    def on_mount(self) -> None:
        self.query_one("#c_url", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()  # don't let it bubble to the app's prompt handler
        self.dismiss(
            (
                self.query_one("#c_url", Input).value.strip(),
                self.query_one("#c_model", Input).value.strip(),
                self.query_one("#c_key", Input).value.strip(),
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class ModelPickerScreen(ModalScreen[str | None]):
    """Pick from the models the endpoint actually serves (/v1/models) — no more
    guessing the exact id. Auto-used when a server lists more than one model."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, models: list[str]) -> None:
        super().__init__()
        self._models = models

    def compose(self) -> ComposeResult:
        from textual.widgets import OptionList

        with Vertical(id="pickbox"):
            yield Label(
                f"{len(self._models)} models available — choose one", id="picktitle"
            )
            yield OptionList(*self._models, id="modellist")
            yield Label("↑↓ choose · enter select · esc cancel", id="pickhint")

    def on_mount(self) -> None:
        from textual.widgets import OptionList

        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event) -> None:
        self.dismiss(str(event.option.prompt))

    def action_cancel(self) -> None:
        self.dismiss(None)


class PermissionModal(ModalScreen[bool]):
    """Approve or deny a tool call (the 'ask' tier of the permission system)."""

    BINDINGS = [
        ("y", "approve", "Allow"),
        ("n", "reject", "Deny"),
        ("escape", "reject", "Deny"),
    ]

    def __init__(self, tool: str, args: str) -> None:
        super().__init__()
        self._tool, self._args = tool, args

    def compose(self) -> ComposeResult:
        with Vertical(id="permbox"):
            yield Label(f"Allow  {self._tool}?", id="permtitle")
            yield Static(self._args[:300] or "(no arguments)", id="permargs")
            yield Label("y allow (for this session) · n deny", id="permhint")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)


class OverlayScreen(ModalScreen[None]):
    """A scrollable modal for a Rich renderable (usage panel, help, etc.) — so long
    content scrolls instead of running off the top like a chat-log write would."""

    BINDINGS = [
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
        ("up", "scroll_up", ""),
        ("down", "scroll_down", ""),
        ("pageup", "page_up", ""),
        ("pagedown", "page_down", ""),
    ]

    def __init__(self, renderable, hint: str = "↑↓ scroll · esc to close") -> None:
        super().__init__()
        self._renderable = renderable
        self._hint = hint

    def compose(self) -> ComposeResult:
        with Vertical(id="overlaybox"):
            with VerticalScroll(id="overlaybody"):
                yield Static(self._renderable)
            yield Label(self._hint, id="overlayhint")

    def on_mount(self) -> None:
        self.query_one("#overlaybody").focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_scroll_up(self) -> None:
        self.query_one("#overlaybody", VerticalScroll).scroll_up()

    def action_scroll_down(self) -> None:
        self.query_one("#overlaybody", VerticalScroll).scroll_down()

    def action_page_up(self) -> None:
        self.query_one("#overlaybody", VerticalScroll).scroll_page_up()

    def action_page_down(self) -> None:
        self.query_one("#overlaybody", VerticalScroll).scroll_page_down()


class RewindScreen(ModalScreen[int | None]):
    """Pick a point to roll the conversation back to. Selecting a row removes that
    point and everything after it (the removed tail is archived, so it's reversible)."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, points: list[tuple[int, str, str]]) -> None:
        super().__init__()
        self._points = points  # (seq, kind, preview)

    def compose(self) -> ComposeResult:
        from textual.widgets import OptionList

        with Vertical(id="rewindbox"):
            yield Label(
                "Rewind — remove from the chosen point onward (tail archived)",
                id="rewindtitle",
            )
            yield OptionList(id="rewindlist")
            yield Label("↑↓ choose · enter rewind · esc cancel", id="rewindhint")

    def on_mount(self) -> None:
        from textual.widgets import OptionList

        ol = self.query_one(OptionList)
        for seq, kind, preview in self._points:
            label = (
                "the original answer" if kind == "answer" else f"follow-up: {preview}"
            )
            ol.add_option(
                Option(
                    Text.assemble(
                        ("✂ ", "bold " + render.C_RESAMPLE), (label, render.C_ANSWER)
                    ),
                    id=str(seq),
                )
            )
        if self._points:
            ol.highlighted = len(self._points) - 1  # default to the most recent point
        ol.focus()

    def on_option_list_option_selected(self, event) -> None:
        self.dismiss(int(event.option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


class PromptInput(Input):
    """The prompt box, with optional vim modal editing. Intercepts keys before the
    Input inserts them: in vim normal mode the app consumes motions/edits; in
    insert mode (or with vim off) everything falls through to normal typing."""

    async def _on_key(self, event) -> None:
        app = self.app
        if getattr(app, "_vim_enabled", False) and app._vim_key(self, event):
            event.prevent_default()
            event.stop()
            return
        await super()._on_key(event)


class SnipScreen(ModalScreen[int | None]):
    """Pick a turn to snip (collapse) — frees context surgically. Lossless: the
    original stays in the log, only the context projection shrinks."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, points: list[tuple[int, str, int]]) -> None:
        super().__init__()
        self._points = points  # (seq, label, size_chars)

    def compose(self) -> ComposeResult:
        from textual.widgets import OptionList

        with Vertical(id="snipbox"):
            yield Label(
                "Snip — collapse a turn to free context (lossless)", id="sniptitle"
            )
            yield OptionList(id="sniplist")
            yield Label("↑↓ choose · enter snip · esc cancel", id="sniphint")

    def on_mount(self) -> None:
        from textual.widgets import OptionList

        ol = self.query_one(OptionList)
        for seq, label, size in self._points:
            ol.add_option(
                Option(
                    Text.assemble(
                        ("✂ ", "bold " + render.C_RESAMPLE),
                        (label, render.C_ANSWER),
                        (f"  (~{render._ktok(size // 4)} tok)", render.C_DIM),
                    ),
                    id=str(seq),
                )
            )
        if self._points:
            ol.highlighted = 0
        ol.focus()

    def on_option_list_option_selected(self, event) -> None:
        self.dismiss(int(event.option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ThemePickerScreen(ModalScreen[str | None]):
    """Pick a theme, with live preview as you arrow through. Enter keeps it, Esc
    restores the one you started on."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, names: list[str], current: str) -> None:
        super().__init__()
        self._names = names
        self._current = current

    def compose(self) -> ComposeResult:
        from textual.widgets import OptionList

        with Vertical(id="themebox"):
            yield Label("Theme — ↑↓ to preview, enter to keep", id="themetitle")
            yield OptionList(id="themelist")
            yield Label("enter select · esc cancel", id="themehint")

    def on_mount(self) -> None:
        from textual.widgets import OptionList

        ol = self.query_one(OptionList)
        for i, name in enumerate(self._names):
            mark = "▸ " if name == self._current else "  "
            ol.add_option(Option(f"{mark}{name}", id=name))
            if name == self._current:
                ol.highlighted = i
        ol.focus()

    def on_option_list_option_highlighted(self, event) -> None:
        self.app._preview_theme(event.option.id)  # live chrome preview

    def on_option_list_option_selected(self, event) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.app._preview_theme(self._current)  # restore what they started on
        self.dismiss(None)


class HarnessApp(App):
    TITLE = "local_harness"
    CSS = """
    #banner { background: $surface; border-bottom: solid $accent; padding: 0 1; }
    #banner_body { padding: 1 1 0 1; }
    #main { height: 1fr; }
    #runs { width: 38; display: none; border-right: solid $accent; }
    #runs.visible { display: block; }
    #chat { height: 1fr; padding: 1 2; scrollbar-gutter: stable; }
    #live { height: auto; max-height: 14; padding: 0 2; border-top: solid $panel; }
    #statusbar { height: 1; padding: 0 2; background: $panel; color: $text; }
    #slashmenu { display: none; height: auto; max-height: 9; margin: 0 1;
                 border: tall $accent; background: $surface; }
    #slashmenu.visible { display: block; }
    Input { border: tall $accent; margin: 0 1; }
    Input.vim-normal { border: tall $warning; }
    ConnectScreen { align: center middle; }
    #connectbox { width: 76; height: auto; padding: 1 2; border: thick $accent; background: $surface; }
    #connecttitle { text-style: bold; padding-bottom: 1; }
    #connecthint { color: $text-muted; padding-top: 1; }
    ModelPickerScreen { align: center middle; }
    #pickbox { width: 76; height: auto; max-height: 80%; padding: 1 2; border: thick $accent; background: $surface; }
    #picktitle { text-style: bold; padding-bottom: 1; }
    #modellist { height: auto; max-height: 16; border: tall $accent; }
    #pickhint { color: $text-muted; padding-top: 1; }
    ThemePickerScreen { align: center middle; }
    #themebox { width: 56; height: auto; max-height: 80%; padding: 1 2; border: thick $accent; background: $surface; }
    #themetitle { text-style: bold; padding-bottom: 1; }
    #themelist { height: auto; max-height: 16; border: tall $accent; }
    #themehint { color: $text-muted; padding-top: 1; }
    SnipScreen { align: center middle; }
    #snipbox { width: 84; height: auto; max-height: 80%; padding: 1 2; border: thick $accent; background: $surface; }
    #sniptitle { text-style: bold; padding-bottom: 1; }
    #sniplist { height: auto; max-height: 16; border: tall $accent; }
    #sniphint { color: $text-muted; padding-top: 1; }
    RewindScreen { align: center middle; }
    #rewindbox { width: 80; height: auto; max-height: 80%; padding: 1 2; border: thick $accent; background: $surface; }
    #rewindtitle { text-style: bold; padding-bottom: 1; }
    #rewindlist { height: auto; max-height: 16; border: tall $accent; }
    #rewindhint { color: $text-muted; padding-top: 1; }
    PermissionModal { align: center middle; }
    #permbox { width: 72; height: auto; padding: 1 2; border: thick $accent; background: $surface; }
    #permtitle { text-style: bold; }
    #permargs { color: $text-muted; padding: 1 0; }
    #permhint { color: $text-muted; }
    OverlayScreen { align: center middle; }
    #overlaybox { width: 92; max-width: 96%; height: 80%; padding: 0 1 1 1; background: $surface; border: thick $accent; }
    #overlaybody { height: 1fr; padding: 1 1 0 1; }
    #overlayhint { color: $text-muted; padding: 1 1 0 1; }
    """
    BINDINGS = [
        Binding("ctrl+c", "confirm_quit", "Quit", priority=True),  # press twice to exit
        Binding("escape", "interrupt", "Stop turn"),
        Binding("ctrl+t", "toggle_runs", "History"),
        Binding("ctrl+u", "usage", "Usage"),
        Binding("ctrl+r", "replay", "Replay"),
        Binding("ctrl+g", "plan_fork", "Plan-fork"),
        Binding("ctrl+o", "connect", "Connect"),
        Binding("ctrl+y", "toggle_thinking", "Thinking", show=False),
        Binding("pageup", "scroll_chat_up", "Scroll ↑", show=False),
        Binding("pagedown", "scroll_chat_down", "Scroll ↓", show=False),
        Binding("ctrl+home", "scroll_chat_top", "Top", show=False),
        Binding("ctrl+end", "scroll_chat_bottom", "Bottom", show=False),
        # delete the highlighted history run — `delete` for full keyboards, `ctrl+d`
        # for laptops (MacBooks) that have no forward-delete key.
        Binding("delete", "delete_run", "Delete run", show=False),
        Binding("ctrl+d", "delete_run", "Delete run", show=False),
    ]

    def __init__(
        self,
        client: OpenAICompatClient,
        db_path: str,
        *,
        max_steps: int = 20,
        use_guardrails: bool = True,
        required_steps: list[str] | None = None,
        terminal_tools: frozenset[str] = frozenset(),
        context_budget: int | None = None,
        skills_dir: str | None = None,
        tools_config: dict | None = None,
        resample_threshold: float | None = None,
        memory_dir: str = ".harness/memory",
        allow_all: bool = False,
        preset: str = "build",
        background: bool = False,
        sandbox: str = "host",
        server: str | None = None,
        shared_db: bool = False,
    ):
        super().__init__()
        from ..agent.presets import get_preset

        # When set, this TUI is a thin CLIENT of a `lo serve` instance: it
        # POSTs sessions to the server and renders the SSE event stream, instead
        # of building and driving an Agent in-process. The default (None) is the
        # original in-process path (kept as the --in-process escape hatch).
        self.server = server.rstrip("/") if server else None
        self._upstream = None  # server mode: the upstream URL reported by /health
        # shared_db: the server writes to the SAME db this TUI polls (embedded,
        # local). Then persisted events arrive via the poll, so we DON'T mirror
        # them from the SSE stream (which would duplicate); we use SSE only for
        # live token/tool deltas. For a remote --server (different db), we mirror.
        self._shared_db = shared_db
        self.resample_threshold = resample_threshold
        self.allow_all = allow_all
        self.background_enabled = background
        self._sandbox_kind = sandbox
        self._sandbox = None  # built at probe time (fail-closed if microvm unavailable)
        self._cycle_running = False
        self._preset = get_preset(preset)
        self.client = client
        self.db_path = db_path
        self.skills_dir = skills_dir
        self.memory_dir = memory_dir
        self.notebook = None
        self._memory = None
        self.tools_config = tools_config
        self._tool_registry = None  # builtins + UTCP/MCP, built once at startup
        self._skills = None
        self.max_steps = max_steps
        self.use_guardrails = use_guardrails
        self.required_steps = required_steps or []
        self.terminal_tools = terminal_tools
        self.context_budget = context_budget
        self.caps: Capabilities | None = None
        self.active: str | None = None
        self._caps_ready = asyncio.Event()
        self._follow = True
        self._blank = False  # /new or /clear: hold a blank slate
        self._rendered = 0
        self._runs_state: list[tuple] = []
        self._run_ids: list[str] = []
        self._status_base: str | None = None
        self._spin = 0
        self._welcomed = False
        self._live_content = ""  # in-progress streamed answer
        self._live_reasoning = ""  # in-progress streamed thinking
        self._live_ghost = ""  # a resampled (rejected) attempt, ghosted
        self._live_streams: list[dict] | None = (
            None  # N concurrent rollouts (fan-out view)
        )
        self._live_fan_title = ""
        self._streaming = False
        self._running_tool: str | None = (
            None  # a tool currently executing (live indicator)
        )
        # persistent status-bar / usage state (the ability surface)
        self._stats_dirty = True
        self._stats = {
            "calls": 0,
            "tokens": 0,
            "saved": 0.0,
            "conf": [0, 0, 0],
            "mean_lp": None,
            "resamples": 0,
        }
        self._learn_summary: str | None = None
        self._banner_built = False
        self._show_thinking = True  # ^y toggles full reasoning in the transcript
        self._banner_collapsed_once = False
        # plan-mode flow: a produced plan is written to a temp file, presented as an
        # artifact, and can be /editor-edited, refined by chatting, or /approve-d
        # (→ switch to build and implement). _expect_plan marks a turn as a plan run.
        self._plan_pending = False
        self._plan_file: str | None = None
        self._expect_plan = False
        # /context gauge + proactive warnings (computed from the latest MODEL_CALL).
        self._ctx: tuple[str, float | None] | None = (
            None  # (label, frac) for the status bar
        )
        self._ctx_warn_level = (
            0  # highest % threshold warned (70, 80); reset on compaction
        )
        self._effort = (
            "medium"  # /effort | /fast — reasoning depth + resample aggressiveness
        )
        self._vim_enabled = False  # /vim — modal editing on the prompt
        self._vim_mode = "insert"  # insert | normal
        self._vim_pending = ""  # for two-key ops like dd
        self._last_quit = 0.0  # ^C-twice-to-exit timestamp
        self._active_worker = (
            None  # the in-flight run worker (for /stop · Esc interrupt)
        )
        self._code_mode = (
            True  # /codemode — model writes code calling tools (default on)
        )

    def compose(self) -> ComposeResult:
        with Collapsible(
            title="local_harness  ·  connecting…", id="banner", collapsed=False
        ):
            yield Static("probing the server…", id="banner_body")
        with Horizontal(id="main"):
            yield DataTable(id="runs")
            yield RichLog(id="chat", wrap=True, markup=False)
        yield Static(id="live")
        yield Static(id="statusbar")
        yield OptionList(id="slashmenu")
        yield PromptInput(
            placeholder="Ask the harness  ·  /help for commands  ·  ^u usage  ·  ^o connect",
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        try:
            for palette in render.THEMES.values():
                self.register_theme(_theme_from_palette(palette))
            cfg = self._load_config()
            saved = cfg.get("theme", render.DEFAULT_THEME)
            self._apply_theme(saved if saved in render.THEMES else render.DEFAULT_THEME)
            self._vim_enabled = bool(cfg.get("vim", False))
        except Exception:
            pass
        self.event_log = EventLog(self.db_path)
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("run", "status", "task")
        self.sub_title = "connecting…"
        self.query_one(Input).focus()
        self.run_worker(self._probe_worker(auto_connect_on_fail=True), exclusive=False)
        self.set_interval(0.1, self._tick)
        self._tick()

    async def on_unmount(self) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.aclose()  # stop the microVM
            except Exception:
                pass

    def get_system_commands(self, screen):  # adds to the ^p command palette
        yield from super().get_system_commands(screen)
        yield SystemCommand(
            "Connect to provider…", "Switch URL / model / API key", self.action_connect
        )
        yield SystemCommand(
            "New conversation", "Clear the screen and start fresh", self.action_new
        )
        yield SystemCommand(
            "Usage & advantages",
            "Cost saved, determinism, confidence",
            self.action_usage,
        )
        yield SystemCommand(
            "Context window",
            "What's in context and how full (/context)",
            self.action_context,
        )
        yield SystemCommand(
            "Memory",
            "USER / MEMORY / PROJECT notes + recall (/memory)",
            self.action_memory,
        )
        yield SystemCommand(
            "Code review",
            "Review the git diff (read-only)",
            lambda: self.action_review("review"),
        )
        yield SystemCommand(
            "Security review",
            "Security-review the git diff",
            lambda: self.action_review("security-review"),
        )
        yield SystemCommand(
            "Rewind",
            "Roll back to an earlier point (tail archived)",
            self.action_rewind,
        )
        yield SystemCommand(
            "Snip", "Collapse a turn to free context (lossless)", self.action_snip
        )
        yield SystemCommand(
            "Theme", "Switch theme (live preview, remembered)", self.action_theme
        )
        yield SystemCommand(
            "Cost", "What this session would cost on a frontier API", self.action_cost
        )
        yield SystemCommand(
            "Export transcript", "Write run-<id>.md to the cwd", self._export_transcript
        )
        yield SystemCommand(
            "Switch model", "Pick a model the endpoint serves", self.action_model
        )
        yield SystemCommand(
            "Agents (team)", "Lead + spawned workers (/agents)", self.action_agents
        )
        yield SystemCommand(
            "Tool sources (MCP)", "Show configured MCP/UTCP sources", self.action_mcp
        )
        yield SystemCommand(
            "Vim mode", "Toggle modal editing on the prompt", self.action_vim
        )
        yield SystemCommand(
            "Plan-fork", "Fork N candidate plans, pick the best", self.action_plan_fork
        )
        yield SystemCommand(
            "Editor", "Edit the plan / transcript in $EDITOR", self._editor_command
        )
        yield SystemCommand(
            "Approve plan",
            "Switch to build mode and implement the plan",
            self._approve_plan,
        )
        yield SystemCommand(
            "Run last code block", "Execute it and show the output", self._run_last_code
        )
        yield SystemCommand(
            "Preview HTML",
            "Open the last HTML block in the browser",
            self._preview_last,
        )

    # --- workers ---------------------------------------------------------

    async def _probe_worker(self, auto_connect_on_fail: bool = False) -> None:
        if self.server is not None:
            await self._probe_server(auto_connect_on_fail)
            return
        try:
            from ..integrations.load import registry_with_sources
            import os as _os
            from ..sandbox import make_sandbox, SandboxUnavailable

            try:
                self._sandbox = make_sandbox(self._sandbox_kind, _os.getcwd())
            except SandboxUnavailable as e:
                self.notify(f"sandbox: {e}", severity="error", timeout=12)
                self._sandbox = None
                self.sub_title = f"sandbox unavailable — {e}"
                return
            if self._sandbox is not None and self._sandbox.kind != "host":
                self.notify(
                    f"tools run inside a {self._sandbox.kind} sandbox "
                    "(workdir mounted, host isolated)",
                    timeout=8,
                )
            self._tool_registry = await registry_with_sources(
                self.tools_config, sandbox=self._sandbox
            )
            # self-editing memory: the notebook + memory/session_search tools
            from ..agent.memory import Memory
            from ..agent.notebook import Notebook, memory_tool, session_search_tool
            from pathlib import Path

            Path(self.memory_dir).mkdir(parents=True, exist_ok=True)
            # PROJECT.md lives in the cwd's .harness/, so it travels with the codebase.
            self.notebook = Notebook(
                self.memory_dir, project_dir=str(Path.cwd() / ".harness")
            )
            self._memory = Memory(Path(self.memory_dir) / "memory.db")
            self._tool_registry.register(memory_tool(self.notebook))
            self._tool_registry.register(session_search_tool(self._memory))
            self._apply_preset()
        except Exception as e:
            self.notify(f"tool sources error: {e}", severity="error")
        try:
            if not self.client.model:
                models = await self.client.list_models()
                self.client.model = models[0] if models else ""
                if len(models) > 1:
                    self.notify(
                        f"{len(models)} models here — using {self.client.model}; "
                        "^o to pick another",
                        timeout=8,
                    )
            self.caps = await probe(self.client)
        except Exception as e:
            self.sub_title = f"upstream error: {e}"
            self._set_banner_title(
                f"local_harness  ·  can't reach {self.client.base_url} — ^o to connect"
            )
            self.notify(
                f"can't reach {self.client.base_url}: {e}", severity="error", timeout=10
            )
            if (
                auto_connect_on_fail
            ):  # surface the connect modal so the default isn't a dead end
                self.action_connect()
            return
        self._status_base = f"{self.client.model} · tier {self.caps.tier()}"
        self.sub_title = self._status_base
        if self.caps.logprobs and not self.caps.stream_logprobs:
            # front the quirk honestly rather than hide the re-pass
            self.notify(
                "This engine can't stream confidence alongside tool-calling — "
                "it's filled in from a deterministic re-pass.",
                timeout=8,
            )
        self._refresh_banner()
        self._update_status()
        self._maybe_welcome()
        self._caps_ready.set()

    # --- server (--server) client mode -----------------------------------

    async def _probe_server(self, auto_connect_on_fail: bool = False) -> None:
        """Connect to a `lo serve` instance: read capabilities from /health,
        skip all in-process agent infra (the server owns tools/sandbox/memory).
        The server's own upstream probe may still be running (≈30s on a 27B), so
        poll /health until capabilities land before declaring ready."""
        import httpx

        health = None
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                for i in range(60):  # ≤~60s for the server to finish probing upstream
                    health = (await c.get(f"{self.server}/health")).json()
                    # error set = probe finished and failed; don't poll the full 60s
                    if health.get("capabilities") or health.get("error"):
                        break
                    if i == 0:
                        self.sub_title = f"⇄ {self.server} · server probing upstream…"
                    await asyncio.sleep(1)
        except Exception as e:
            self.sub_title = f"can't reach server {self.server}: {e}"
            self._set_banner_title(f"local_harness  ·  can't reach {self.server}")
            self.notify(
                f"can't reach lo serve at {self.server}: {e}",
                severity="error",
                timeout=10,
            )
            if auto_connect_on_fail:
                self.action_connect()
            return
        self._upstream = health.get("upstream")
        if not health.get("capabilities"):
            self.notify(
                f"server reached but its upstream {self._upstream} is unavailable "
                f"({health.get('error')}) — ^o to connect to your model server",
                severity="warning",
                timeout=12,
            )
            if auto_connect_on_fail:
                self.action_connect()
        self.caps = _caps_from_health(health)
        model = health.get("model", "?")
        tier = self.caps.tier() if self.caps else "?"
        self._status_base = f"⇄ {self.server} · {model} · tier {tier}"
        self.sub_title = self._status_base
        self._set_banner_title(f"local_harness  ·  client of {self.server}")
        self._refresh_banner()
        self._update_status()
        self._maybe_welcome()
        self._caps_ready.set()

    async def _server_submit(self, text: str) -> None:
        """Start or continue a session on the server, then follow its stream."""
        import httpx

        if self.caps is None:
            # Upstream was down at startup — maybe it's up now: one re-probe
            # before giving up (covers "started llama.cpp after launching the TUI").
            try:
                async with httpx.AsyncClient(timeout=90) as c:
                    (await c.post(f"{self.server}/connect", json={})).raise_for_status()
            except Exception:
                pass
            await self._probe_server()
            if self.caps is None:
                self.notify(
                    f"upstream {self._upstream} still unreachable — ^o to connect",
                    severity="error",
                    timeout=10,
                )
                return

        preset = self._preset.name  # let the server build the agent for this preset
        cm = self._code_mode
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                if (
                    self.active
                    and not self._blank
                    and self._active_status() in ("completed", "failed")
                ):
                    await c.post(
                        f"{self.server}/session/{self.active}/message",
                        json={"content": text, "preset": preset, "code_mode": cm},
                    )
                    run_id = self.active
                else:
                    r = await c.post(
                        f"{self.server}/session",
                        json={"task": text, "preset": preset, "code_mode": cm},
                    )
                    r.raise_for_status()
                    run_id = r.json()["run_id"]
                    if not self._shared_db:  # remote: seed the local mirror's run row
                        self.event_log.ensure_run(run_id, text)
                    self._blank = False
                    self._follow = True
                    self._set_active(run_id)
        except Exception as e:
            self.notify(f"server error: {e}", severity="error")
            return
        self.run_worker(self._server_stream(run_id), exclusive=False)

    async def _server_stream(self, run_id: str) -> None:
        """Subscribe to the session's SSE stream; mirror persisted events into the
        local log (so _tick/_render_pending render them) and feed ephemeral token /
        tool deltas into the same live-render hooks the in-process path uses."""
        import httpx

        url = f"{self.server}/session/{run_id}/events?replay=1&once=1"
        try:
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream("GET", url) as resp:
                    resp.raise_for_status()
                    etype = None
                    async for line in resp.aiter_lines():
                        if line.startswith("event:"):
                            etype = line.split(":", 1)[1].strip()
                        elif line.startswith("data:"):
                            data = json.loads(line[5:].strip())
                            self._ingest_server_event(
                                run_id, etype, data.get("payload", {})
                            )
        except Exception as e:
            self.notify(f"stream ended: {e}", severity="warning")

    def _ingest_server_event(self, run_id: str, etype: str, payload: dict) -> None:
        if etype == "token_delta":
            if not self._streaming:
                self._on_token("start", "")
            self._on_token("content", payload.get("text", ""))
        elif etype == "reasoning_delta":
            if not self._streaming:
                self._on_token("start", "")
            self._on_token("reasoning", payload.get("text", ""))
        elif etype == "tool_progress":
            self._on_tool_event(payload.get("name", ""), payload.get("phase", ""))
        elif etype == "notice":
            self.notify(payload.get("message", ""), severity="warning", timeout=8)
        elif etype == "permission_request":
            # the server is asking us to approve an ask-tier tool — show the modal
            # and POST the decision back, exactly like the in-process approver.
            self.run_worker(self._handle_permission(run_id, payload), exclusive=False)
        elif not self._shared_db:  # remote server: mirror persisted events locally
            if etype == "run_started":
                self.event_log.ensure_run(run_id, payload.get("task", ""))
            else:
                self.event_log.import_event(run_id, etype, payload)
        # shared_db (embedded): persisted events are already in the db we poll —
        # don't mirror them (that would duplicate). SSE here is only the live feed.

    async def _handle_permission(self, run_id: str, payload: dict) -> None:
        """Show the permission modal for a server-side ask-tier tool, then POST the
        decision back so the server's approver resumes (or denies)."""
        import httpx

        request_id = payload.get("request_id", "")
        approved = await self.push_screen_wait(
            PermissionModal(payload.get("tool", ""), payload.get("arguments", ""))
        )
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                await c.post(
                    f"{self.server}/session/{run_id}/permission",
                    json={"request_id": request_id, "approved": bool(approved)},
                )
        except Exception as e:  # noqa: BLE001
            self.notify(f"permission post failed: {e}", severity="error")

    def _set_banner_title(self, title: str) -> None:
        from textual.css.query import NoMatches

        try:
            self.query_one("#banner", Collapsible).title = title
        except NoMatches:
            pass

    def _refresh_banner(self) -> None:
        """Populate the Hermes-style banner: capabilities, tools, skills, memory."""
        if self.caps is None:
            return
        from textual.css.query import NoMatches

        tools = [
            s["function"]["name"]
            for s in (self._tool_registry.schemas() if self._tool_registry else [])
        ]
        skills = sorted(self._skill_names())
        mem = self._memory_summary()
        try:
            self.query_one("#banner_body", Static).update(
                banner_body(
                    self.client.model,
                    self.caps,
                    tools=tools,
                    skills=skills,
                    memory_summary=mem,
                    preset_name=self._preset.name,
                    preset_blurb=_preset_blurb(self._preset.name),
                )
            )
        except NoMatches:
            pass
        self._set_banner_title(
            f"local_harness  ·  {self.client.model}  ·  tier {self.caps.tier()}  ·  "
            f"{self._preset.name}  ·  $0.00"
        )
        self._banner_built = True

    def _memory_summary(self) -> str:
        bits: list[str] = []
        try:
            if self.notebook is not None:
                text = (self.notebook.memory.read() or "").strip()
                facts = sum(
                    1 for ln in text.splitlines() if ln.strip().startswith(("-", "*"))
                )
                if facts:
                    bits.append(f"{facts} notes")
        except Exception:
            pass
        try:
            if self._memory is not None:
                n = self._memory.count()
                if n:
                    bits.append(f"{n} memories")
        except Exception:
            pass
        return " · ".join(bits)

    def _learn_state(self) -> str:
        if not self.background_enabled:
            return "off"
        return "learning" if self._cycle_running else "idle"

    def _recompute_stats(self) -> None:
        """Sum tokens, frontier-equivalent cost, confidence bands and resamples
        across the whole session — the data behind the status bar and ^u panel."""
        tin = tout = calls = resamples = 0
        bands = [0, 0, 0]
        mean_lps: list[float] = []
        for r in self.event_log.runs():
            for ev in self.event_log.events(r.run_id):
                if ev.type == MODEL_CALL:
                    calls += 1
                    a, b = tokens_of(ev.payload)
                    tin += a
                    tout += b
                    s = ev.payload.get("logprob_summary")
                    if s and s.get("mean_logprob") is not None:
                        mean_lps.append(s["mean_logprob"])
                    choice = (ev.payload.get("response") or {}).get("choices", [{}])[0]
                    for tok in (choice.get("logprobs") or {}).get("content") or []:
                        lp = tok.get("logprob", 0.0) if isinstance(tok, dict) else 0.0
                        bands[0 if lp > -0.3 else 1 if lp > -1.5 else 2] += 1
                elif ev.type == POLICY_TRIGGERED:
                    resamples += 1
        self._stats = {
            "calls": calls,
            "tokens": tin + tout,
            "saved": frontier_saved(tin, tout),
            "conf": bands,
            "mean_lp": (sum(mean_lps) / len(mean_lps)) if mean_lps else None,
            "resamples": resamples,
        }
        self._recompute_context()

    def _context_snapshot(self) -> tuple[list[dict] | None, list[dict] | None]:
        """(messages, tools) from the active run's most recent MODEL_CALL request
        body — exactly what's currently in the model's context."""
        if not self.active:
            return None, None
        calls = self.event_log.events(self.active, type=MODEL_CALL)
        if not calls:
            return None, None
        body = calls[-1].payload.get("request_body") or {}
        return body.get("messages") or [], body.get("tools")

    def _recompute_context(self) -> None:
        """Update the status-bar context gauge and fire the ~70% heads-up warning."""
        msgs, tools = self._context_snapshot()
        if msgs is None:
            self._ctx = None
            return
        used = render.context_used(msgs, tools)
        window = self.caps.context_window if self.caps else None
        if window:
            frac = used / window
            pct = round(frac * 100)
            self._ctx = (f"{pct}%", frac)
            for thresh in (80, 70):  # check higher first so a jump fires only once
                if pct >= thresh and self._ctx_warn_level < thresh:
                    self._ctx_warn_level = thresh
                    tail = (
                        "auto-compaction imminent (85%)"
                        if thresh == 80
                        else "approaching auto-compaction (85%)"
                    )
                    self.notify(
                        f"⚠ context {pct}% of {render._ktok(window)} — {tail}",
                        severity="warning",
                        timeout=8,
                    )
                    break
        else:
            self._ctx = (render._ktok(used), None)

    def _update_status(self) -> None:
        from textual.css.query import NoMatches

        if self.caps is None:
            return
        if self._stats_dirty:
            self._recompute_stats()
            self._stats_dirty = False
        bar = status_bar(
            preset=self._preset.name,
            tier=self.caps.tier(),
            glyphs=ability_glyphs(self.caps),
            saved=self._stats["saved"],
            deterministic=self.caps.seed,
            learn=self._learn_state(),
            ctx=self._ctx,
            vim=self._vim_mode if self._vim_enabled else None,
        )
        try:
            self.query_one("#statusbar", Static).update(bar)
        except NoMatches:
            pass

    def _apply_preset(self) -> None:
        """Set the active agent preset's tool-permission profile on the registry."""
        if self._tool_registry is None:
            return
        self._tool_registry.permissions = (
            None
            if self.allow_all
            else self._preset.permissions(approver=self._approve_tool)
        )
        if self._status_base:
            self.sub_title = f"{self._status_base} · {self._preset.name}"
        if self._banner_built:
            self._refresh_banner()
            self._update_status()

    def _maybe_welcome(self) -> None:
        if self._welcomed or self.active is not None or self.caps is None:
            return
        self._welcomed = True
        self.query_one(RichLog).write(welcome_panel(self.client.model, self.caps))

    async def _approve_tool(self, tool: str, args: str) -> bool:
        return await self.push_screen_wait(PermissionModal(tool, args))

    def _build_agent(self, preset=None) -> Agent:
        preset = preset or self._preset
        tools = self._tool_registry or ToolRegistry(builtin_tools())
        factory = None
        if self.use_guardrails:
            from ..agent.codemode import RUN_CODE_NAME
            from ..agent.tools import TOOL_SEARCH_NAME
            from ..guardrails.guardrails import Guardrails
            from ..server.coordinator import SEND_MESSAGE_NAME, SPAWN_AGENTS_NAME

            names = [s["function"]["name"] for s in tools.schemas()] + [
                TOOL_SEARCH_NAME,
                SPAWN_AGENTS_NAME,
                SEND_MESSAGE_NAME,
                RUN_CODE_NAME,
            ]
            factory = lambda: Guardrails(  # noqa: E731
                tool_names=names,
                required_steps=self.required_steps,
                terminal_tools=self.terminal_tools,
            )
        # Effort tunes resampling: low=off, else a StepPolicy threshold. An explicit
        # --resample-threshold flag overrides only at the default (medium) effort.
        eff = _EFFORT[self._effort]
        threshold = eff["resample"]
        if self._effort == "medium" and self.resample_threshold is not None:
            threshold = self.resample_threshold
        policy = None
        if threshold is not None:
            from ..signals.policies import StepPolicy

            policy = StepPolicy(min_mean_logprob=threshold, max_retries=1)
        # Effort also steers reasoning depth via a system-prompt suffix.
        system_prompt = preset.system_prompt
        if eff["think"]:
            system_prompt = f"{system_prompt}\n\n{eff['think']}"
        return Agent(
            self.client,
            tools,
            EventLog(self.db_path),
            capabilities=self.caps,
            system_prompt=system_prompt,
            sampling=preset.sampling,
            max_steps=self.max_steps,
            guardrails_factory=factory,
            policy=policy,
            context_budget=self.context_budget,
            on_token=self._on_token,
            code_mode=self._code_mode,
            sandbox=self._sandbox,
            notebook=self.notebook,
            exposed_tools=preset.exposed(),
            retrieval=self._memory,
            on_tool=self._on_tool_event,
            on_notice=lambda m: self.notify(m, severity="warning", timeout=8),
        )

    async def _run_worker(self, task: str) -> None:
        await self._caps_ready.wait()
        try:
            await self._build_agent().run(task)
        except Exception as e:
            self.notify(f"run error: {e}", severity="error")
        await self._auto_consolidate()
        self._maybe_background()

    async def _continue_worker(self, run_id: str, message: str) -> None:
        await self._caps_ready.wait()
        try:
            await self._build_agent().continue_run(run_id, message)
        except Exception as e:
            self.notify(f"run error: {e}", severity="error")
        await self._auto_consolidate()
        self._maybe_background()

    async def _auto_consolidate(self) -> None:
        """Automatic capture: after a completed run, write a one-line episodic note
        to the recall DB on free local tokens. Idempotent (skips already-summarized
        runs); the heavier reflect/induce stays on the --background cycle."""
        if self._memory is None:
            return
        try:
            from ..background.consolidate import consolidate

            n = await consolidate(
                EventLog(self.db_path), self._memory, self.client, limit=2
            )
            if n:
                self._stats_dirty = True
                self._refresh_banner()
        except Exception:
            pass  # consolidation must never disrupt the conversation

    def _maybe_background(self) -> None:
        """After a run, while the user reads the result (idle), learn for free."""
        if (
            self.background_enabled
            and not self._cycle_running
            and self._memory is not None
        ):
            self.run_worker(self._background_worker(), exclusive=False)

    async def _background_worker(self) -> None:
        from ..background import background_cycle, summarize_cycle
        from pathlib import Path

        self._cycle_running = True
        try:
            counts = await background_cycle(
                EventLog(self.db_path),
                self._memory,
                self.client,
                Path(self.memory_dir) / "drafts",
                caps=self.caps,
            )
            if any(counts.values()):
                self._learn_summary = summarize_cycle(counts)
                self.notify(self._learn_summary, timeout=8)
                self._stats_dirty = True
                self._refresh_banner()
        except Exception as e:
            self.notify(f"background error: {e}", severity="warning")
        finally:
            self._cycle_running = False

    async def _replay_worker(self, run_id: str) -> None:
        await self._caps_ready.wait()
        self.notify(f"replaying {run_id[:8]} against the live server…")
        try:
            report = await replay_run(self.event_log, run_id, self.client)
        except Exception as e:  # noqa: BLE001 — surface the failure instead of dying silently
            self.query_one(RichLog).write(
                Text(f"  ⟲ replay: failed — {e}", style="red")
            )
            return
        self.query_one(RichLog).write(
            Text(
                f"  ⟲ replay: {report.summary()}",
                style="green" if report.identical else "red",
            )
        )

    async def _reconnect(self, url: str, model: str, key: str) -> None:
        if self.server is not None:
            # Server mode: the embedded/remote server owns the upstream client —
            # repoint it over HTTP instead of swapping the (unused) local client.
            import httpx

            self.caps = None
            self._status_base = None
            self._caps_ready = asyncio.Event()
            self.sub_title = f"repointing server upstream to {url}…"
            try:
                # generous timeout: probing a large model can take a while
                async with httpx.AsyncClient(timeout=90) as c:
                    r = await c.post(
                        f"{self.server}/connect",
                        json={"url": url, "model": model or None},
                    )
                    r.raise_for_status()
            except Exception as e:
                self.notify(f"connect failed: {e}", severity="error", timeout=10)
                return
            await self._probe_server()
            return
        try:
            await self.client.aclose()
        except Exception:
            pass
        self.client = OpenAICompatClient(url, model, api_key=key or None)
        self.caps = None
        self._status_base = None
        self._caps_ready = asyncio.Event()
        self.sub_title = f"connecting to {url}…"
        self._set_banner_title(f"local_harness  ·  connecting to {url}…")
        self.action_new()
        chosen = await self._resolve_model(model)
        if chosen is None:  # user cancelled the picker
            self.notify("connect cancelled — staying on the previous server")
            return
        self.client.model = chosen
        await self._probe_worker()

    async def _resolve_model(self, typed: str) -> str | None:
        """Resolve the model against what the endpoint actually serves: use the
        typed id if valid, auto-select when there's exactly one, otherwise pop a
        picker. Returns None only if the user cancels the picker."""
        try:
            models = await self.client.list_models()
        except Exception as e:
            self.notify(
                f"couldn't list models at {self.client.base_url}: {e}",
                severity="error",
                timeout=8,
            )
            return typed  # fall back to whatever was typed (may be blank)
        if not models:
            if not typed:
                self.notify(
                    "this endpoint lists no models — set one explicitly with ^o",
                    severity="warning",
                )
            return typed
        if typed and typed in models:
            return typed
        if typed:
            self.notify(
                f"'{typed}' isn't served here — pick from the list", severity="warning"
            )
        if len(models) == 1:
            self.notify(f"auto-selected the only model: {models[0]}")
            return models[0]
        return await self.push_screen_wait(ModelPickerScreen(models))

    # --- streaming callback (runs on the event loop, from the agent worker) --

    def _on_tool_event(self, name: str, phase: str) -> None:
        # phase: "start" when a tool begins executing → show the live indicator
        self._running_tool = name if phase == "start" else None
        self._render_live()

    def _on_token(self, kind: str, text: str) -> None:
        if kind == "start":
            if self._live_content:  # a previous attempt is being resampled — ghost it
                self._live_ghost = self._live_content
            self._streaming = True
            self._live_content = self._live_reasoning = ""
        elif kind == "content":
            self._live_content += text
        elif kind == "reasoning":
            self._live_reasoning += text
        self._render_live()

    def _render_live(self) -> None:
        from textual.css.query import NoMatches

        try:
            live = self.query_one("#live", Static)
        except NoMatches:  # mid mount/teardown — nothing to draw yet
            return
        active_running = any(
            s == "running" and r == self.active for r, s, _ in self._runs_state
        )
        if not (self._streaming or active_running):
            live.update("")
            return
        # Fan-out view: N concurrent rollouts, each streaming into its own row
        # (the free-tokens advantage made visible — best-of-N, self-consistency, …).
        if self._live_streams is not None:
            circ = "①②③④⑤⑥⑦⑧⑨⑩"
            rows: list = [Text(self._live_fan_title, style="bold " + render.C_GOLD)]
            for i, s in enumerate(self._live_streams):
                done = s["status"] == "done"
                mark = "✓" if done else SPINNER[self._spin]
                num = circ[i] if i < len(circ) else str(i + 1)
                live_txt = (s["text"] or s["reason"]).replace("\n", " ").strip()
                tail = live_txt[-64:] if live_txt else "…"
                row = Text.assemble(
                    (
                        f"  {num} {mark} ",
                        "bold " + (render.C_OK if done else render.C_RESAMPLE),
                    ),
                    (tail, render.C_ANSWER if s["text"] else render.C_REASON),
                )
                if s["result"]:
                    row.append(f"  → {s['result']}", "bold " + render.C_GOLD)
                rows.append(row)
            live.update(Group(*rows))
            return
        parts: list = []
        if self._live_reasoning:
            # a stable tail of WHOLE lines — not a sliding char window (which made
            # lines look like they were being "edited into nothing" as it slid).
            tail = "\n".join(self._live_reasoning.strip().splitlines()[-6:])
            parts.append(
                Text.assemble(
                    (
                        f"  ✎ thinking {SPINNER[self._spin]}\n",
                        "italic " + render.C_REASON,
                    ),
                    ("  " + tail.replace("\n", "\n  "), render.C_REASON),
                )
            )
        if self._live_content:
            tail = "\n".join(self._live_content.splitlines()[-12:])
            parts.append(
                Text.assemble(
                    ("⏺ ", "bold " + render.C_OK),
                    (tail, render.C_ANSWER),
                    ("▌", render.C_RESAMPLE),
                )
            )
        if (
            self._running_tool
        ):  # a tool is executing — show it so the user isn't in the dark
            parts.append(
                Text.assemble(
                    (f"  ⚙ {SPINNER[self._spin]} ", render.C_TOOLMARK),
                    (f"running {self._running_tool}…", render.C_TOOL),
                )
            )
        if self._live_ghost:  # the rejected attempt, struck through beneath
            parts.append(
                Text.assemble(
                    ("  ", ""),
                    (self._live_ghost[-120:], "strike " + render.C_DIM),
                    ("  ↻ resampled", render.C_RESAMPLE + " italic"),
                )
            )
        if not parts:  # working, no tokens yet — the indicator sits right here
            parts.append(
                Text.assemble(
                    (SPINNER[self._spin] + " ", render.C_RESAMPLE),
                    ("working", render.C_DIM),
                )
            )
        live.update(Group(*parts))

    # --- UI events -------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Only the main prompt submits a turn. Modal inputs (Connect, etc.) bubble
        # their Input.Submitted up here too — ignore them so a typed provider URL
        # isn't mistaken for a message.
        if event.input.id != "prompt":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self._handle_slash(text)
            return
        self._submit_turn(text)

    def _submit_turn(self, text: str) -> None:
        if not self._banner_collapsed_once:  # reclaim vertical space for the transcript
            self._banner_collapsed_once = True
            try:
                self.query_one("#banner", Collapsible).collapsed = True
            except Exception:
                pass
        if not self._caps_ready.is_set():
            self.notify("upstream still connecting — task starts when it's ready")
        status = self._active_status()
        is_continue = (
            bool(self.active) and not self._blank and status in ("completed", "failed")
        )
        # plan mode: mark the turn so its result is presented as an approvable plan;
        # wrap a *fresh* task with the plan instruction (a continue/refine is left raw).
        send = text
        if self._preset.name == "plan":
            self._expect_plan = True
            self._plan_pending = (
                False  # a refined plan is coming; supersede the old one
            )
            if not is_continue:
                send = _PLAN_INSTRUCTION + text
        if self.server is not None:  # thin client: the server runs the agent
            if status == "running":
                self.notify("still working — wait for the current turn to finish")
                return
            self._active_worker = self.run_worker(
                self._server_submit(send), exclusive=False
            )
            return
        if is_continue:
            # continue the loaded conversation rather than starting a new one
            self._active_worker = self.run_worker(
                self._continue_worker(self.active, send), exclusive=False
            )
        elif status == "running":
            self.notify("still working — wait for the current turn to finish")
        else:
            self._blank = False
            self._follow = True
            self._active_worker = self.run_worker(
                self._run_worker(send), exclusive=False
            )

    def action_confirm_quit(self) -> None:
        """^C once arms the exit; a second ^C within 2s quits — so a stray ^C
        doesn't drop you out of a long run."""
        import time

        now = time.monotonic()
        if now - self._last_quit < 2.0:
            self.exit()
        else:
            self._last_quit = now
            self.notify("press ^C again to exit", timeout=2)

    def action_interrupt(self) -> None:
        """Stop the current turn (Esc or /stop) so you can rewind / re-ask without
        waiting for it to finish. A no-op (quietly) when nothing is running."""
        if self._active_status() != "running":
            return
        if self.server is not None:
            self.run_worker(self._server_interrupt(self.active), exclusive=False)
        else:
            if self._active_worker is not None:
                try:
                    self._active_worker.cancel()
                except Exception:
                    pass
            self.event_log.append(
                self.active, RUN_FAILED, {"error": "interrupted by user"}
            )
        self._streaming = False
        self._live_content = self._live_reasoning = self._live_ghost = ""
        self._running_tool = None
        self.notify("interrupted — you can /rewind, /undo, or re-ask")

    async def _server_interrupt(self, run_id: str) -> None:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(f"{self.server}/session/{run_id}/interrupt")
        except Exception as e:  # noqa: BLE001
            self.notify(f"interrupt failed: {e}", severity="error")

    def _snip_points(self) -> list[tuple[int, str, int]]:
        """Snippable turns in the active run: tool results and assistant answers
        (not already snipped), as (seq, label, size_chars)."""
        if not self.active:
            return []
        events = self.event_log.events(self.active)
        snipped = {e.payload.get("seq") for e in events if e.type == MESSAGE_SNIPPED}
        pts: list[tuple[int, str, int]] = []
        for ev in events:
            if ev.seq in snipped:
                continue
            if ev.type == TOOL_CALL:
                res = ev.payload.get("result") or ""
                pts.append(
                    (
                        ev.seq,
                        f"tool {ev.payload.get('name')} → {res.strip()[:46]}",
                        len(res),
                    )
                )
            elif ev.type == MODEL_CALL:
                msg = (
                    (ev.payload.get("response") or {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                )
                c = (msg.get("content") or "").strip()
                if c:
                    pts.append((ev.seq, f"answer: {c[:46]}", len(c)))
        return pts

    def action_snip(self) -> None:
        """/snip — collapse a chosen turn to free context (lossless; the original
        stays in the event log, so /rewind and replay still see it)."""
        if not self.active:
            self.notify("no conversation to snip")
            return
        points = self._snip_points()
        if not points:
            self.notify("nothing snippable yet — ask something first")
            return

        def done(seq: int | None) -> None:
            if seq is not None:
                self.event_log.append(self.active, MESSAGE_SNIPPED, {"seq": seq})
                self._rendered = 0
                self.query_one(RichLog).clear()
                self._render_pending()
                self._stats_dirty = True
                self.notify(
                    "snipped — collapsed in future context (lossless)", timeout=5
                )

        self.push_screen(SnipScreen(points), done)

    def action_rewind(self) -> None:
        """Roll the conversation back to an earlier point (the removed tail is
        archived, so it's reversible). `/undo` is an alias."""
        if not self.active:
            self.notify("no conversation to rewind")
            return
        if self._active_status() == "running":
            self.notify("can't rewind while a turn is running")
            return
        points = self.event_log.rewind_points(self.active)
        if not points:
            self.notify("nothing to rewind yet — ask something first")
            return

        def done(seq: int | None) -> None:
            if seq is not None:
                self._do_rewind(seq)

        self.push_screen(RewindScreen(points), done)

    def _do_rewind(self, from_seq: int) -> None:
        archive_id = self.event_log.rewind(self.active, from_seq)
        if archive_id is None:
            return
        self._rendered = 0
        self.query_one(RichLog).clear()
        self._render_pending()
        self._stats_dirty = True
        self._runs_state = []  # force the sidebar to pick up the new archive run
        self.notify(
            f"rewound · tail archived as {archive_id[:8]} (browse it with ^t)",
            timeout=6,
        )

    def _active_status(self) -> str | None:
        return next((s for r, s, _ in self._runs_state if r == self.active), None)

    # --- plan flow · external editor · code artifacts --------------------

    def _final_answer(self, run_id: str) -> str:
        """The conversation's latest answer text — RUN_COMPLETED payload, else the
        last assistant message content."""
        events = self.event_log.events(run_id)
        for ev in reversed(events):
            if ev.type == RUN_COMPLETED:
                return (ev.payload.get("answer") or "").strip()
        for ev in reversed(events):
            if ev.type == MODEL_CALL:
                msg = (
                    (ev.payload.get("response") or {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                )
                return (msg.get("content") or "").strip()
        return ""

    def _present_plan(self, run_id: str) -> None:
        """A plan-mode run finished: write the plan to a temp file, show it as an
        artifact, and offer the approve / edit / refine affordances."""
        import os
        import tempfile

        plan = self._final_answer(run_id)
        if not plan:
            return
        fd, path = tempfile.mkstemp(prefix=f"harness-plan-{run_id[:8]}-", suffix=".md")
        with os.fdopen(fd, "w") as f:
            f.write(plan)
        self._plan_file = path
        self._plan_pending = True
        chat = self.query_one(RichLog)
        chat.write(render.plan_artifact(plan, title="plan", subtitle="plan mode"))
        chat.write(
            Text.assemble(
                ("  📋 plan ready — ", "bold " + render.C_OK),
                ("/approve", "bold " + render.C_TOOL),
                (" build it  ·  ", render.C_DIM),
                ("/editor", "bold " + render.C_TOOL),
                (" edit in $EDITOR  ·  ", render.C_DIM),
                ("or just reply to refine", render.C_DIM),
            )
        )
        if self._follow:
            chat.scroll_end(animate=False)
        self.notify(f"plan saved to {path}", timeout=6)

    def _approve_plan(self) -> None:
        """Approve the pending plan: switch to build mode and implement the
        (possibly user-edited) plan in the same conversation."""
        if not self._plan_pending:
            self.notify("no plan awaiting approval — produce one in plan mode first")
            return
        plan = ""
        if self._plan_file:
            try:
                with open(self._plan_file) as f:
                    plan = f.read().strip()
            except OSError:
                pass
        self._plan_pending = False
        self._expect_plan = False
        from ..agent.presets import get_preset

        self._preset = get_preset("build")  # implementation needs write/shell tools
        self._apply_preset()
        self.notify("approved → build mode; implementing the plan")
        self._submit_turn(
            "The plan above is approved. Implement it now — create and edit the "
            "necessary files and run whatever is needed. The plan:\n\n" + plan
        )

    def _open_in_editor(self, path: str) -> None:
        """Hand the terminal to $EDITOR (nano by default) on `path`, then return to
        the TUI. Blocks while the editor is open — that's the point."""
        import os
        import subprocess

        editor = os.environ.get("EDITOR") or "nano"
        try:
            with self.suspend():
                subprocess.run([*editor.split(), path])
        except Exception as e:  # noqa: BLE001
            self.notify(f"editor error: {e}", severity="error")
        finally:
            try:
                self.refresh()
            except Exception:
                pass

    def _dump_transcript(self) -> str | None:
        """Write the active conversation to a temp Markdown file for /editor."""
        if not self.active:
            return None
        import os
        import tempfile
        from ..events.export import transcript_markdown

        fd, path = tempfile.mkstemp(
            prefix=f"harness-convo-{self.active[:8]}-", suffix=".md"
        )
        with os.fdopen(fd, "w") as f:
            f.write(transcript_markdown(self.event_log, self.active))
        return path

    def _export_transcript(self) -> None:
        """/export — write the active conversation to ./run-<id>.md in the cwd."""
        if not self.active:
            self.notify("no conversation to export")
            return
        from ..events.export import transcript_markdown

        path = f"run-{self.active}.md"
        try:
            with open(path, "w") as f:
                f.write(transcript_markdown(self.event_log, self.active))
        except OSError as e:
            self.notify(f"export failed: {e}", severity="error")
            return
        self.notify(f"wrote {path} (Markdown transcript)", timeout=6)

    def action_cost(self) -> None:
        """/cost — a compact, glanceable line: $0 spent vs what it'd cost on a
        frontier API, across this session's calls."""
        if self.caps is None:
            self.notify("still connecting")
            return
        self._recompute_stats()
        self._stats_dirty = False
        s = self._stats
        self.query_one(RichLog).write(
            Text.assemble(
                ("  $0.00 spent", "bold " + render.C_OK),
                (
                    f"  ·  ~{render._money(s['saved'])} the same {s['tokens']:,} tokens would cost "
                    f"on a frontier API",
                    render.C_DIM,
                ),
                (f"  ·  {s['calls']} call(s)", render.C_DIM),
            )
        )

    def action_model(self) -> None:
        """/model — pick a model the endpoint serves and switch to it (re-probes)."""

        async def _switch() -> None:
            chosen = await self._resolve_model("")
            if chosen and chosen != self.client.model:
                await self._reconnect(self.client.base_url, chosen, "")

        self.run_worker(_switch(), exclusive=False)

    def action_effort(self, level: str) -> None:
        """/effort low|medium|high (and /fast = low): tune reasoning depth +
        resample retries for subsequent turns."""
        level = (level or "").lower()
        if level not in _EFFORT:
            self.notify(f"effort: {', '.join(_EFFORT)} (current: {self._effort})")
            return
        self._effort = level
        cfg = _EFFORT[level]
        self.notify(f"effort: {level} — {cfg['blurb']}")

    def _editor_command(self) -> None:
        """/editor — edit the pending plan if there is one, else open the whole
        conversation transcript in $EDITOR (and return to the TUI on exit)."""
        if self._plan_pending and self._plan_file:
            self._open_in_editor(self._plan_file)
            self.notify("plan updated — /approve to build it, or reply to refine")
            return
        path = self._dump_transcript()
        if not path:
            self.notify("no active conversation to open")
            return
        self._open_in_editor(path)
        self.notify(f"opened transcript in $EDITOR ({path})", timeout=5)

    def _last_code_blocks(self) -> list[tuple[str, str]]:
        if not self.active:
            return []
        return render.extract_code_blocks(self._final_answer(self.active))

    def _run_last_code(self) -> None:
        """/run — execute the last fenced code block from the latest answer and
        show its output as an artifact."""
        blocks = self._last_code_blocks()
        if not blocks:
            self.notify("no code block found in the last answer")
            return
        lang, code = blocks[-1]
        self.run_worker(self._run_code_worker(lang, code), exclusive=False)

    _RUNNERS = {
        "python": ["python3"],
        "py": ["python3"],
        "python3": ["python3"],
        "bash": ["bash"],
        "sh": ["bash"],
        "shell": ["bash"],
        "zsh": ["bash"],
        "javascript": ["node"],
        "js": ["node"],
        "node": ["node"],
        "ruby": ["ruby"],
    }
    _RUN_EXT = {
        "python": ".py",
        "py": ".py",
        "python3": ".py",
        "javascript": ".js",
        "js": ".js",
        "node": ".js",
        "ruby": ".rb",
        "bash": ".sh",
        "sh": ".sh",
    }

    async def _run_code_worker(self, lang: str, code: str) -> None:
        import asyncio as aio
        import os
        import tempfile

        runner = self._RUNNERS.get(lang)
        if runner is None:
            self.notify(
                f"/run doesn't know how to execute '{lang or 'plain'}' "
                f"(try python/bash/js)",
                severity="warning",
            )
            return
        suffix = self._RUN_EXT.get(lang, ".txt")
        fd, path = tempfile.mkstemp(prefix="harness-run-", suffix=suffix)
        with os.fdopen(fd, "w") as f:
            f.write(code)
        self.notify(f"running {lang} block… (host, 30s timeout)")
        try:
            proc = await aio.create_subprocess_exec(
                *runner, path, stdout=aio.subprocess.PIPE, stderr=aio.subprocess.STDOUT
            )
            try:
                out_b, _ = await aio.wait_for(proc.communicate(), timeout=30)
                rc = proc.returncode
            except aio.TimeoutError:
                proc.kill()
                out_b, rc = b"(timed out after 30s)", 1
        except FileNotFoundError:
            self.notify(
                f"interpreter '{runner[0]}' not found on PATH", severity="error"
            )
            return
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        output = out_b.decode("utf-8", "replace")
        out_lang = "python" if lang in ("python", "py", "python3") else lang
        self.query_one(RichLog).write(
            render.run_output_artifact(
                f"{runner[0]} <block>",
                output,
                ok=(rc == 0),
                lang=out_lang if rc == 0 else "text",
            )
        )
        self.query_one(RichLog).scroll_end(animate=False)

    def _preview_last(self) -> None:
        """/preview — open the last HTML code block (or a just-written .html file)
        in the system browser."""
        import os
        import tempfile
        import webbrowser

        html = next(
            (
                c
                for lang, c in reversed(self._last_code_blocks())
                if lang in ("html", "htm")
            ),
            None,
        )
        if html is None:  # fall back to the most recent write_file of an .html path
            for ev in reversed(
                self.event_log.events(self.active) if self.active else []
            ):
                if ev.type == TOOL_CALL and ev.payload.get("name") == "write_file":
                    try:
                        args = json.loads(ev.payload.get("arguments") or "{}")
                    except (ValueError, TypeError):
                        continue
                    if str(args.get("path", "")).lower().endswith((".html", ".htm")):
                        html = args.get("content") or ""
                        break
        if not html:
            self.notify("no HTML block or .html file to preview")
            return
        fd, path = tempfile.mkstemp(prefix="harness-preview-", suffix=".html")
        with os.fdopen(fd, "w") as f:
            f.write(html)
        try:
            webbrowser.open(f"file://{path}")
            self.notify(f"opened preview in your browser: {path}", timeout=6)
        except Exception as e:  # noqa: BLE001
            self.notify(f"couldn't open browser: {e}", severity="error")

    # --- slash commands & user-invoked skills ----------------------------

    def _skill_registry(self):
        if self._skills is None:
            from ..skills.skill import SkillRegistry

            self._skills = SkillRegistry(self.skills_dir)
        return self._skills

    def _skill_names(self) -> set[str]:
        try:
            return set(self._skill_registry().names())
        except Exception:
            return set()

    _SLASH_HELP = {
        "help": "keys & commands",
        "new": "reset the conversation",
        "clear": "reset the conversation",
        "rewind": "roll back to an earlier point (alias /undo)",
        "undo": "roll back to an earlier point (alias of /rewind)",
        "snip": "collapse a turn to free context (lossless)",
        "stop": "interrupt the current turn (also Esc)",
        "delete": "delete this conversation",
        "sessions": "browse / switch conversations",
        "agent": "switch preset (build/plan/explore/general)",
        "think": "toggle thinking",
        "usage": "cost / determinism / confidence",
        "replay": "verify determinism",
        "context": "what's in the context window (and how full)",
        "memory": "show memory (USER/MEMORY/PROJECT) · /memory edit <scope>",
        "review": "code-review the git diff (/review [ref])",
        "security-review": "security-review the git diff (/security-review [ref])",
        "mcp": "show configured MCP/UTCP tool sources",
        "vim": "toggle vim editing on the prompt",
        "codemode": "toggle code-mode: model writes code calling tools (default on)",
        "agents": "show the team tree (lead + spawned workers)",
        "tasks": "team tree (alias /agents)",
        "theme": "switch theme (osaka-jade · sakura · light · gruvbox · catppuccin · …)",
        "cost": "$ saved vs a frontier API (compact)",
        "export": "write run-<id>.md to the cwd",
        "model": "switch the model (picker)",
        "fast": "low-effort mode (brief, single pass)",
        "effort": "reasoning depth: low | medium | high",
        "plan": "fork N candidate plans",
        "connect": "switch URL / model",
        "btw": "add a steering aside",
        "editor": "edit the plan / transcript in $EDITOR",
        "approve": "approve the pending plan → build & implement",
        "run": "run the last code block, show output",
        "preview": "open last HTML in the browser",
        "grammar": "live: exactly-N output by construction (frontier miscounts)",
        "samplers": "live: DRY/min_p/XTC steer the trajectory (frontier can't)",
        "antislop": "live: ban a phrase, KV-rewind & resample",
        "overlay": "live: per-token confidence overlay from logprobs",
        "consistency": "live: self-consistency — consensus = confidence",
        "escalate": "live: agreement routes escalation (not permissions)",
        "bestof": "live: best-of-N, verifier-ranked (forks are cache hits)",
        "thinkbudget": "live: cap reasoning at the token level (s1-style)",
    }

    def _slash_catalog(self) -> list[tuple[str, str]]:
        cmds = list(self._SLASH_HELP.items())
        cmds += [
            (n, "grammar skill — guaranteed-valid output")
            for n in sorted(self._skill_names())
        ]
        return cmds

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        menu = self.query_one("#slashmenu", OptionList)
        val = event.value
        if val.startswith("/") and " " not in val:
            prefix = val[1:].lower()
            menu.clear_options()
            matches = [(n, d) for n, d in self._slash_catalog() if n.startswith(prefix)]
            for name, desc in matches:
                menu.add_option(
                    Option(
                        Text.assemble(
                            (f"/{name}", "bold " + render.C_OK),
                            (f"   {desc}", render.C_DIM),
                        ),
                        id=name,
                    )
                )
            menu.set_class(bool(matches), "visible")
            if matches:
                menu.highlighted = 0
        else:
            menu.set_class(False, "visible")

    def _accept_slash(self) -> None:
        menu = self.query_one("#slashmenu", OptionList)
        if menu.highlighted is None:
            return
        name = menu.get_option_at_index(menu.highlighted).id
        box = self.query_one("#prompt", Input)
        box.value = f"/{name} "
        box.cursor_position = len(box.value)
        menu.set_class(False, "visible")

    def on_key(self, event) -> None:
        menu = self.query_one("#slashmenu", OptionList)
        if menu.has_class("visible"):
            if event.key == "down":
                menu.action_cursor_down()
                event.prevent_default()
                event.stop()
            elif event.key == "up":
                menu.action_cursor_up()
                event.prevent_default()
                event.stop()
            elif event.key in ("tab", "right"):
                self._accept_slash()
                event.prevent_default()
                event.stop()
            elif event.key == "escape":
                menu.set_class(False, "visible")
                event.prevent_default()
                event.stop()
            return

    # --- vim mode on the prompt ------------------------------------------

    def _vim_set_mode(self, mode: str) -> None:
        self._vim_mode = mode
        self._vim_pending = ""
        try:
            self.query_one("#prompt", Input).set_class(mode == "normal", "vim-normal")
        except Exception:
            pass

    @staticmethod
    def _vim_word(text: str, pos: int, direction: int) -> int:
        if direction > 0:
            i = pos
            while i < len(text) and not text[i].isspace():
                i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            return i
        i = pos - 1
        while i > 0 and text[i].isspace():
            i -= 1
        while i > 0 and not text[i - 1].isspace():
            i -= 1
        return max(0, i)

    def _vim_key(self, box, event) -> bool:
        """Vim key handling for the prompt. Returns True if consumed (the Input
        then won't insert it). Esc→normal; i/a/A/I→insert; h l 0 $ w b motions;
        x/D/dd deletes. In insert mode only Esc is consumed; Enter always submits."""
        if event.key == "enter":
            return False  # let the prompt submit
        if self._vim_mode == "insert":
            if event.key == "escape":
                self._vim_set_mode("normal")
                return True
            return False  # normal typing
        # --- normal mode: consume the key, act on it ---
        ch = event.character or ""
        v, p = box.value, box.cursor_position
        if self._vim_pending == "d":  # dd clears the line
            self._vim_pending = ""
            if ch == "d":
                box.value = ""
                box.cursor_position = 0
            return True
        if ch == "i":
            self._vim_set_mode("insert")
        elif ch == "a":
            box.cursor_position = min(len(v), p + 1)
            self._vim_set_mode("insert")
        elif ch == "A":
            box.cursor_position = len(v)
            self._vim_set_mode("insert")
        elif ch == "I":
            box.cursor_position = 0
            self._vim_set_mode("insert")
        elif ch == "h" or event.key == "left":
            box.cursor_position = max(0, p - 1)
        elif ch == "l" or event.key == "right":
            box.cursor_position = min(len(v), p + 1)
        elif ch == "0":
            box.cursor_position = 0
        elif ch == "$":
            box.cursor_position = len(v)
        elif ch == "w":
            box.cursor_position = self._vim_word(v, p, 1)
        elif ch == "b":
            box.cursor_position = self._vim_word(v, p, -1)
        elif ch == "x" and p < len(v):
            box.value = v[:p] + v[p + 1 :]
            box.cursor_position = min(p, max(0, len(box.value) - 1))
        elif ch == "D":
            box.value = v[:p]
        elif ch == "d":
            self._vim_pending = "d"
        return True  # normal mode consumes all keys (nothing inserts)

    def action_codemode(self, arg: str = "") -> None:
        """/codemode [on|off] — toggle code-mode (model writes Python calling tools
        in one round-trip) vs classic per-tool calling. Default on."""
        arg = (arg or "").strip().lower()
        self._code_mode = (
            (arg != "off") if arg in ("on", "off") else (not self._code_mode)
        )
        where = (
            "microVM-isolated"
            if getattr(self._sandbox, "kind", "host") == "microvm"
            else "in-process"
        )
        self.notify(
            f"code-mode {'on' if self._code_mode else 'off'}"
            + (f" ({where})" if self._code_mode else " — classic tool calls")
        )

    def action_vim(self) -> None:
        """/vim — toggle modal editing on the prompt (persisted)."""
        self._vim_enabled = not self._vim_enabled
        self._save_config_value("vim", self._vim_enabled)
        self._vim_set_mode("insert")
        self.notify(f"vim mode {'on (Esc → normal)' if self._vim_enabled else 'off'}")

    def _handle_slash(self, text: str) -> None:
        parts = text[1:].split(None, 1)
        cmd = parts[0] if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in ("new", "clear"):
            self.action_new()
        elif cmd in ("help", "?"):
            self._write_help()
        elif cmd in ("agent", "mode"):
            from ..agent.presets import PRESETS, get_preset

            if arg in PRESETS:
                self._preset = get_preset(arg)
                self._apply_preset()
                self.notify(f"agent preset: {arg}")
            else:
                self.notify(f"presets: {', '.join(PRESETS)}")
        elif cmd == "sessions":
            self.action_toggle_runs()  # browse/switch conversations
        elif cmd in ("rewind", "undo"):
            self.action_rewind()
        elif cmd in ("stop", "interrupt"):
            self.action_interrupt()
        elif cmd == "snip":
            self.action_snip()
        elif cmd == "delete":
            self._delete_active()
        elif cmd in ("usage",):
            self.action_usage()
        elif cmd in ("context", "ctx"):
            self.action_context()
        elif cmd in ("memory", "mem"):
            self.action_memory(arg)
        elif cmd == "review":
            self.action_review("review", arg)
        elif cmd in ("security-review", "sec-review", "secreview"):
            self.action_review("security-review", arg)
        elif cmd in ("agents", "tasks"):
            self.action_agents()
        elif cmd == "mcp":
            self.action_mcp()
        elif cmd == "vim":
            self.action_vim()
        elif cmd in ("codemode", "code-mode"):
            self.action_codemode(arg)
        elif cmd == "theme":
            self.action_theme(arg or None)
        elif cmd == "cost":
            self.action_cost()
        elif cmd == "export":
            self._export_transcript()
        elif cmd == "model":
            self.action_model()
        elif cmd == "fast":
            self.action_effort("low")
        elif cmd == "effort":
            self.action_effort(arg)
        elif cmd in ("replay",):
            self.action_replay()
        elif cmd in ("plan", "fork", "plan-fork"):
            self.action_plan_fork()
        elif cmd in ("connect", "provider"):
            self.action_connect()
        elif cmd in ("think", "thinking"):
            self.action_toggle_thinking()
        elif cmd in ("editor", "edit"):
            self._editor_command()
        elif cmd in ("approve", "build-it"):
            self._approve_plan()
        elif cmd == "run":
            self._run_last_code()
        elif cmd == "preview":
            self._preview_last()
        elif cmd == "btw":
            if arg:
                self._submit_turn(f"By the way, {arg}")
        elif cmd in _ADVANTAGE_NAMES:
            self._blank = False
            self._follow = True
            self.run_worker(self._advantage_worker(cmd), exclusive=False)
        elif cmd in self._skill_names():
            self._blank = False
            self._follow = True
            self.run_worker(self._skill_worker(cmd, arg), exclusive=False)
        else:
            self.notify(f"unknown command /{cmd} — try /help", severity="warning")

    def _delete_active(self) -> None:
        """Delete the whole active conversation from history."""
        if not self.active:
            self.notify("no active conversation to delete")
            return
        rid = self.active
        self.event_log.delete_run(rid)
        self.action_new()
        self._runs_state = []  # force the sidebar to rebuild next tick
        self.notify(f"deleted conversation {rid[:8]}")

    def _write_help(self) -> None:
        names = sorted(self._skill_names())
        body = Text()
        body.append(
            "type a message and press Enter to talk to the model.\n\n", render.C_DIM
        )
        body.append("keys\n", "bold")
        for k, d in [
            ("Esc", "stop the current turn (then /rewind or just re-ask)"),
            ("^C ^C", "press twice to quit (one press only arms it)"),
            ("^t", "history sidebar — browse / switch conversations"),
            ("Del / ^d", "delete the highlighted history run (^d for laptops)"),
            ("^u", "usage & advantages — cost saved, determinism, confidence"),
            ("^r", "replay the active run (bit-exact determinism check)"),
            ("^g", "plan-fork — generate N candidate plans, ranked"),
            ("^o", "connect — switch URL / model / API key"),
            ("^y", "show/hide the model's full thinking"),
            ("PgUp/PgDn", "scroll the transcript · ^Home/^End jump to top/bottom"),
            ("^p", "command palette (everything, searchable)"),
        ]:
            body.append(f"  {k:<11}", render.C_OK)
            body.append(f"{d}\n", render.C_DIM)
        # The command list is generated from the registry, so it never goes stale.
        body.append("\ncommands  (type / for autocomplete)\n", "bold")
        for name, desc in sorted(self._SLASH_HELP.items()):
            body.append(f"  /{name:<16}", render.C_OK)
            body.append(f"{desc}\n", render.C_DIM)
        if names:
            body.append(
                "\nskills  (run with /<name> <text> — grammar-constrained, valid output)\n",
                "bold",
            )
            body.append("  " + "   ".join(names), render.C_OK)
        panel = Panel(
            body,
            title="help — keys & commands",
            title_align="left",
            border_style=render.B_ACCENT,
            padding=(1, 2),
        )
        self.push_screen(OverlayScreen(panel, hint="↑↓ scroll · esc to close"))

    async def _skill_worker(self, name: str, prompt: str) -> None:
        await self._caps_ready.wait()
        from ..skills.exec import generate_with_skill

        log = EventLog(self.db_path)
        rid = log.create_run(f"/{name} {prompt}".strip())
        t0 = time.monotonic()
        try:
            skill = self._skill_registry().get(name)
            result = await generate_with_skill(self.client, self.caps, skill, prompt)
        except Exception as e:
            log.append(rid, RUN_FAILED, {"error": f"{type(e).__name__}: {e}"})
            self.notify(f"skill error: {e}", severity="error")
            return
        log.append(
            rid,
            MODEL_CALL,
            {
                "call_index": 0,
                "seed": 1,
                "request_body": {},
                "response": {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": result.text},
                            "finish_reason": "stop",
                        }
                    ]
                },
                "timing_ms": (time.monotonic() - t0) * 1000,
                "logprob_summary": None,
                "grammar_valid": result.valid,
            },
        )
        log.append(rid, RUN_COMPLETED, {"answer": result.text})

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row = event.cursor_row
        if row is None or not (0 <= row < len(self._run_ids)):
            return
        rid = self._run_ids[row]
        self._blank = False
        self._follow = row == len(self._run_ids) - 1
        if rid != self.active:
            self._set_active(rid)
        self.query_one(Input).focus()

    def action_toggle_runs(self) -> None:
        runs = self.query_one("#runs", DataTable)
        runs.toggle_class("visible")
        if runs.has_class("visible"):
            runs.focus()
            self.notify("conversations:  ↵ open  ·  Del delete  ·  ^t close", timeout=6)
        else:
            self.query_one(Input).focus()

    def action_replay(self) -> None:
        if self.server is not None:
            self.notify("replay runs on the server — use `lo replay` there", timeout=6)
            return
        if self.active:
            self.run_worker(self._replay_worker(self.active), exclusive=False)

    def action_scroll_chat_up(self) -> None:
        self.query_one("#chat", RichLog).scroll_page_up()

    def action_scroll_chat_down(self) -> None:
        self.query_one("#chat", RichLog).scroll_page_down()

    def action_scroll_chat_top(self) -> None:
        self.query_one("#chat", RichLog).scroll_home()

    def action_scroll_chat_bottom(self) -> None:
        self.query_one("#chat", RichLog).scroll_end()

    def action_toggle_thinking(self) -> None:
        self._show_thinking = not self._show_thinking
        self._rerender_active()
        self.notify(f"thinking {'shown' if self._show_thinking else 'hidden'}")

    def _rerender_active(self) -> None:
        """Repaint the whole active transcript from the log (e.g. after a toggle)."""
        self._rendered = 0
        self.query_one(RichLog).clear()
        self._render_pending()

    def action_usage(self) -> None:
        """Open the usage & advantages overlay (cost / determinism / confidence)."""
        if self.caps is None:
            self.notify("still connecting — usage available once the server is probed")
            return
        self._recompute_stats()
        self._stats_dirty = False
        panel = usage_panel(
            self._stats,
            model=self.client.model,
            tier=self.caps.tier(),
            deterministic=self.caps.seed,
            learn_summary=self._learn_summary,
        )
        self.push_screen(OverlayScreen(panel, hint="↑↓ scroll · esc / ^u to close"))

    # --- theming ---------------------------------------------------------

    def _load_config(self) -> dict:
        try:
            with open(_CONFIG_PATH) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_config_value(self, key: str, value) -> None:
        cfg = self._load_config()
        cfg[key] = value
        try:
            os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
            with open(_CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
        except OSError:
            pass

    def _preview_theme(self, name: str) -> None:
        """Swap the active palette + Textual chrome (cheap — no transcript repaint)."""
        if name not in render.THEMES:
            return
        render.set_palette(render.THEMES[name])
        try:
            self.theme = name
        except Exception:
            pass

    def _apply_theme(self, name: str) -> None:
        """Switch theme and repaint everything already on screen with the new palette."""
        self._preview_theme(name)
        from textual.css.query import NoMatches

        try:
            self._rendered = 0
            self.query_one("#chat", RichLog).clear()
            self._render_pending()
        except NoMatches:
            pass
        if self._banner_built:
            self._refresh_banner()
        self._stats_dirty = True
        self._update_status()
        if self.active is None and self.caps is not None:
            self._welcomed = False
            self._maybe_welcome()

    def action_theme(self, name: str | None = None) -> None:
        """Switch theme. `/theme <name>` applies directly; `/theme` opens a picker
        with live preview. The choice persists in ~/.harness/config.json."""
        if name and name in render.THEMES:
            self._apply_theme(name)
            self._save_config_value("theme", name)
            self.notify(f"theme: {name}")
            return
        current = str(self.theme)

        def done(chosen: str | None) -> None:
            self._apply_theme(chosen or current)
            if chosen:
                self._save_config_value("theme", chosen)
                self.notify(f"theme: {chosen}")

        self.push_screen(ThemePickerScreen(list(render.THEMES), current), done)

    def action_memory(self, arg: str = "") -> None:
        """/memory — show the USER/MEMORY/PROJECT panel + recall; /memory edit
        <scope> opens that file in $EDITOR."""
        if self.notebook is None:
            self.notify("memory isn't enabled for this session")
            return
        arg = (arg or "").strip()
        if arg.split()[:1] == ["edit"]:
            scope = arg.split(None, 1)[1].strip() if " " in arg else "project"
            return self._memory_edit(scope)
        sections = [
            ("USER.md", self.notebook.user.read()),
            ("MEMORY.md", self.notebook.memory.read()),
        ]
        if self.notebook.project is not None:
            sections.append(("PROJECT.md", self.notebook.project.read()))
        recall: list[tuple[str, str]] = []
        if self._memory is not None and self.active:
            meta = self.event_log.run(self.active)
            if meta:
                recall = [
                    (h.kind, h.text) for h in self._memory.recall(meta.task, limit=5)
                ]
        self.push_screen(
            OverlayScreen(
                render.memory_panel(sections, recall),
                hint="/memory edit <user|memory|project> · esc to close",
            )
        )

    def _memory_edit(self, scope: str) -> None:
        f = self.notebook._file(scope) if self.notebook else None
        if f is None:
            self.notify(f"unknown scope {scope!r} — use user | memory | project")
            return
        f.path.parent.mkdir(parents=True, exist_ok=True)
        if not f.path.exists():
            f.path.write_text(f"# {f.title}\n")
        self._open_in_editor(str(f.path))
        self.notify(f"edited {f.title}", timeout=4)

    def action_agents(self) -> None:
        """/agents — the team tree: the lead run and the workers it fanned out
        (spawn_agents), with each worker's status."""
        if not self.active:
            self.notify("no active run")
            return
        spawned = [
            ev.payload for ev in self.event_log.events(self.active, type=AGENT_SPAWNED)
        ]
        status = {r.run_id: r.status for r in self.event_log.runs()}
        meta = self.event_log.run(self.active)
        body: list = [
            Text.assemble(
                ("● lead ", "bold " + render.C_OK),
                (self.active[:8], render.C_DIM),
                (f"  {status.get(self.active, '?')}", render.C_DIM),
            )
        ]
        if meta:
            body.append(
                Padding(Text(meta.task[:72], style=render.C_ANSWER), (0, 0, 1, 2))
            )
        if not spawned:
            body.append(
                Text(
                    "no workers yet — the lead can call spawn_agents([...]) to fan out",
                    style=render.C_DIM,
                )
            )
        for s in spawned:
            cid = s.get("child_run_id", "")
            st = status.get(cid, "?")
            glyph = "✓" if st == "completed" else "✗" if st == "failed" else "…"
            body.append(
                Text.assemble(
                    (f"  └─ {glyph} worker ", "bold " + render.C_GOLD),
                    (cid[:8], render.C_DIM),
                    (f"  {st}", render.C_DIM),
                )
            )
            body.append(
                Padding(
                    Text(str(s.get("task", ""))[:68], style=render.C_ANSWER),
                    (0, 0, 1, 5),
                )
            )
        self.push_screen(
            OverlayScreen(
                Panel(
                    Group(*body),
                    title="✦ agents (team)",
                    title_align="left",
                    border_style=render.B_ACCENT,
                    padding=(1, 2),
                ),
                hint="esc to close",
            )
        )

    def action_mcp(self) -> None:
        """/mcp — show configured MCP servers + UTCP manuals and the loaded tool count."""
        cfg = self.tools_config or {}
        mcp, utcp = cfg.get("mcp", []), cfg.get("utcp", [])
        deferred = (
            self._tool_registry.deferrable_names() if self._tool_registry else set()
        )
        body: list = [Text("MCP servers", style="bold " + render.C_GOLD)]
        if not mcp:
            body.append(
                Text("  (none — add via the --tools config)", style=render.C_DIM)
            )
        for s in mcp:
            loc = s.get("url") or " ".join(s.get("command", [])) or "?"
            tag = "http" if s.get("url") else "stdio"
            if s.get("token") or s.get("token_env"):
                tag += " · 🔒 auth"
            if s.get("namespace"):
                tag += " · " + s["namespace"]
            body.append(
                Text.assemble(
                    ("  • ", render.C_OK),
                    (loc, render.C_ANSWER),
                    (f"  [{tag}]", render.C_DIM),
                )
            )
        body.append(Text(""))
        body.append(Text("UTCP manuals", style="bold " + render.C_GOLD))
        if not utcp:
            body.append(Text("  (none)", style=render.C_DIM))
        for m in utcp:
            name = m if isinstance(m, str) else (m.get("namespace") or "inline manual")
            body.append(
                Text.assemble(("  • ", render.C_OK), (str(name), render.C_ANSWER))
            )
        body.append(
            Text(
                f"\n{len(deferred)} external tool(s) loaded · deferred behind "
                "tool_search when the toolset exceeds 15",
                style=render.C_DIM,
            )
        )
        self.push_screen(
            OverlayScreen(
                Panel(
                    Group(*body),
                    title="✦ tool sources (MCP / UTCP)",
                    title_align="left",
                    border_style=render.B_ACCENT,
                    padding=(1, 2),
                ),
                hint="esc to close",
            )
        )

    def action_review(self, kind: str = "review", arg: str = "") -> None:
        """/review and /security-review: gather the git diff and review it read-only
        (a one-shot run with the review preset; your active preset is untouched)."""
        self.run_worker(self._review_worker(kind, (arg or "").strip()), exclusive=False)

    async def _review_worker(self, kind: str, arg: str) -> None:
        await self._caps_ready.wait()
        import subprocess

        spec = arg or "HEAD"
        try:
            diff = subprocess.run(
                ["git", "diff", spec], capture_output=True, text=True, timeout=15
            ).stdout
            if (
                not diff.strip() and not arg
            ):  # nothing vs HEAD → plain working-tree diff
                diff = subprocess.run(
                    ["git", "diff"], capture_output=True, text=True, timeout=15
                ).stdout
        except Exception as e:  # noqa: BLE001
            self.notify(f"git diff failed: {e}", severity="error")
            return
        if not diff.strip():
            self.notify(
                "nothing to review — the working tree is clean (try /review <ref>)"
            )
            return
        from ..agent.presets import get_preset

        label = "security review" if kind == "security-review" else "code review"
        task = (
            f"Perform a {label} of the following diff"
            + (f" ({arg})" if arg else "")
            + ".\n\n```diff\n"
            + diff[:20000]
            + "\n```"
        )
        self.notify(f"{label}: analyzing {len(diff)} chars of diff…")
        self._blank = False
        self._follow = True
        result = None
        try:
            result = await self._build_agent(preset=get_preset(kind)).run(task)
        except Exception as e:  # noqa: BLE001
            self.notify(f"review error: {e}", severity="error")
        # Grammar-enforce the findings into a guaranteed-valid structured list (a
        # local-only edge — needs a Tier-2 server with grammar support).
        if (
            result is not None
            and self.caps
            and self.caps.grammar
            and (result.answer or "").strip()
        ):
            await self._enforce_findings(kind, result.answer)
        await self._auto_consolidate()

    async def _enforce_findings(self, kind: str, prose: str) -> None:
        try:
            from ..skills.exec import generate_with_skill

            prompt = (
                "Convert the review below into the structured findings list — one per "
                "line as `<file>:<line> — <severity> — <issue>`. If there are no real "
                "issues, output exactly: No findings.\n\nReview:\n" + prose
            )
            r = await generate_with_skill(
                self.client,
                self.caps,
                _findings_skill(kind),
                prompt,
                seed=1,
                max_tokens=1024,
            )
            if r.valid and r.text.strip():
                self.query_one(RichLog).write(render.findings_panel(r.text, kind))
        except Exception:  # noqa: BLE001 — the prose review already showed; this is a bonus
            pass

    def action_context(self) -> None:
        """Open the /context view — a segmented bar of what's in the model's
        context right now (system · task · assistant · tool I/O · free)."""
        msgs, tools = self._context_snapshot()
        if not msgs:
            self.notify("no context yet — the breakdown appears after the first turn")
            return
        breakdown = render.context_breakdown(msgs, tools)
        window = self.caps.context_window if self.caps else None
        self.push_screen(
            OverlayScreen(
                render.context_panel(breakdown, window=window), hint="esc to close"
            )
        )

    def action_plan_fork(self) -> None:
        """Fork N candidate plans for the active task and show them ranked. In
        server mode this calls the server's /session/plan endpoint (the server owns
        the model client); in-process it runs plan_search directly."""
        if self.active is None:
            self.notify("plan-fork needs an active conversation — ask something first")
            return
        self.run_worker(self._plan_fork_worker(self.active), exclusive=False)

    async def _plan_fork_worker(self, run_id: str) -> None:
        await self._caps_ready.wait()
        meta = self.event_log.run(run_id)
        task = meta.task if meta else None
        if not task:
            self.notify("no task to fork", severity="warning")
            return
        self.notify(
            "plan-fork: generating candidate plans (free on local prefix-cache)…"
        )
        try:
            if self.server is not None:
                cands = await self._server_plan_fork(task, n=4)
            else:
                from ..tree.search.plan_search import plan_search
                from ..inference.types import Message

                cands = await plan_search(
                    self.client, self.caps, [Message(role="user", content=task)], n=4
                )
        except Exception as e:
            self.notify(f"plan-fork error: {e}", severity="error")
            return
        self.query_one(RichLog).write(render.plan_fork_panel(cands))
        self._stats_dirty = True

    async def _advantage_worker(self, name: str) -> None:
        """Run one live advantage demo (/samplers, /overlay, …) against the
        connected endpoint and write its panel to the transcript."""
        from .advantages import ADVANTAGES

        await self._caps_ready.wait()
        if self.client is None or self.caps is None:
            self.notify(
                f"/{name} needs a live endpoint — run `lo tui --in-process`",
                severity="warning",
            )
            return
        self.notify(f"/{name}: running live on {self.client.model}…")
        # Stream into the live region so the model types its output: a single stream
        # for the one-shot advantages (watch the grammar force out 39 sevens), or N
        # concurrent rollouts side-by-side for the fan-out ones (best-of-N, self-
        # consistency, escalation) — the free-tokens advantage, made visible.
        self._streaming = True
        self._live_content = self._live_reasoning = ""
        self._live_streams = None
        self._render_live()

        def token(kind: str, text: str) -> None:
            if kind == "reasoning":
                self._live_reasoning += (
                    text  # shows "✎ thinking…" so it's visibly working
                )
            else:
                self._live_content += text
            self._render_live()

        def fan(labels: list[str], title: str) -> None:
            self._live_fan_title = title
            self._live_streams = [
                {"label": l, "text": "", "reason": "", "status": "run", "result": ""}
                for l in labels
            ]
            self._render_live()

        def fan_token(i: int, kind: str, text: str) -> None:
            s = self._live_streams[i]
            s["reason" if kind == "reasoning" else "text"] += text
            self._render_live()

        def fan_done(i: int, result: str) -> None:
            self._live_streams[i].update(status="done", result=result)
            self._render_live()

        def reset() -> None:  # clear the live area between grammar retries
            self._live_content = self._live_reasoning = ""
            self._render_live()

        from types import SimpleNamespace

        live = SimpleNamespace(
            token=token, fan=fan, fan_token=fan_token, fan_done=fan_done, reset=reset
        )
        try:
            renderable = await ADVANTAGES[name](self.client, self.caps, live=live)
        except Exception as e:  # noqa: BLE001 — surface any endpoint/runtime error to the user
            self.notify(f"/{name} error: {e}", severity="error")
            return
        finally:
            self._streaming = False
            self._live_content = self._live_reasoning = ""
            self._live_streams = None
            self._render_live()
        self.query_one(RichLog).write(renderable)
        self._stats_dirty = True

    async def _server_plan_fork(self, task: str, n: int = 4) -> list[dict]:
        """Ask the session server to run the fan-out plan search and return its
        ranked candidates (dicts: {"text", "score"}, best-first)."""
        import httpx

        async with httpx.AsyncClient(timeout=None) as c:
            r = await c.post(f"{self.server}/session/plan", json={"task": task, "n": n})
            r.raise_for_status()
            return r.json()["candidates"]

    def action_connect(self) -> None:
        def done(result):
            if result:
                url, model, key = result
                self.run_worker(self._reconnect(url, model, key), exclusive=False)

        url = getattr(self, "_upstream", None) or self.client.base_url
        self.push_screen(ConnectScreen(url, self.client.model), done)

    def action_delete_run(self) -> None:
        """Delete the run highlighted in the history sidebar (Delete key)."""
        runs = self.query_one("#runs", DataTable)
        if not runs.has_class("visible"):
            return
        row = runs.cursor_row
        if row is None or not (0 <= row < len(self._run_ids)):
            return
        rid = self._run_ids[row]
        self.event_log.delete_run(rid)
        if rid == self.active:
            self.action_new()
        self._runs_state = []  # force the table to rebuild on the next tick
        self.notify(f"deleted run {rid[:8]}")

    def action_new(self) -> None:
        """Blank slate: clear the view and don't snap back to old runs until a
        new task is launched. History is untouched (browse it with ^t)."""
        self.active = None
        self._rendered = 0
        self._follow = False
        self._blank = True
        self._welcomed = False
        self._streaming = False
        self._live_content = self._live_reasoning = self._live_ghost = ""
        self.query_one(RichLog).clear()
        self.query_one("#live", Static).update("")
        self._maybe_welcome()
        self.query_one(Input).focus()

    # --- read-side polling ----------------------------------------------

    def _set_active(self, run_id: str) -> None:
        self.active = run_id
        self._rendered = 0
        self.query_one(RichLog).clear()
        self._render_pending()

    def _render_pending(self) -> None:
        if self.active is None:
            return
        events = self.event_log.events(self.active)
        if self._rendered >= len(events):
            return
        snipped = {e.payload.get("seq") for e in events if e.type == MESSAGE_SNIPPED}
        chat = self.query_one(RichLog)
        wrote = False
        for ev in events[self._rendered :]:
            if ev.type == MESSAGE_SNIPPED:
                continue  # the marker itself isn't shown
            if ev.seq in snipped:  # collapsed turn — show a one-line snip marker
                name = ev.payload.get("name") or ev.type
                chat.write(
                    Text.assemble(
                        ("  ✂ ", render.C_RESAMPLE),
                        (f"[snipped: {name}]", render.C_DIM),
                    )
                )
                wrote = True
                continue
            renderable = chat_render_event(ev, show_thinking=self._show_thinking)
            if renderable is not None:
                chat.write(renderable)
                wrote = True
            if ev.type in (MODEL_CALL, POLICY_TRIGGERED):
                self._stats_dirty = True  # cost/confidence/resample totals moved
            if ev.type == MODEL_CALL:  # the streamed turn is now committed
                self._streaming = False
                self._live_content = self._live_reasoning = self._live_ghost = ""
            if ev.type == TOOL_CALL:  # its result is in — stop the "running" indicator
                self._running_tool = None
            if (
                ev.type == RUN_COMPLETED
                and self._expect_plan
                and not self._plan_pending
                and ev.run_id == self.active
            ):
                self._expect_plan = False
                self._present_plan(self.active)
            if (
                ev.type == CONTEXT_COMPACTED
            ):  # auto-compaction fired — say so, re-arm the warning
                p = ev.payload
                self.notify(
                    f"⛁ context compacted: {render._ktok(p.get('before_tokens', 0))} → "
                    f"{render._ktok(p.get('after_tokens', 0))} tokens",
                    timeout=6,
                )
                self._ctx_warn_level = 0
                self._stats_dirty = True
        self._rendered = len(events)
        if wrote and self._follow:  # keep the just-committed turn in view (auto-scroll)
            chat.scroll_end(animate=False)

    def _tick(self) -> None:
        runs = self.event_log.runs()
        state = [(r.run_id, r.status, r.task) for r in runs]
        if state != self._runs_state:
            self._runs_state = state
            self._run_ids = [r.run_id for r in runs]
            self._stats_dirty = True
            table = self.query_one(DataTable)
            table.clear()
            for rid, status, task in state:
                table.add_row(rid[:8], status_text(status), task[:24], key=rid)
            if (
                not self._blank
                and self._follow
                and runs
                and runs[-1].run_id != self.active
            ):
                self._set_active(runs[-1].run_id)
            if self.active in self._run_ids:
                table.move_cursor(row=self._run_ids.index(self.active))
        self._spin = (self._spin + 1) % len(SPINNER)
        if self._status_base is not None:
            self.sub_title = self._status_base
        self._render_pending()
        self._render_live()
        self._update_status()
