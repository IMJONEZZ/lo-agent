"""Render event-log rows as Rich renderables.

Pure functions over event payloads — no Textual imports — so the transcript
view is unit-testable and reusable outside the app (e.g. a future
`lo show <run-id>`).
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from ..events.log import (
    GUARDRAIL,
    MODEL_CALL,
    POLICY_TRIGGERED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    TOOL_CALL,
    USER_MESSAGE,
    Event,
)

# ── Themeable palette ───────────────────────────────────────────────────────
# A Palette holds every hue the TUI uses (transcript + chrome). set_palette()
# projects one onto this module's color globals, which every render function —
# and the app, via render.C_* — reads by NAME at call time. So `/theme` recolors
# the whole UI (transcript, artifacts, chrome) by swapping the active palette.


@dataclass(frozen=True)
class Palette:
    name: str
    jade: str  # primary voice / success
    jade_bright: str  # emphasis
    jade_deep: str  # quiet borders
    gold: str  # tools / first accent
    gold_deep: str  # highlighted borders
    sakura: str  # reasoning / second accent
    rose: str  # errors
    amber: str  # resample / working
    cream: str  # body text / answers
    grey: str  # chrome / badges / separators
    mute: str  # "unsure" tokens
    ink: str  # chip foreground on an accent bg
    info: str  # assistant / info border
    bg: str  # Textual background
    surface: str  # Textual surface (panels / modals)
    panel: str  # Textual panel (status / live)
    code_theme: str  # Pygments style for code blocks & artifacts
    dark: bool = True


def set_palette(p: Palette) -> None:
    """Make `p` the active palette by projecting it onto the module color globals
    (read by name at call time, so this recolors everything on the next render)."""
    globals().update(
        PALETTE=p,
        JADE=p.jade,
        JADE_BRIGHT=p.jade_bright,
        JADE_DEEP=p.jade_deep,
        GOLD=p.gold,
        GOLD_DEEP=p.gold_deep,
        SAKURA=p.sakura,
        ROSE=p.rose,
        AMBER=p.amber,
        CREAM=p.cream,
        JADE_GREY=p.grey,
        JADE_MUTE=p.mute,
        INK=p.ink,
        CODE_THEME=p.code_theme,
        C_USER="bold " + p.jade,
        C_REASON="italic " + p.sakura,
        C_ANSWER=p.cream,
        C_TOOLMARK="bold " + p.gold,
        C_TOOL=p.gold,
        C_OK=p.jade,
        C_ERR=p.rose,
        C_RESAMPLE=p.amber,
        C_DIM=p.grey,
        C_SAKURA=p.sakura,
        C_GOLD=p.gold,
        B_OK=p.jade_deep,
        B_ERR=p.rose,
        B_INFO=p.info,
        B_ACCENT=p.gold_deep,
        B_SAKURA=p.sakura,
        _STATUS_STYLE={"running": p.gold, "completed": p.jade, "failed": p.rose},
        _SEP=("  │  ", p.grey),
        _CTX_COLORS={
            "system": p.jade_deep,
            "task": p.gold,
            "assistant": p.jade,
            "tool I/O": p.sakura,
        },
    )


THEMES: dict[str, Palette] = {
    p.name: p
    for p in [
        # Osaka Jade (default) — a soft, OpenCode-style dark jade-slate (no black),
        # lower contrast: muted jade/gold/sakura on a lifted background.
        Palette(
            "osaka-jade",
            jade="#52cc9e",
            jade_bright="#74e6bd",
            jade_deep="#3a9d8c",
            gold="#dcbb7a",
            gold_deep="#c2a056",
            sakura="#e8a6c2",
            rose="#e6788c",
            amber="#e3a366",
            cream="#cfe0d6",
            grey="#7e978c",
            mute="#93a99c",
            ink="#101915",
            info="#64b4be",
            bg="#1a2722",
            surface="#21302a",
            panel="#283a32",
            code_theme="nord",
        ),
        # Osaka Midnight — a deeper jade than the default (still soft, no black), dimmer.
        Palette(
            "osaka-midnight",
            jade="#46c79b",
            jade_bright="#63e0b8",
            jade_deep="#2e8c7c",
            gold="#cfb074",
            gold_deep="#b08f44",
            sakura="#dca4bc",
            rose="#df6f86",
            amber="#d99c66",
            cream="#c3d4cb",
            grey="#6a8278",
            mute="#86998f",
            ink="#0a120e",
            info="#5aa6b0",
            bg="#121b17",
            surface="#18241f",
            panel="#1e2c26",
            code_theme="nord",
        ),
        # Sakura — pink-forward on a soft dark plum (no black), accents eased.
        Palette(
            "sakura",
            jade="#a0d4ad",
            jade_bright="#bce6c6",
            jade_deep="#6fae8c",
            gold="#e6c074",
            gold_deep="#d6a64e",
            sakura="#eda6c2",
            rose="#e87088",
            amber="#e8a36a",
            cream="#e8d5de",
            grey="#9a8490",
            mute="#bba6b2",
            ink="#1c1218",
            info="#7fb9c4",
            bg="#241a20",
            surface="#2c2028",
            panel="#342530",
            code_theme="material",
        ),
        # Osaka Light — warm-paper background, dark jade text (needs dark=False).
        Palette(
            "osaka-light",
            jade="#1f8a6b",
            jade_bright="#26a37f",
            jade_deep="#16705a",
            gold="#b07d1a",
            gold_deep="#8a5e10",
            sakura="#c25c86",
            rose="#c0344f",
            amber="#b5632a",
            cream="#2a2620",
            grey="#7a8278",
            mute="#5e655c",
            ink="#f3efe7",
            info="#2f7d8a",
            bg="#f3efe7",
            surface="#e9e3d6",
            panel="#e0d8c8",
            code_theme="friendly",
            dark=False,
        ),
        # Gruvbox — the classic warm retro palette (the harness's original look).
        Palette(
            "gruvbox",
            jade="#b8bb26",
            jade_bright="#d3ff5e",
            jade_deep="#689d6a",
            gold="#fabd2f",
            gold_deep="#d79921",
            sakura="#d3869b",
            rose="#fb4934",
            amber="#fe8019",
            cream="#ebdbb2",
            grey="#928374",
            mute="#a89984",
            ink="#1d2021",
            info="#83a598",
            bg="#282828",
            surface="#32302f",
            panel="#3c3836",
            code_theme="gruvbox-dark",
        ),
        # Catppuccin Mocha — soft pastel dark.
        Palette(
            "catppuccin-mocha",
            jade="#a6e3a1",
            jade_bright="#b9f0b4",
            jade_deep="#94e2d5",
            gold="#f9e2af",
            gold_deep="#fab387",
            sakura="#f5c2e7",
            rose="#f38ba8",
            amber="#fab387",
            cream="#cdd6f4",
            grey="#6c7086",
            mute="#a6adc8",
            ink="#11111b",
            info="#89b4fa",
            bg="#1e1e2e",
            surface="#181825",
            panel="#252537",
            code_theme="material",
        ),
        # Catppuccin Macchiato — a touch warmer/lighter than mocha.
        Palette(
            "catppuccin-macchiato",
            jade="#a6da95",
            jade_bright="#b8e6aa",
            jade_deep="#8bd5ca",
            gold="#eed49f",
            gold_deep="#f5a97f",
            sakura="#f5bde6",
            rose="#ed8796",
            amber="#f5a97f",
            cream="#cad3f5",
            grey="#6e738d",
            mute="#b8c0e0",
            ink="#181926",
            info="#8aadf4",
            bg="#24273a",
            surface="#1e2030",
            panel="#2a2d44",
            code_theme="material",
        ),
    ]
}

DEFAULT_THEME = "osaka-jade"
set_palette(THEMES[DEFAULT_THEME])  # the app re-applies any saved choice on mount


def status_text(status: str) -> Text:
    return Text(status, style=_STATUS_STYLE.get(status, ""))


def confidence_style(mean_logprob: float) -> str:
    if mean_logprob > -0.3:
        return JADE
    if mean_logprob > -1.0:
        return GOLD
    return ROSE


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def render_event(ev: Event) -> RenderableType | None:
    """One event row → one renderable; None means 'nothing worth showing'."""
    p = ev.payload
    if ev.type == RUN_STARTED:
        return Text(f"▶ {p.get('task', '')}", style="bold")
    if ev.type == MODEL_CALL:
        return _model_call(p)
    if ev.type == TOOL_CALL:
        return _tool_call(p)
    if ev.type == POLICY_TRIGGERED:
        return Text(
            f"↻ policy: {p.get('action')} (attempt {p.get('attempt')}) — {p.get('reason')}",
            style=GOLD,
        )
    if ev.type == GUARDRAIL:
        return _guardrail(p)
    if ev.type == RUN_COMPLETED:
        return Panel(
            Text(p.get("answer") or ""),
            title="✓ completed",
            title_align="left",
            border_style=B_OK,
        )
    if ev.type == RUN_FAILED:
        return Panel(
            Text(p.get("error") or ""),
            title="✗ failed",
            title_align="left",
            border_style=B_ERR,
        )
    return None


def _model_call(p: dict) -> RenderableType:
    msg = (p.get("response") or {}).get("choices", [{}])[0].get("message", {})
    body = Text()
    reasoning = (msg.get("reasoning_content") or "").strip()
    if reasoning:
        body.append(_truncate(reasoning, 400), style="dim italic")
    content = (msg.get("content") or "").strip()
    if content:
        if body.plain:
            body.append("\n")
        body.append(content)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        if body.plain:
            body.append("\n")
        body.append(
            f"→ {fn.get('name')}({_truncate(fn.get('arguments') or '', 120)})",
            style=C_TOOL,
        )
    subtitle = Text(
        f"call {p.get('call_index')} · seed {p.get('seed')} · {p.get('timing_ms') or 0:.0f} ms",
        style="dim",
    )
    summary = p.get("logprob_summary")
    if summary:
        subtitle.append(" · ")
        subtitle.append(
            f"logprob {summary['mean_logprob']:.2f}",
            style=confidence_style(summary["mean_logprob"]),
        )
    return Panel(
        body,
        title="assistant",
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style=B_INFO,
    )


def _tool_call(p: dict) -> RenderableType:
    result = p.get("result") or ""
    failed = result.startswith("error:")
    t = Text()
    t.append(
        f"🔧 {p.get('name')}({_truncate(p.get('arguments') or '', 100)}) → ",
        style=C_TOOL,
    )
    t.append(_truncate(result, 200), style=C_ERR if failed else C_OK)
    return t


def _guardrail(p: dict) -> RenderableType | None:
    if p.get("action") == "fatal":
        return Text(f"🛑 guardrail fatal: {p.get('reason')}", style="bold " + ROSE)
    if p.get("rescued"):
        names = ", ".join(
            tc.get("function", {}).get("name", "?")
            for tc in p.get("rescued_calls") or []
        )
        return Text(f"🛟 rescued tool call(s) from text: {names}", style=C_SAKURA)
    nudge = p.get("nudge")
    if nudge:
        return Text(
            f"⚠ nudge[{p.get('kind')}] via {nudge.get('role')}: "
            f"{_truncate(nudge.get('content') or '', 160)}",
            style=C_SAKURA,
        )
    return None  # routine execute/final checks add nothing over the transcript


# ── chat-first rendering ───────────────────────────────────────
# A conversation stream (like Claude Code) instead of a dashboard transcript.
# Advantages surface as inline badges on each assistant turn; a confidence-
# driven resample renders as its own visible line, because "the model noticed
# it was unsure and tried again" is itself one of the advantages.

# (line-kind hues live in the Osaka Jade palette at the top of this module)


def _msg_of(p: dict) -> dict:
    return (p.get("response") or {}).get("choices", [{}])[0].get("message", {})


def chat_badges(p: dict) -> Text:
    out = Text("    ")
    bits: list[tuple[str, str]] = []
    if p.get("seed") is not None:
        bits.append((f"seed {p['seed']}", "dim"))
    s = p.get("logprob_summary")
    if s and s.get("mean_logprob") is not None:
        # mean token logprob — a surface-form/surprisal signal, NOT correctness.
        bits.append(
            (f"logprob {s['mean_logprob']:+.2f}", confidence_style(s["mean_logprob"]))
        )
    if p.get("timing_ms"):
        bits.append((f"{p['timing_ms'] / 1000:.1f}s", "dim"))
    if p.get("grammar_valid"):
        bits.append(("✓ grammar-valid", "green"))
    if p.get("antislop"):
        bits.append(("✓ no banned phrases", "green"))
    for i, (txt, st) in enumerate(bits):
        if i:
            out.append(" · ", "dim")
        out.append(txt, st)
    return out


def confidence_text(lp_content: list) -> Text:
    """Color each token by its logprob — confident bright, unsure faded, very
    unsure flagged. Confidence shown as visual weight, no numbers."""
    out = Text()
    for tok in lp_content:
        token = (
            tok.get("token", "") if isinstance(tok, dict) else getattr(tok, "token", "")
        )
        lp = (
            tok.get("logprob", 0.0)
            if isinstance(tok, dict)
            else getattr(tok, "logprob", 0.0)
        )
        if lp > -0.3:
            style = C_ANSWER  # confident
        elif lp > -1.5:
            style = JADE_MUTE  # muted — the model was unsure
        else:
            style = ROSE  # flagged — low confidence, worth a look
        out.append(token, style)
    return out


def _markdown(content: str) -> RenderableType:
    """Render an answer as Markdown (code blocks, lists, headings) like other
    harnesses — falling back to plain text if anything goes sideways."""
    try:
        from rich.markdown import Markdown

        return Padding(Markdown(content, code_theme=CODE_THEME), (0, 0, 0, 2))
    except Exception:
        return Text.assemble(("  ", ""), (content, C_ANSWER))


# ── artifacts ───────────────────────────────────────────────────────────────
# Framed, syntax-highlighted renderables for the things worth seeing in full: a
# file the model wrote or edited, a code block, a plan. The goal is that file
# editing, code, and plans look good *inside* the harness — so dropping out to an
# external $EDITOR is a preference (neovim muscle-memory), not a necessity.

_LANG_BY_EXT = {
    "py": "python",
    "pyi": "python",
    "js": "javascript",
    "mjs": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "jsx": "jsx",
    "rs": "rust",
    "go": "go",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "hpp": "cpp",
    "java": "java",
    "rb": "ruby",
    "php": "php",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "fish": "fish",
    "html": "html",
    "htm": "html",
    "css": "css",
    "scss": "scss",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "ini": "ini",
    "md": "markdown",
    "rst": "rst",
    "sql": "sql",
    "lua": "lua",
    "kt": "kotlin",
    "swift": "swift",
    "r": "r",
    "jl": "julia",
    "ex": "elixir",
    "exs": "elixir",
    "vim": "vim",
    "diff": "diff",
    "patch": "diff",
    "xml": "xml",
    "proto": "proto",
}


def lang_for_path(path: str) -> str:
    """Best-effort Pygments lexer name from a file path (for syntax highlighting)."""
    base = (path or "").rsplit("/", 1)[-1].lower()
    if base in ("dockerfile", "containerfile"):
        return "docker"
    if base in ("makefile",):
        return "make"
    ext = base.rsplit(".", 1)[-1] if "." in base else ""
    return _LANG_BY_EXT.get(ext, "text")


def code_artifact(
    code: str,
    lang: str = "text",
    *,
    title=None,
    subtitle=None,
    border: str = B_INFO,
    line_numbers: bool = True,
    start_line: int = 1,
    max_lines: int | None = None,
) -> RenderableType:
    """A syntax-highlighted, framed code panel — the core artifact primitive."""
    from rich.syntax import Syntax

    note = None
    lines = code.split("\n")
    if max_lines is not None and len(lines) > max_lines:
        shown = "\n".join(lines[:max_lines])
        note = f"… +{len(lines) - max_lines} more line(s)"
    else:
        shown = code
    try:
        body: RenderableType = Syntax(
            shown,
            lang,
            theme=CODE_THEME,
            line_numbers=line_numbers,
            word_wrap=True,
            start_line=start_line,
            background_color="default",
        )
    except Exception:
        body = Text(shown, style=C_ANSWER)
    if note:
        body = Group(body, Text("  " + note, style=C_DIM))
    return Panel(
        body,
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style=border,
        padding=(0, 1),
    )


def _artifact_title(glyph: str, path: str, *, bg: str) -> Text:
    return Text.assemble(
        (f" {glyph} ", f"bold {INK} on {bg}"), ("  ", ""), (path, "bold " + GOLD)
    )


def file_artifact(
    path: str, content: str, *, created: bool = False, max_lines: int = 60
) -> RenderableType:
    """A file the model just wrote — highlighted by extension, titled with the path."""
    lang = lang_for_path(path)
    n = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    title = _artifact_title("✎ created" if created else "✎ wrote", path, bg=JADE)
    sub = Text(f"{lang} · {n} line{'' if n == 1 else 's'}", style=C_DIM)
    return code_artifact(
        content, lang, title=title, subtitle=sub, border=B_OK, max_lines=max_lines
    )


def diff_artifact(path: str, old: str, new: str) -> RenderableType:
    """A file edit as a unified diff, +added in jade / −removed in rose."""
    import difflib

    rows = list(
        difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2)
    )
    body = Text()
    for ln in rows[2:]:  # drop the ---/+++ file header
        if ln.startswith("@@"):
            body.append(ln + "\n", style="bold " + C_SAKURA)
        elif ln.startswith("+"):
            body.append(ln + "\n", style=JADE)
        elif ln.startswith("-"):
            body.append(ln + "\n", style=ROSE)
        else:
            body.append(" " + ln + "\n", style=C_DIM)
    title = _artifact_title("✎ edited", path, bg=GOLD)
    add = sum(1 for ln in rows[2:] if ln.startswith("+"))
    rem = sum(1 for ln in rows[2:] if ln.startswith("-"))
    sub = Text.assemble((f"+{add}", JADE), (" / ", C_DIM), (f"-{rem}", ROSE))
    return Panel(
        body if body.plain else Text("(no textual change)", style=C_DIM),
        title=title,
        title_align="left",
        subtitle=sub,
        subtitle_align="right",
        border_style=B_ACCENT,
        padding=(0, 1),
    )


def plan_artifact(
    text: str, *, title: str = "plan", subtitle: str | None = None
) -> RenderableType:
    """A single plan, rendered as Markdown inside a sakura-bordered card — what the
    model proposes in plan mode, shown beautifully so it can be read/approved/edited."""
    return Panel(
        _markdown(text.strip()),
        title=Text.assemble((f" ✦ {title} ", f"bold {INK} on {SAKURA}")),
        title_align="left",
        subtitle=Text(subtitle, style=C_DIM) if subtitle else None,
        subtitle_align="right",
        border_style=B_SAKURA,
        padding=(1, 1),
    )


def run_output_artifact(
    command: str, output: str, *, ok: bool = True, lang: str = "text"
) -> RenderableType:
    """Result of running a code block / command — output framed and highlighted."""
    glyph = "▷ ran" if ok else "▷ ran (error)"
    bg = JADE if ok else ROSE
    title = Text.assemble(
        (f" {glyph} ", f"bold {INK} on {bg}"),
        ("  ", ""),
        (_truncate(command, 80), C_DIM),
    )
    return code_artifact(
        output or "(no output)",
        lang,
        title=title,
        border=B_OK if ok else B_ERR,
        line_numbers=False,
        max_lines=80,
    )


def extract_code_blocks(markdown_text: str) -> list[tuple[str, str]]:
    """Pull fenced ``` code blocks out of an answer as (lang, code) pairs — the
    raw material for /run and /preview. Lang defaults to '' when unspecified."""
    blocks: list[tuple[str, str]] = []
    lines = (markdown_text or "").splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("```"):
            lang = stripped[3:].strip().split()[0] if stripped[3:].strip() else ""
            buf: list[str] = []
            i += 1
            while i < n and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i])
                i += 1
            blocks.append((lang.lower(), "\n".join(buf)))
        i += 1
    return blocks


def chat_assistant(p: dict, show_thinking: bool = True) -> RenderableType:
    msg = _msg_of(p)
    choice = (p.get("response") or {}).get("choices", [{}])[0]
    parts: list[RenderableType] = []
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    if reasoning and show_thinking:  # full thinking, not truncated — it's scrollable
        parts.append(Text.assemble(("  ✎ ", C_REASON), (reasoning, C_REASON)))
    elif reasoning:
        parts.append(
            Text.assemble(
                ("  ✎ ", C_REASON),
                (
                    f"thinking hidden ({len(reasoning)} chars) · ^y to show",
                    "italic " + C_DIM,
                ),
            )
        )
    if choice.get("finish_reason") == "length":  # output hit a cap — say so plainly
        parts.append(
            Text.assemble(
                ("  ⚠ ", C_RESAMPLE),
                (
                    "output truncated (token cap hit) — unset HARNESS_MAX_TOKENS",
                    "italic " + C_RESAMPLE,
                ),
            )
        )
    content = (msg.get("content") or "").strip()
    if content:  # answer marker + rendered markdown
        parts.append(Text.assemble(("⏺ answer", "bold " + C_OK)))
        parts.append(_markdown(content))
    elif not msg.get("tool_calls"):
        parts.append(Text.assemble(("⏺ ", "bold " + C_OK), ("(no content)", C_DIM)))
    parts.append(chat_badges(p))
    return Group(*parts)


def chat_tool(p: dict) -> RenderableType:
    import json as _json

    result = p.get("result") or ""
    failed = result.startswith("error:")
    name = p.get("name")
    raw_args = p.get("arguments") or ""
    # code-mode: show the model's Python as a highlighted artifact + its output.
    if name == "run_code":
        try:
            code = _json.loads(raw_args).get("code", "")
        except (ValueError, TypeError):
            code = raw_args
        head = Text.assemble(
            ("  ⚡ ", C_TOOLMARK),
            ("code-mode", C_TOOL),
            (" → ", C_DIM),
            (_truncate(result, 400), C_ERR if failed else C_OK),
        )
        if code.strip():
            art = code_artifact(
                code,
                "python",
                border=B_INFO,
                max_lines=24,
                title=_artifact_title("⚡ ran", "python", bg=GOLD_DEEP),
            )
            return Group(head, art)
        return head
    # file tools: show just the path in the header, then a rich artifact below.
    short = _truncate(raw_args, 200)
    parsed = None
    if name in ("edit_file", "write_file", "read_file", "list_dir"):
        try:
            parsed = _json.loads(raw_args)
            short = parsed.get("path", short)
        except (ValueError, TypeError):
            parsed = None
    head = Text.assemble(
        ("  ⚙ ", C_TOOLMARK),
        (f"{name}(", C_TOOL),
        (short, C_ANSWER),
        (")", C_TOOL),
        (" → ", C_DIM),
        (_truncate(result, 600), C_ERR if failed else C_OK),
    )
    if failed or parsed is None:
        return head
    # Beautiful artifacts for file activity — the whole point of seeing it inline.
    path = parsed.get("path", "")
    if name == "write_file":
        return Group(
            head, file_artifact(path, parsed.get("content") or "", created=True)
        )
    if name == "edit_file":
        return Group(
            head,
            diff_artifact(
                path, parsed.get("old_string") or "", parsed.get("new_string") or ""
            ),
        )
    if name == "read_file":
        # the result IS the file content — highlight it by extension
        return Group(
            head,
            code_artifact(
                result,
                lang_for_path(path),
                title=_artifact_title("⊙ read", path, bg=JADE_DEEP),
                border=B_INFO,
                max_lines=60,
            ),
        )
    return head


def welcome_panel(model: str, caps) -> RenderableType:
    """Empty-state card: what this server unlocks, so first-use is informative."""
    feats: list[str] = []
    if caps.seed and caps.logprobs:
        feats += ["exact replay", "uncertainty signals"]
    elif caps.seed:
        feats.append("seeded determinism")
    if caps.grammar:
        feats.append("grammar skills")
    if caps.sampler_zoo:
        feats.append(f"sampler zoo ({len(caps.sampler_zoo)})")
    if caps.raw_completion:
        feats.append("think-budget · anti-slop")
    if caps.kv_snapshot or caps.parallel_n:
        feats.append("KV-fork search")
    body = Text()
    body.append(f"connected to {model}", "bold")
    body.append(f"   {caps.server} · tier {caps.tier()}\n\n", "dim")
    body.append("unlocked here   ", "dim")
    body.append("  ·  ".join(feats) or "basic agent loop", "green")
    body.append("\n\ntype a task and press Enter", "dim italic")
    body.append(
        "   ·  /help commands  ·  ^o connect  ·  ^t history  ·  Esc stops a turn", "dim"
    )
    return Panel(
        body,
        title="✦ local_harness",
        title_align="left",
        border_style=B_ACCENT,
        padding=(1, 3),
    )


# ── persistent ability surface: status bar · banner · usage panel ──────────
# OpenCode/Hermes both keep model + cost + mode on a permanent status line; we
# do the same, but the cost segment reads $0 (and what the same tokens WOULD
# have cost on a frontier API), and we add the things an API can't show: the
# capability tier, bit-exact determinism, and free background learning.

# Frontier (Opus-class) per-token pricing for the saved-vs-frontier estimate.
FRONTIER_IN_PER_TOK = 15.0 / 1_000_000
FRONTIER_OUT_PER_TOK = 75.0 / 1_000_000
# _SEP and _CTX_COLORS are set by set_palette() (they depend on the active theme).


def tokens_of(p: dict) -> tuple[int, int]:
    """(prompt, completion) tokens from a MODEL_CALL payload's usage block."""
    u = (p.get("response") or {}).get("usage") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def frontier_saved(prompt_tok: int, completion_tok: int) -> float:
    """What these tokens would have cost on an Opus-class API. Local spend is $0."""
    return prompt_tok * FRONTIER_IN_PER_TOK + completion_tok * FRONTIER_OUT_PER_TOK


def ability_glyphs(caps) -> list[str]:
    g: list[str] = []
    if caps.logprobs:
        g.append("logprobs")
    if caps.grammar:
        g.append("grammar")
    if caps.sampler_zoo:
        g.append(f"zoo({len(caps.sampler_zoo)})")
    if caps.kv_snapshot or caps.parallel_n:
        g.append("fork")
    return g


def _money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.005 else "$0.00"


def status_bar(
    *,
    preset: str,
    tier: int,
    glyphs: list[str],
    saved: float,
    deterministic: bool,
    learn: str,
    ctx: tuple[str, float | None] | None = None,
    vim: str | None = None,
) -> Text:
    """The permanent status line above the input — the harness's edge, always on.
    `ctx` is an optional (label, frac) context gauge; `vim` is the vim mode when on."""
    out = Text()
    out.append(f" {preset} ", style=f"bold {INK} on {GOLD}")  # mode chip (gold)
    if vim is not None:  # vim mode indicator (only when vim is enabled)
        out.append(
            f" {vim.upper()} ",
            style=f"bold {INK} on {(SAKURA if vim == 'normal' else JADE_DEEP)}",
        )
    out.append(*_SEP)
    out.append(f"tier {tier}", style="bold " + C_OK)
    if glyphs:
        out.append(" ◈ ", style=C_OK)
        out.append(" ".join(glyphs), style=C_DIM)
    if ctx is not None:
        label, frac = ctx
        color = (
            C_OK
            if (frac is None or frac < 0.70)
            else (C_TOOL if frac < 0.85 else C_ERR)
        )
        out.append(*_SEP)
        out.append("ctx ", style=C_DIM)
        out.append(label, style="bold " + color)
    out.append(*_SEP)
    out.append("$0.00 spent", style="bold " + C_OK)  # the signature segment
    out.append(f" · ~{_money(saved)} saved", style=C_OK)
    out.append(*_SEP)
    if deterministic:
        out.append("⟲ deterministic", style=C_OK)
    else:
        out.append("⟲ best-effort", style=C_TOOL)
    out.append(*_SEP)
    if learn == "learning":
        out.append("⚛ learning…", style=C_RESAMPLE)
    elif learn == "idle":
        out.append("⚛ idle-learn", style=C_OK)
    else:
        out.append("⚛ off", style=C_DIM)
    return out


def _feats(caps) -> list[str]:
    feats: list[str] = []
    if caps.seed and caps.logprobs:
        feats += ["replay", "logprobs"]
    elif caps.seed:
        feats.append("seeded")
    if caps.grammar:
        feats.append("grammar")
    if caps.sampler_zoo:
        feats.append(f"sampler-zoo({len(caps.sampler_zoo)})")
    if caps.raw_completion:
        feats.append("think-budget")
    if caps.banned_strings or caps.raw_completion:
        feats.append("anti-slop")
    if caps.kv_snapshot or caps.parallel_n:
        feats.append("KV-fork")
    return feats


def _row(label: str, value: Text | str, value_style: str = "") -> Text:
    line = Text("  ")
    line.append(f"{label:<13}", style=C_DIM)
    if isinstance(value, str):
        line.append(value, style=value_style)
    else:
        line.append_text(value)
    return line


def banner_body(
    model: str,
    caps,
    *,
    tools: list[str],
    skills: list[str],
    memory_summary: str,
    preset_name: str,
    preset_blurb: str,
) -> RenderableType:
    """Hermes-style compact banner content — what this server unlocks, the loaded
    tools/skills/memory, and the economics, in a few aligned rows."""
    rows: list[RenderableType] = []
    head = Text()
    head.append(f"{model}", style="bold")
    head.append(f"   {caps.server} · tier {caps.tier()}", style=C_DIM)
    rows.append(head)
    feats = Text(" · ".join(_feats(caps)) or "basic agent loop", style=C_OK)
    rows.append(_row("capabilities", feats))
    if tools:
        shown = " ".join(tools[:9]) + (f"  +{len(tools) - 9}" if len(tools) > 9 else "")
        rows.append(_row(f"tools ({len(tools)})", shown, C_TOOL))
    rows.append(
        _row(
            f"skills ({len(skills)})",
            " ".join(sorted(skills)) if skills else "none",
            C_USER if skills else C_DIM,
        )
    )
    if memory_summary:
        rows.append(_row("memory", memory_summary, C_REASON))
    rows.append(_row("preset", f"{preset_name} — {preset_blurb}", C_DIM))
    econ = Text()
    econ.append("$0.00 spent", style="bold " + C_OK)
    econ.append(" · same work on a frontier API would meter every token", style=C_DIM)
    rows.append(_row("economics", econ))
    return Group(*rows)


def _conf_bar(bands: list[int], width: int = 28) -> Text:
    """A confident/unsure/flagged distribution bar from token counts."""
    total = sum(bands) or 1
    segs = [round(b / total * width) for b in bands]
    out = Text()
    for n, style in zip(segs, (C_OK, JADE_MUTE, C_ERR)):
        out.append("█" * n, style=style)
    return out


def usage_panel(
    stats: dict,
    *,
    model: str,
    tier: int,
    deterministic: bool,
    learn_summary: str | None,
) -> RenderableType:
    """The ^u overlay — Hermes' rich token/cost/context panel, our way: $0 spent,
    what was saved, the confidence distribution, resamples, determinism, learning."""
    body = Text()
    saved, calls, toks = stats["saved"], stats["calls"], stats["tokens"]
    body.append("cost\n", style="bold")
    body.append("  $0.00 spent", style="bold " + C_OK)
    body.append(
        f"   ~{_money(saved)} the same {toks:,} tokens would cost on an "
        f"Opus-class API\n",
        style=C_DIM,
    )
    body.append(
        f"  across {calls} model call(s) — local marginal cost is zero, so "
        "best-of-N and background learning are free.\n\n",
        style=C_DIM,
    )

    body.append("determinism\n", style="bold")
    if deterministic:
        body.append("  ⟲ bit-exact & replayable", style=C_OK)
        body.append(
            "   every call logs its seed; ^r re-runs identically.\n\n", style=C_DIM
        )
    else:
        body.append("  ⟲ best-effort", style=C_TOOL)
        body.append(
            "   this server doesn't pin a seed — replay may drift.\n\n", style=C_DIM
        )

    body.append("token surprisal\n", style="bold")
    bands = stats["conf"]
    if sum(bands) and stats.get("mean_lp") is not None:
        body.append("  ")
        body.append_text(_conf_bar(bands))
        body.append(
            f"  mean logprob {stats['mean_lp']:+.2f}\n",
            style=confidence_style(stats["mean_lp"]),
        )
        body.append(
            f"  {bands[0]} expected · {bands[1]} surprising · {bands[2]} very-surprising "
            "tokens\n",
            style=C_DIM,
        )
        body.append(
            "  surface-form only — NOT a correctness signal. Real uncertainty comes "
            "from sample agreement (semantic entropy).\n",
            style=C_DIM,
        )
    else:
        body.append("  no logprobs captured yet\n", style=C_DIM)
    body.append(f"  ↻ {stats['resamples']} resample(s)", style=C_RESAMPLE)
    body.append("   re-tried when tokens were low-probability.\n\n", style=C_DIM)

    body.append("background learning\n", style="bold")
    body.append(
        f"  {learn_summary or 'idle — consolidation runs free between turns.'}\n",
        style=C_OK if learn_summary else C_DIM,
    )
    return Panel(
        body,
        title="✦ usage & advantages",
        title_align="left",
        subtitle=f"{model} · tier {tier}",
        subtitle_align="right",
        border_style=B_ACCENT,
        padding=(1, 2),
    )


def _cand_field(c, field: str, default):
    """Candidates are either Candidate objects (in-process) or plain dicts (from
    the server's /session/plan endpoint) — read either shape."""
    if isinstance(c, dict):
        return c.get(field, default)
    return getattr(c, field, default)


# ── context-window visualizer (/context) ───────────────────────────────────
# A segmented bar of what's in the model's context right now — system · task ·
# assistant · tool I/O · free — with the 85% auto-compaction line marked. Source
# of truth is the latest MODEL_CALL's request body (exactly what was sent).


def _ktok(n: int) -> str:
    """Humanize a token count: 511 · 1.2k · 262k."""
    if n < 1000:
        return str(int(n))
    if n < 100_000:
        return f"{n / 1000:.1f}k"
    return f"{round(n / 1000)}k"


def _dict_msg_tokens(msg: dict) -> int:
    n = len(msg.get("content") or "")
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        n += len(fn.get("name", "")) + len(fn.get("arguments", "")) + 16
    if msg.get("name"):
        n += len(msg["name"])
    return n // 4


def context_breakdown(
    messages: list[dict], tools: list[dict] | None = None
) -> list[tuple[str, int]]:
    """Bucket the messages actually sent (the logged request body) into the five
    coarse categories, in tokens. Tool *schemas* count as fixed system overhead."""
    buckets = {"system": 0, "task": 0, "assistant": 0, "tool I/O": 0}
    for m in messages or []:
        role = m.get("role")
        t = _dict_msg_tokens(m)
        if role == "system":
            buckets["system"] += t
        elif role == "user":
            buckets["task"] += t
        elif role == "assistant":
            buckets["assistant"] += t
        elif role == "tool":
            buckets["tool I/O"] += t
    if tools:
        import json as _json

        buckets["system"] += len(_json.dumps(tools)) // 4
    return [(k, buckets[k]) for k in ("system", "task", "assistant", "tool I/O")]


def context_used(messages: list[dict], tools: list[dict] | None = None) -> int:
    return sum(t for _, t in context_breakdown(messages, tools))


def context_panel(
    breakdown: list[tuple[str, int]],
    *,
    window: int | None,
    trigger_frac: float = 0.85,
    width: int = 44,
) -> RenderableType:
    """The segmented-bar /context view (the style chosen): one bar across the whole
    window, colored per category, free space dim, with the compaction line marked."""
    used = sum(t for _, t in breakdown)
    parts: list[RenderableType] = []
    if window:
        pct = round(used / window * 100)
        parts.append(
            Text.assemble(
                (f"{_ktok(used)} ", "bold " + C_ANSWER),
                (f"/ {_ktok(window)} ", C_DIM),
                (
                    f"({pct}%)",
                    "bold " + (C_OK if pct < 70 else C_TOOL if pct < 85 else C_ERR),
                ),
            )
        )
        cells: list[tuple[str, str]] = []
        for label, tokens in breakdown:
            cells += [("█", _CTX_COLORS[label])] * int(round(tokens / window * width))
        cells = cells[:width] + [("░", C_DIM)] * max(0, width - len(cells))
        cells = cells[:width]
        trig = min(width - 1, int(round(trigger_frac * width)))
        cells[trig] = ("┊", "bold " + C_SAKURA)  # auto-compaction line
        bar = Text()
        for ch, st in cells:
            bar.append(ch, st)
        parts.append(bar)
        parts.append(
            Text.assemble(
                (" " * trig, ""),
                ("↑ auto-compaction at ", C_DIM),
                (f"{trigger_frac:.0%} ", "bold " + C_SAKURA),
                (f"({_ktok(int(window * trigger_frac))})", C_DIM),
            )
        )
    else:
        parts.append(
            Text.assemble(
                (f"{_ktok(used)} used", "bold " + C_ANSWER),
                ("  · context window not reported by this server", C_DIM),
            )
        )
    legend = Text()
    for i, (label, tokens) in enumerate(breakdown):
        if i:
            legend.append("   ")
        legend.append("■ ", _CTX_COLORS[label])
        legend.append(f"{label} ", C_ANSWER)
        legend.append(_ktok(tokens), C_DIM)
    parts.append(Text(""))
    parts.append(legend)
    return Panel(
        Group(*parts),
        title="✦ context",
        title_align="left",
        border_style=B_ACCENT,
        padding=(1, 2),
    )


def memory_panel(
    sections: list[tuple[str, str]], recall: list[tuple[str, str]] | None = None
) -> RenderableType:
    """The /memory view: the USER/MEMORY/PROJECT files' contents, plus the top
    recall hits relevant to the current task."""
    body: list[RenderableType] = []
    for title, content in sections:
        body.append(Text(title, style="bold " + C_GOLD))
        text = content.strip()
        body.append(
            Padding(
                Text(text or "(empty)", style=C_ANSWER if text else C_DIM), (0, 0, 1, 2)
            )
        )
    if recall:
        body.append(Text("relevant past runs (recall)", style="bold " + C_SAKURA))
        for kind, text in recall:
            body.append(
                Padding(
                    Text.assemble((f"[{kind}] ", C_DIM), (text, C_ANSWER)), (0, 0, 0, 2)
                )
            )
    return Panel(
        Group(*body),
        title="✦ memory",
        title_align="left",
        border_style=B_ACCENT,
        padding=(1, 2),
    )


def findings_panel(text: str, kind: str) -> RenderableType:
    """Grammar-validated review findings: each line is `file:line — severity —
    issue`. Colored by severity. The grammar guarantees this shape."""
    sev_color = {
        "blocker": ROSE,
        "critical": ROSE,
        "major": AMBER,
        "high": AMBER,
        "minor": GOLD,
        "medium": GOLD,
        "nit": JADE_GREY,
        "low": JADE_GREY,
    }
    body = Text()
    if text.strip() == "No findings.":
        body.append("✓ No findings.", style="bold " + JADE)
    else:
        for line in text.strip().splitlines():
            parts = line.split(" — ", 2)
            if len(parts) == 3:
                loc, sev, desc = parts
                color = sev_color.get(sev.strip().lower(), C_DIM)
                body.append(f"  {loc}  ", style="bold " + C_GOLD)
                body.append(sev.strip(), style="bold " + color)
                body.append(f"  {desc}\n", style=C_ANSWER)
            elif line.strip():
                body.append("  " + line + "\n", style=C_DIM)
    title = "✦ security findings" if kind == "security-review" else "✦ review findings"
    return Panel(
        body,
        title=title + "  (grammar-validated)",
        title_align="left",
        border_style=B_ACCENT,
        padding=(1, 2),
    )


def plan_fork_panel(candidates: list) -> RenderableType:
    """Render best-of-N plan candidates, best first — the KV-fork search made visible.
    The chosen plan is shown as Markdown; runners-up are listed compactly."""
    parts: list[RenderableType] = [
        Text(
            f"forked {len(candidates)} candidate plans, ranked by a verifier "
            "(free on the local prefix-cache):",
            style=C_DIM,
        )
    ]
    for i, c in enumerate(candidates):
        score = _cand_field(c, "score", 0.0)
        text = str(_cand_field(c, "text", str(c))).strip()
        if i == 0:
            parts.append(
                Text.assemble(
                    ("\n★ chosen", "bold " + C_OK), (f"   score {score:+.2f}", C_DIM)
                )
            )
            parts.append(
                plan_artifact(
                    text, title="winning plan", subtitle=f"score {score:+.2f}"
                )
            )
        else:
            parts.append(
                Text.assemble(
                    (f"\n  #{i + 1}", "bold " + C_DIM), (f"  score {score:+.2f}", C_DIM)
                )
            )
            parts.append(Padding(Text(_truncate(text, 240), style=C_DIM), (0, 0, 0, 4)))
    return Panel(
        Group(*parts),
        title="⑂ plan-fork",
        title_align="left",
        border_style=B_OK,
        padding=(1, 2),
    )


def chat_render_event(ev: Event, show_thinking: bool = True) -> RenderableType | None:
    """One event -> one chat-stream element (None = skip)."""
    p = ev.payload
    if ev.type == RUN_STARTED:
        return Text.assemble(
            ("\n› ", "bold " + C_USER), (p.get("task", ""), "bold " + C_USER)
        )
    if ev.type == USER_MESSAGE:  # a follow-up turn — show what the user said
        return Text.assemble(
            ("\n› ", "bold " + C_USER), (p.get("content", ""), "bold " + C_USER)
        )
    if ev.type == MODEL_CALL:
        return chat_assistant(p, show_thinking=show_thinking)
    if ev.type == TOOL_CALL:
        return chat_tool(p)
    if ev.type == POLICY_TRIGGERED:
        return Text.assemble(
            ("  ↻ ", C_RESAMPLE),
            (f"{p.get('action')}: {p.get('reason')} — trying again", C_RESAMPLE),
        )
    if ev.type == GUARDRAIL:
        return _guardrail(p)
    if ev.type == RUN_FAILED:
        return Text(f"  ✗ {p.get('error')}", style="bold " + ROSE)
    return None  # RUN_COMPLETED: the answer is already the last assistant turn
