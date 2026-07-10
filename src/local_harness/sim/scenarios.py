"""The journey catalog: what a user actually does in the TUI, as scenarios.

Prompts are written so the SAME scenario runs against a scripted mock and a
real model: every completion marker is text the model is told to CONSTRUCT
(e.g. "reply with the concatenation of BAN and ANA"), so the marker never
appears in the echoed prompt or a rendered plan — only in the answer. Mock
tests script chat_fn to return exactly those answers (tests/e2e_support.py).

Timeouts here are sized for the mock tier; live runs scale them with
`scaled(scenario, factor)`.
"""

from __future__ import annotations

from dataclasses import replace

from .scenario import Scenario, Step

# Key sequences, PTY-raw; the Pilot driver translates them to key names.
ENTER = "\r"
ESC = "\x1b"
CTRL_T = "\x14"
CTRL_D = "\x04"
CTRL_U = "\x15"  # Input: delete to start (clears a pre-filled field)

# Controlled-answer prompts: the model constructs the marker, so it can't
# match the echoed input. Mock chat_fn returns the same strings.
CHAT_TASK = "Concatenate BAN and ANA and reply with only the result, in uppercase."
CHAT_MARKER = "BANANA"
PLAN_TASK = (
    "Make a plan for creating a file named hello.txt containing the word hello. "
    "The plan's final step must be: reply with the concatenation of FIN and ISHED-OK."
)
BUILD_DONE_MARKER = "FINISHED-OK"
PERM_TASK = (
    "Use the write_file tool to create a file named perm-test.txt containing "
    "PERMOK. If it succeeds, reply with the concatenation of PERM and -GRANTED; "
    "if the tool is denied, reply with the concatenation of PERM and -REFUSED."
)


def _chat_turn(task: str = CHAT_TASK, marker: str = CHAT_MARKER, **kw) -> Step:
    kw.setdefault("label", "chat turn")
    kw.setdefault("timeout", 30.0)
    return Step(task + ENTER, marker=marker, **kw)


def scaled(scenario: Scenario, factor: float) -> Scenario:
    """The same journey with stretched timeouts (for real-model runs)."""
    return replace(
        scenario,
        steps=[replace(s, timeout=s.timeout * factor) for s in scenario.steps],
    )


SCENARIOS: dict[str, Scenario] = {}


def _register(s: Scenario) -> Scenario:
    SCENARIOS[s.name] = s
    return s


_register(
    Scenario(
        "first-run-welcome",
        description="Fresh db: the welcome card shows the model, tier, and unlocked features.",
        needs={"live-ok"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="welcome panel"),
            Step("", marker="type a task", timeout=5.0, label="empty-state hint"),
        ],
    )
)

_register(
    Scenario(
        "chat-turn-streaming",
        description="A plain turn: type a task, watch the streamed answer land.",
        needs={"live-ok"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            _chat_turn(),
        ],
    )
)

_register(
    Scenario(
        "slash-autocomplete",
        description="/he filters the slash menu; Tab completes; the help overlay opens.",
        needs={"live-ok"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("/he", settle=0.3, label="menu filters"),
            Step("\t" + ENTER, marker="keys & commands", timeout=10.0, label="complete + run /help"),
            Step(ESC, settle=0.3, label="close help"),
        ],
    )
)

_register(
    Scenario(
        "shell-mode-buffering",
        description="! runs a shell command; its output is buffered into the next turn.",
        needs={"live-ok", "fs"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("!echo sim-shell-ok" + ENTER, marker="sim-shell-ok", timeout=15.0,
                 label="shell artifact"),
            _chat_turn(label="turn carrying shell context"),
        ],
    )
)

_register(
    Scenario(
        "plan-approve-build",
        description="Plan mode produces an approvable plan; /approve implements it in build mode.",
        needs={"live-ok", "fs"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("/codemode off" + ENTER, marker="code-mode off", timeout=10.0,
                 label="classic tool calls"),
            Step("/mode plan" + ENTER, marker="agent preset: plan", timeout=10.0,
                 label="switch to plan"),
            Step(PLAN_TASK + ENTER, marker="plan ready", timeout=60.0, label="plan artifact"),
            Step("/approve" + ENTER, marker="approved → build mode", timeout=10.0,
                 label="approve"),
            Step("", marker=BUILD_DONE_MARKER, timeout=60.0, label="build turn finishes"),
        ],
    )
)

_register(
    Scenario(
        "permission-allow",
        description="An ask-tier tool raises the modal over HTTP; allowing runs it.",
        needs={"live-ok", "interactive-permissions", "fs"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("/codemode off" + ENTER, marker="code-mode off", timeout=10.0,
                 label="classic tool calls"),
            Step(PERM_TASK + ENTER, on_permission="allow", marker="PERM-GRANTED",
                 timeout=60.0, label="allow write_file"),
        ],
    )
)

_register(
    Scenario(
        "permission-deny",
        description="Denying the modal denies the tool; the file is never written.",
        needs={"live-ok", "interactive-permissions", "fs"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("/codemode off" + ENTER, marker="code-mode off", timeout=10.0,
                 label="classic tool calls"),
            Step(PERM_TASK + ENTER, on_permission="deny", marker="PERM-REFUSED",
                 timeout=60.0, label="deny write_file"),
        ],
    )
)

_register(
    Scenario(
        "history-bulk-delete",
        description="Create several conversations, then delete them ALL through the "
        "history sidebar — including the active one — and start fresh.",
        needs={"live-ok"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            _chat_turn(label="turn 1"),
            Step("/new" + ENTER, settle=0.3, label="new conversation"),
            _chat_turn(label="turn 2"),
            Step("/new" + ENTER, settle=0.3, label="new conversation"),
            _chat_turn(label="turn 3"),
            Step(CTRL_T, marker="conversations:", timeout=10.0, label="open history"),
            Step(CTRL_D, marker="deleted run", timeout=10.0, label="delete newest (active)"),
            Step(CTRL_D, marker="deleted run", timeout=10.0, label="delete next"),
            Step(CTRL_D, marker="deleted run", timeout=10.0, label="delete last"),
            Step(CTRL_T, settle=0.3, label="close history"),
            _chat_turn(label="fresh turn after wipe"),
        ],
    )
)

_register(
    Scenario(
        "rewind-picker",
        description="Two turns, then /rewind rolls back to the first answer.",
        needs={"live-ok"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            _chat_turn(label="turn 1"),
            _chat_turn(
                "Concatenate SEC and OND-ANSWER and reply with only the result.",
                "SECOND-ANSWER",
                label="turn 2 (same conversation)",
            ),
            Step("/rewind" + ENTER, marker="Rewind — remove from the chosen point",
                 timeout=10.0, label="picker opens"),
            Step(ENTER, marker="rewound · tail archived", timeout=10.0, label="rewind"),
        ],
    )
)

def _len_run_ids(n: int):
    def check(app) -> None:
        assert len(app._run_ids) == n, (
            f"expected {n} sidebar rows, got {len(app._run_ids)}: {app._run_ids}"
        )

    return check


_register(
    Scenario(
        "history-filter-rename",
        description="'/' filters the history sidebar down to a match; Esc clears; "
        "'r' renames the highlighted conversation.",
        needs={"mock-only"},  # uses app-side row-count checks (Pilot driver)
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            _chat_turn(label="turn 1"),
            Step("/new" + ENTER, settle=0.3, label="new conversation"),
            _chat_turn(
                "Concatenate SEC and OND-ANSWER and reply with only the result.",
                "SECOND-ANSWER",
                label="turn 2",
            ),
            Step(CTRL_T, marker="conversations:", timeout=10.0, label="open history"),
            Step("/", settle=0.3, label="focus the filter"),
            Step("ond-answer", settle=0.5, label="narrow to one row",
                 check=_len_run_ids(1)),
            Step(ESC, settle=0.5, label="clear filter, back to the table",
                 check=_len_run_ids(2)),
            Step("r", marker="rename conversation", timeout=10.0,
                 label="rename modal opens"),
            Step(CTRL_U + "standup notes" + ENTER, marker="renamed", timeout=10.0,
                 label="save the title"),
        ],
    )
)

_register(
    Scenario(
        "inspect-model-calls",
        description="/inspect opens per-model-call stats: timing, tokens, "
        "logprob confidence, finish reason.",
        needs={"live-ok"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            # settle: the marker streams in BEFORE the model call commits to the
            # event log; /inspect too early would truthfully say "no model calls
            # yet" (seen live vs glm-5.2 — the mock commits instantly).
            _chat_turn(settle=3.0),
            Step("/inspect" + ENTER, marker="what each model call cost",
                 timeout=10.0, label="inspector opens"),
            Step(ESC, settle=0.3, label="close"),
        ],
    )
)

_register(
    Scenario(
        "custom-file-command",
        description="A .lo/commands/greet.md file-authored command runs as /greet.",
        needs={"live-ok", "fs"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("/greet" + ENTER, marker="GREET-ACK", timeout=45.0, label="/greet runs"),
        ],
    )
)

# The command file the custom-file-command scenario expects in cwd/.lo/commands/.
GREET_COMMAND_MD = """---
description: simulation smoke command
---
Concatenate GREET and -ACK and reply with only the result.
"""

_register(
    Scenario(
        "mode-preset-switch",
        description="/mode switches presets and says so; unknown names list presets.",
        needs={"live-ok"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("/mode explore" + ENTER, marker="agent preset: explore", timeout=10.0,
                 label="to explore"),
            Step("/mode nosuch" + ENTER, marker="presets:", timeout=10.0,
                 label="unknown lists presets"),
            Step("/mode build" + ENTER, marker="agent preset: build", timeout=10.0,
                 label="back to build"),
        ],
    )
)

_register(
    Scenario(
        "theme-switch",
        description="/theme <name> applies and persists a palette.",
        needs={"live-ok"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("/theme osaka-midnight" + ENTER, marker="theme: osaka-midnight",
                 timeout=10.0, label="switch"),
            Step("/theme osaka-jade" + ENTER, marker="theme: osaka-jade",
                 timeout=10.0, label="back to default"),
        ],
    )
)

_register(
    Scenario(
        "export-transcript",
        description="/export writes run-<id>.md into the cwd.",
        needs={"live-ok", "fs"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            _chat_turn(),
            Step("/export" + ENTER, marker="Markdown transcript", timeout=10.0,
                 label="export"),
        ],
    )
)

_register(
    Scenario(
        "interrupt-mid-stream",
        description="Esc stops a streaming turn without hanging the session.",
        # Needs a genuinely slow stream, so it only means anything against a
        # real model (the mock streams a whole answer in one transport call).
        needs={"live-ok", "slow-stream"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("Count slowly from 1 to 500, one number per line." + ENTER,
                 marker="7", timeout=60.0, label="stream starts"),
            Step(ESC, marker="interrupted — you can", timeout=15.0, label="interrupt"),
        ],
    )
)

_register(
    Scenario(
        "codemode-import-recovery",
        description="A model that reflexively writes `import os` in code mode gets a "
        "teaching error and self-corrects to the tools API within the same turn.",
        needs={"mock-only"},  # requires scripted misbehavior
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("List the files in this project." + ENTER, marker="RECOVERED-OK",
                 timeout=45.0, label="import → teach → recover"),
        ],
    )
)

_register(
    Scenario(
        "codemode-crash-loop-breaker",
        description="A model that NEVER stops writing broken imports is cut off by the "
        "tool-error budget with an on-screen reason — not an infinite retry loop.",
        needs={"mock-only"},
        steps=[
            Step("", marker="unlocked here", timeout=30.0, label="booted"),
            Step("List the files in this project." + ENTER,
                 marker="tool error budget exhausted", timeout=45.0,
                 label="budget stops the loop"),
        ],
    )
)

_register(
    Scenario(
        "upstream-down-notice",
        description="With a dead upstream, submitting a task explains itself instead of hanging.",
        needs={"dead-upstream"},
        steps=[
            Step("", marker="unavailable", timeout=30.0, label="boot notice"),
            # A dead upstream auto-opens the Connect modal at boot; a user who
            # just wants to type dismisses it first.
            Step(ESC, settle=0.5, label="dismiss connect modal"),
            Step("hello" + ENTER, marker="still unreachable", timeout=30.0,
                 label="honest error"),
        ],
    )
)
