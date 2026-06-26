"""Context compaction.

Two strategies, both in-flight only — the event log keeps full history, so
replay and resume are unaffected:

1. `summarize_and_compact` (default, Claude-Code-parity): when the running
   transcript crosses the auto-compact trigger, replace the old turns with a
   single structured summary produced by the model itself, mirroring Claude
   Code's `/compact` behaviour (`src/services/compact/autoCompact.ts`). The
   summary follows Claude Code's eight-section schema (see SUMMARY_PROMPT) so
   nothing load-bearing — intent, files touched, errors, pending work — is lost.

2. `compact` (mechanical fallback, priority scheme after forge, MIT): no model
   call. Cut order (first→last): ephemeral nudges, then tool-result bodies, then
   whole old tool exchanges (collapsed into a one-line recap). Used when there is
   no client, or if the summary call fails — compaction must never crash a run.

Never touched by either: the system prompt, the task message, and the most
recent `keep_recent` messages.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..inference.types import GenerationRequest, Message, SamplingParams

NUDGE_NAME = "nudge"  # marker set on injected guardrail nudges
SUMMARY_NAME = "summary"  # marker set on an injected compaction summary

# Mirrors Claude Code's compaction prompt (the eight-section schema its
# autoCompact pass asks the model to produce). Keeping the schema verbatim is
# the point: it's what makes the summary preserve intent, touched files, errors,
# and pending work rather than just shrinking bytes.
SUMMARY_PROMPT = """\
Your task is to create a detailed summary of the conversation so far, paying \
close attention to the user's explicit requests and your previous actions. This \
summary will REPLACE the older turns in the context window, so it must capture \
everything needed to continue the work without loss.

Write the summary using exactly these eight sections:

1. Primary Request and Intent: the user's explicit requests and overall goal, in detail.
2. Key Technical Concepts: technologies, libraries, patterns, and decisions that matter.
3. Files and Code Sections: every file read or modified, why it matters, and the key code.
4. Errors and fixes: each error hit and how it was resolved (and any user correction).
5. Problem Solving: problems solved and the reasoning, plus open threads.
6. All user messages: list every non-tool user message verbatim or near-verbatim.
7. Pending Tasks: what remains to be done.
8. Current Work: precisely what was being done at the moment of this summary.

Output ONLY the summary in the eight sections above. Do not ask questions or add commentary."""


def _render_for_summary(messages: list[Message]) -> str:
    """Flatten old turns into a plain transcript the summarizer can read."""
    lines: list[str] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            calls = "; ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls)
            body = (m.content or "").strip()
            lines.append(f"[assistant] {body}\n  →calls: {calls}" if body
                         else f"[assistant] →calls: {calls}")
        elif m.role == "tool":
            lines.append(f"[tool:{m.name}] {(m.content or '').strip()}")
        else:
            lines.append(f"[{m.role}] {(m.content or '').strip()}")
    return "\n".join(lines)


def recent_window(tail: list[Message], keep_recent: int) -> list[Message]:
    """The last `keep_recent` messages, but never starting on an orphaned `tool`
    message: once the old turns are summarized away, a leading tool reply has no
    matching assistant tool_call and strict servers (vLLM) 400 on it. Drop such
    leading tool messages (they're represented in the summary)."""
    recent = tail[-keep_recent:] if keep_recent else []
    while recent and recent[0].role == "tool":
        recent = recent[1:]
    return recent


def estimate_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += len(m.content or "")
        for tc in m.tool_calls:
            total += len(tc.name) + len(tc.arguments) + 16
    return total // 4


def compact(
    messages: list[Message], budget_tokens: int, keep_recent: int = 6
) -> tuple[list[Message], int]:
    """Returns (messages, phase): phase 0 = untouched, 1 = nudges dropped,
    2 = old tool results truncated, 3 = old tool exchanges collapsed."""
    if estimate_tokens(messages) <= budget_tokens:
        return messages, 0

    head, tail = messages[:2], messages[2:]  # system + task are sacred
    recent = recent_window(tail, keep_recent)
    old = tail[: len(tail) - len(recent)]

    # Phase 1: drop guardrail nudges outside the recent window.
    old = [m for m in old if m.name != NUDGE_NAME]
    if estimate_tokens(head + old + recent) <= budget_tokens:
        return head + old + recent, 1

    # Phase 2: truncate old tool results to their first line.
    truncated = []
    for m in old:
        if m.role == "tool" and m.content and "\n" in m.content:
            truncated.append(Message(role="tool", content=m.content.split("\n", 1)[0] + " …",
                                     tool_call_id=m.tool_call_id, name=m.name))
        else:
            truncated.append(m)
    old = truncated
    if estimate_tokens(head + old + recent) <= budget_tokens:
        return head + old + recent, 2

    # Phase 3: collapse old assistant-tool exchanges into a one-line recap,
    # preserving old plain-assistant messages (the model's reasoning context).
    recap_lines: list[str] = []
    kept: list[Message] = []
    for m in old:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                recap_lines.append(f"{tc.name}({tc.arguments[:60]})")
        elif m.role == "tool":
            if recap_lines:
                recap_lines[-1] += f" -> {(m.content or '')[:60]}"
        else:
            kept.append(m)
    recap: list[Message] = []
    if recap_lines:
        recap = [Message(role="user", name="recap",
                         content="[Earlier tool activity: " + "; ".join(recap_lines) + "]")]
    return head + recap + kept + recent, 3


# Progress callback: invoked with phase ∈ {"start","tick","done"} and a dict of
# {frac, before_tokens, after_tokens, context_window, method}. A CLI renders a
# bar from it; headless callers pass None and pay nothing.
CompactCallback = Callable[[str, dict[str, Any]], None]


async def summarize_and_compact(
    client,
    messages: list[Message],
    *,
    trigger_tokens: int,
    context_window: int | None = None,
    keep_recent: int = 6,
    seed: int = 0,
    on_compact: CompactCallback | None = None,
    extra_instructions: str | None = None,
) -> tuple[list[Message], dict[str, Any]]:
    """Claude-Code-parity compaction: summarize the old turns with a model call.

    Returns (new_messages, info) where info is the payload for a CONTEXT_COMPACTED
    event: {method, before_tokens, after_tokens, trigger_tokens, summary}. The
    model call is plain (no tools, no logprobs) and stays off the agent's
    determinism path — it's logged as its own event type, not a MODEL_CALL, so
    replay/resume rebuild from the full history exactly as before.

    Robustness: any failure (server error, empty summary) falls back to the
    mechanical `compact` — compaction must never crash a run."""
    before = estimate_tokens(messages)
    info: dict[str, Any] = {
        "method": "summarize", "before_tokens": before, "after_tokens": before,
        "trigger_tokens": trigger_tokens, "context_window": context_window,
        "summary": None,
    }
    if on_compact:
        on_compact("start", dict(info, frac=0.0))

    head, tail = messages[:2], messages[2:]
    recent = recent_window(tail, keep_recent)
    old = tail[: len(tail) - len(recent)]
    if not old:  # nothing old enough to summarize — fall back to mechanical
        new_messages, phase = compact(messages, trigger_tokens, keep_recent)
        info.update(method="mechanical", after_tokens=estimate_tokens(new_messages),
                    summary=None, phase=phase)
        if on_compact:
            on_compact("done", dict(info, frac=1.0))
        return new_messages, info

    import asyncio

    async def _ticker() -> None:
        # Indeterminate, time-based bar — like Claude Code's, the real summary
        # call is one opaque await, so we ease toward 90% and snap to 100% after.
        frac = 0.0
        try:
            while True:
                await asyncio.sleep(0.12)
                frac = min(0.9, frac + (0.9 - frac) * 0.12 + 0.01)
                if on_compact:
                    on_compact("tick", dict(info, frac=frac))
        except asyncio.CancelledError:
            return

    ticker = asyncio.ensure_future(_ticker())
    try:
        prompt = SUMMARY_PROMPT if not extra_instructions else \
            f"{SUMMARY_PROMPT}\n\nAdditional instructions: {extra_instructions}"
        request = GenerationRequest(
            messages=[
                Message(role="system", content=prompt),
                Message(role="user",
                        content="Here is the conversation to summarize:\n\n"
                                + _render_for_summary(head[1:] + old)),
            ],
            sampling=SamplingParams(temperature=0.0, seed=seed),  # no token cap: tokens are free
        )
        resp = await client.chat_body(request.to_body(client.model))
        summary = resp.text.strip()
        if not summary:  # reasoning-only model: recover text from the thinking channel
            raw = (resp.raw.get("choices") or [{}])[0].get("message", {})
            summary = (raw.get("reasoning_content") or raw.get("reasoning") or "").strip()
        if not summary:
            raise ValueError("summary call returned no text")
    except Exception as exc:  # noqa: BLE001 — compaction must never crash a run
        ticker.cancel()
        new_messages, phase = compact(messages, trigger_tokens, keep_recent)
        info.update(method="mechanical", after_tokens=estimate_tokens(new_messages),
                    summary=None, phase=phase, fallback_reason=f"{type(exc).__name__}: {exc}")
        if on_compact:
            on_compact("done", dict(info, frac=1.0))
        return new_messages, info
    finally:
        ticker.cancel()

    summary_msg = Message(
        role="user", name=SUMMARY_NAME,
        content="[Summary of the earlier conversation, compacted to free context:]\n\n" + summary,
    )
    new_messages = head + [summary_msg] + recent
    info.update(after_tokens=estimate_tokens(new_messages), summary=summary)
    if on_compact:
        on_compact("done", dict(info, frac=1.0))
    return new_messages, info
