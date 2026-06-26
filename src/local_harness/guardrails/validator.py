"""Response validation: rescue tool calls from text, reject unknown tools and
malformed arguments (after forge, MIT). Stateless."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..inference.types import Message, ToolCallRequest
from .nudges import Nudge, bad_args_nudge, retry_nudge, unknown_tool_nudge
from .rescue import rescue_tool_calls


@dataclass
class ValidationResult:
    """Exactly one of tool_calls / nudge / final is meaningful:
    - tool_calls: validated calls (possibly rescued from text), execute them
    - nudge: corrective message to inject, then re-prompt
    - final=True: a legitimate plain-text answer (no tools involved)
    """

    tool_calls: list[ToolCallRequest] | None = None
    nudge: Nudge | None = None
    final: bool = False
    rescued: bool = False


class ResponseValidator:
    def __init__(self, tool_names: list[str], rescue_enabled: bool = True,
                 text_is_final: bool = True):
        self.tool_names = tool_names
        self.rescue_enabled = rescue_enabled
        # In this harness a bare text response normally *is* the final answer.
        # Set False for workflows that must always end via a terminal tool.
        self.text_is_final = text_is_final

    def validate(self, message: Message) -> ValidationResult:
        if not message.tool_calls:
            text = message.content or ""
            if self.rescue_enabled:
                rescued = rescue_tool_calls(text, self.tool_names)
                if rescued:
                    return ValidationResult(tool_calls=rescued, rescued=True)
            if self.text_is_final and text.strip():
                return ValidationResult(final=True)
            return ValidationResult(
                nudge=Nudge(role="user", content=retry_nudge(), kind="retry")
            )

        unknown = [tc for tc in message.tool_calls if tc.name not in self.tool_names]
        if unknown:
            return ValidationResult(nudge=Nudge(
                role="tool", kind="unknown_tool", tool_call_id=unknown[0].id,
                content=unknown_tool_nudge(unknown[0].name, self.tool_names),
            ))

        for tc in message.tool_calls:
            args = tc.arguments.strip()
            if args:
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    parsed = None
                if not isinstance(parsed, dict):
                    return ValidationResult(nudge=Nudge(
                        role="tool", kind="bad_args", tool_call_id=tc.id,
                        content=bad_args_nudge(tc.name, tc.arguments),
                    ))
        return ValidationResult(tool_calls=list(message.tool_calls))
