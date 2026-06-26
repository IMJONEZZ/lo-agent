"""Skill induction: mine the event log for recurring successful output shapes
and propose draft grammar skills.

Deterministic, model-free: if N+ completed runs produced JSON answers with the
same key set, that's a recurring spec — emit a json_schema skill draft to
skills/drafts/ for human review. The draft is validated by loading it through
the real skill loader before it's written.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..events.log import RUN_COMPLETED, EventLog
from ..skills.skill import load_skill_toml


def _json_shape(answer: str) -> tuple[str, ...] | None:
    try:
        data = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict) and data:
        return tuple(sorted(data.keys()))
    return None


def _draft_toml(name: str, keys: tuple[str, ...], samples: list[dict]) -> str:
    props = []
    for key in keys:
        values = [s[key] for s in samples]
        if all(isinstance(v, bool) for v in values):
            t = "boolean"
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            t = "number"
        elif all(isinstance(v, list) for v in values):
            t = "array"
        else:
            t = "string"
        props.append(f'[grammar.json_schema.properties.{key}]\ntype = "{t}"')
    required = ", ".join(f'"{k}"' for k in keys)
    return (
        f'[skill]\nname = "{name}"\n'
        f'description = "DRAFT (induced from {len(samples)} successful runs) — review before use"\n'
        f'system_prompt = "Reply with only the JSON object."\n\n'
        f"[grammar.json_schema]\ntype = \"object\"\nrequired = [{required}]\n\n"
        + "\n\n".join(props) + "\n\n[sampling]\ntemperature = 0.2\nmax_tokens = 256\n"
    )


def induce_skills(
    log: EventLog,
    drafts_dir: str | Path,
    min_count: int = 3,
) -> list[Path]:
    """Returns paths of draft skill files written (idempotent per shape)."""
    shapes: dict[tuple[str, ...], list[dict]] = {}
    for run in log.runs():
        if run.status != "completed":
            continue
        completed = log.events(run.run_id, type=RUN_COMPLETED)
        if not completed:
            continue
        answer = completed[-1].payload.get("answer", "")
        shape = _json_shape(answer)
        if shape:
            shapes.setdefault(shape, []).append(json.loads(answer))

    drafts_dir = Path(drafts_dir)
    written: list[Path] = []
    for keys, samples in shapes.items():
        if len(samples) < min_count:
            continue
        name = "induced_" + "_".join(keys)[:40]
        path = drafts_dir / f"{name}.toml"
        if path.exists():
            continue
        toml_text = _draft_toml(name, keys, samples)
        drafts_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(toml_text)
        load_skill_toml(path)  # must compile through the real loader
        written.append(path)
    return written
