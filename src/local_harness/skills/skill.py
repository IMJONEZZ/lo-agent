"""Skill = grammar (the executable spec) + prompt + logit-pipeline config.

TOML format:

    [skill]
    name = "sql_select"
    description = "Generate a valid SQL SELECT statement"
    imports = ["sql_core"]            # merge rules from sibling skills
    system_prompt = "..."             # optional

    [grammar]                         # either EBNF rules...
    root = "select_stmt"
    [grammar.rules]
    select_stmt = '"SELECT " columns " FROM " ident ";"'

    [grammar.json_schema]             # ...or a JSON schema instead
    # type = "object" ...

    [sampling]                        # SamplingParams overrides
    temperature = 0.1
    [sampling.samplers]               # normalized sampler-zoo settings
    min_p = 0.05

    [bias]
    profile = "concise"               # named BiasProfile applied with the skill
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ir import Grammar, GrammarError


@dataclass
class Skill:
    name: str
    description: str = ""
    system_prompt: str | None = None
    grammar: Grammar | None = None
    json_schema: dict[str, Any] | None = None
    sampling_overrides: dict[str, Any] = field(default_factory=dict)
    samplers: dict[str, Any] = field(default_factory=dict)
    bias_profile: str | None = None
    adapter: str | None = None  # LoRA adapter to hot-swap in for this skill ("name" or "name=path")
    imports: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)  # optimized prompts etc.

    def validate_output(self, text: str) -> bool:
        if self.json_schema is not None:
            import json

            try:
                instance = json.loads(text)
            except json.JSONDecodeError:
                return False
            return _check_schema(instance, self.json_schema)
        if self.grammar is not None:
            return self.grammar.validate(text)
        return True


def _check_schema(instance: Any, schema: dict[str, Any]) -> bool:
    """Minimal JSON-schema check: type, required, properties, items, enum."""
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "number": (int, float),
             "integer": int, "boolean": bool, "null": type(None)}
    if t and not isinstance(instance, types[t]):
        return False
    if t == "object":
        for key in schema.get("required", []):
            if key not in instance:
                return False
        for key, sub in schema.get("properties", {}).items():
            if key in instance and not _check_schema(instance[key], sub):
                return False
    if t == "array":
        sub = schema.get("items")
        if sub and not all(_check_schema(i, sub) for i in instance):
            return False
    if "enum" in schema and instance not in schema["enum"]:
        return False
    return True


def load_skill_toml(path: str | Path, registry: "SkillRegistry | None" = None) -> Skill:
    path = Path(path)
    data = tomllib.loads(path.read_text())
    meta = data.get("skill", {})
    sampling = dict(data.get("sampling", {}))
    samplers = sampling.pop("samplers", {})

    grammar = None
    json_schema = None
    g = data.get("grammar", {})
    has_imports = bool(meta.get("imports"))
    if "json_schema" in g:
        json_schema = g["json_schema"]
    elif "rules" in g:
        grammar = Grammar.from_rules(g["rules"], root=g["root"], check=not has_imports)

    skill = Skill(
        name=meta.get("name", Path(path).stem),
        description=meta.get("description", ""),
        system_prompt=meta.get("system_prompt"),
        grammar=grammar,
        json_schema=json_schema,
        sampling_overrides=sampling,
        samplers=samplers,
        bias_profile=data.get("bias", {}).get("profile"),
        adapter=data.get("lora", {}).get("adapter") or meta.get("adapter"),
        imports=meta.get("imports", []),
        metadata=data.get("metadata", {}),
    )
    if skill.imports:
        if registry is None:
            raise GrammarError(f"skill {skill.name!r} has imports but no registry to resolve them")
        for dep_name in skill.imports:
            dep = registry.get(dep_name)
            if skill.grammar is not None and dep.grammar is not None:
                skill.grammar = skill.grammar.merge(dep.grammar)
        if skill.grammar is not None:
            skill.grammar.check()  # deferred from load: imports are merged now

    # Optimizer output (FewShotProgram.save) rides along as a sidecar file.
    sidecar = path.with_suffix(".optimized.json")
    if sidecar.exists():
        import json

        skill.metadata["optimized"] = json.loads(sidecar.read_text())
    return skill


class SkillRegistry:
    def __init__(self, skill_dir: str | Path | None = None):
        self._skills: dict[str, Skill] = {}
        self.skill_dir = Path(skill_dir) if skill_dir else None
        if self.skill_dir and self.skill_dir.is_dir():
            # Two passes so imports resolve regardless of load order.
            paths = sorted(self.skill_dir.glob("*.toml"))
            deferred = []
            for p in paths:
                try:
                    self.register(load_skill_toml(p, registry=self))
                except (GrammarError, KeyError):
                    deferred.append(p)
            for p in deferred:
                self.register(load_skill_toml(p, registry=self))

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"unknown skill: {name!r} (have: {sorted(self._skills)})")
        return self._skills[name]

    def names(self) -> list[str]:
        return sorted(self._skills)
