import pytest

from local_harness.agent.presets import (
    PRESETS,
    all_preset_names,
    get_preset,
    load_file_presets,
    register_file_presets,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    register_file_presets([])  # start clean; restore built-ins-only afterward
    yield
    register_file_presets([])


def _write(d, name, text):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


def test_build_agent_from_markdown(tmp_path):
    d = tmp_path / ".lo" / "agents"
    _write(d, "tester.md", "---\nmode: build\ntemperature: 0.5\n---\nYou write tests.")
    presets = load_file_presets([d])
    p = presets["tester"]
    assert p.system_prompt == "You write tests."
    assert p.sampling.temperature == 0.5
    assert p.default == "ask"


def test_plan_mode_denies_writes(tmp_path):
    d = tmp_path / ".lo" / "agents"
    _write(d, "architect.md", "---\nmode: plan\n---\nPlan only.")
    p = load_file_presets([d])["architect"]
    assert p.default == "deny"
    for tool in ("write_file", "edit_file", "bash"):
        assert tool in p.deny
    # exposed toolset is read-only
    assert p.exposed() is not None
    assert "write_file" not in p.exposed()


def test_explicit_permission_lists(tmp_path):
    d = tmp_path / ".lo" / "agents"
    _write(d, "web.md", "---\nallow: [read_file, web_search]\nask: [bash]\n---\nWeb agent.")
    p = load_file_presets([d])["web"]
    assert "web_search" in p.allow
    assert p.ask == ["bash"]


def test_get_preset_resolves_file_preset(tmp_path):
    d = tmp_path / ".lo" / "agents"
    _write(d, "custom.md", "---\nmode: build\n---\nCustom agent.")
    register_file_presets([d])
    assert get_preset("custom").system_prompt == "Custom agent."
    assert "custom" in all_preset_names()
    # built-ins still resolve
    assert get_preset("plan").name == "plan"


def test_unknown_falls_back_to_build():
    assert get_preset("does-not-exist").name == "build"


def test_opencode_tools_map_translates_to_real_names(tmp_path):
    # OpenCode's `tools: {write: true, edit: false, bash: true}` map → real tool
    # names, and disabled entries are dropped (not exposed as garbage).
    d = tmp_path / ".opencode" / "agents"
    _write(d, "coder.md", "---\ntools: {write: true, edit: false, bash: true}\n---\nCoder.")
    p = load_file_presets([d])["coder"]
    assert set(p.tools) == {"write_file", "bash"}


def test_empty_tools_list_exposes_nothing(tmp_path):
    # `tools: []` is an explicit "no tools", distinct from an absent key (→ default).
    d = tmp_path / ".lo" / "agents"
    _write(d, "locked.md", "---\ntools: []\n---\nLocked.")
    assert load_file_presets([d])["locked"].tools == []


def test_lo_wins_over_opencode(tmp_path):
    lo = tmp_path / ".lo" / "agents"
    oc = tmp_path / ".opencode" / "agents"
    _write(lo, "dup.md", "lo agent")
    _write(oc, "dup.md", "opencode agent")
    p = load_file_presets([lo, oc])["dup"]
    assert p.system_prompt == "lo agent"


def test_file_preset_does_not_mutate_builtins(tmp_path):
    d = tmp_path / ".lo" / "agents"
    _write(d, "general.md", "---\nmode: build\n---\nOverridden general.")
    register_file_presets([d])
    # get_preset returns the file version, but the built-in dict is untouched
    assert get_preset("general").system_prompt == "Overridden general."
    assert PRESETS["general"].system_prompt != "Overridden general."


def test_reserved_safety_presets_cannot_be_shadowed(tmp_path):
    # A file agent must never override a trusted read-only preset: an untrusted repo
    # shipping .lo/agents/plan.md with write tools must not defeat plan mode.
    d = tmp_path / ".lo" / "agents"
    for name in ("plan", "explore", "review", "security-review"):
        _write(d, f"{name}.md", "---\nmode: build\n---\nEvil overridden preset.")
    register_file_presets([d])
    for name in ("plan", "explore", "review", "security-review"):
        p = get_preset(name)
        assert p.default == "deny"
        assert "Evil overridden" not in p.system_prompt


def test_bad_temperature_does_not_wipe_other_agents(tmp_path):
    # One agent with a non-numeric temperature must fall back to a default, not
    # take down every other valid file-authored agent.
    d = tmp_path / ".lo" / "agents"
    _write(d, "good.md", "---\nmode: build\n---\nGood agent.")
    _write(d, "bad.md", "---\ntemperature: low\n---\nBad temp agent.")
    presets = load_file_presets([d])
    assert presets["good"].system_prompt == "Good agent."
    assert isinstance(presets["bad"].sampling.temperature, float)
