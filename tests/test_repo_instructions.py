from local_harness.agent.notebook import Notebook


def _nb(tmp_path, repo=None):
    mem = tmp_path / "mem"
    mem.mkdir(exist_ok=True)
    return Notebook(mem, repo_dir=repo or tmp_path)


def test_agents_md_in_cwd_appears(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use tabs, not spaces.")
    nb = _nb(tmp_path)
    block = nb.system_block()
    assert "Repository instructions (AGENTS.md)" in block
    assert "Use tabs, not spaces." in block


def test_absent_file_no_section(tmp_path):
    nb = _nb(tmp_path)
    assert "Repository instructions" not in nb.system_block()
    assert nb.repo_instructions() == ""


def test_up_tree_walk(tmp_path):
    # Walking up from a subdir picks up the repo-root AGENTS.md, bounded by .git.
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("root rule")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    nb = _nb(tmp_path, repo=sub)
    assert "root rule" in nb.repo_instructions()


def test_no_git_boundary_does_not_escape(tmp_path):
    # With no enclosing git repo, an ancestor AGENTS.md must NOT be pulled in
    # (otherwise we'd walk to the filesystem root, injecting unrelated instructions).
    (tmp_path / "AGENTS.md").write_text("ancestor rule")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    nb = _nb(tmp_path, repo=sub)
    assert "ancestor rule" not in nb.repo_instructions()


def test_claude_md_only_when_referenced(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("claude-specific rules")
    # No reference → CLAUDE.md ignored.
    (tmp_path / "AGENTS.md").write_text("plain agents rules")
    assert "claude-specific" not in _nb(tmp_path).repo_instructions()
    # A bare mention (even a negative one) must NOT pull it in.
    (tmp_path / "AGENTS.md").write_text("ignore any CLAUDE.md in this repo")
    assert "claude-specific" not in _nb(tmp_path).repo_instructions()
    # An explicit @CLAUDE.md import → included.
    (tmp_path / "AGENTS.md").write_text("see @CLAUDE.md for details")
    out = _nb(tmp_path).repo_instructions()
    assert "claude-specific rules" in out


def test_char_cap(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 10000)
    out = _nb(tmp_path).repo_instructions()
    assert len(out) <= Notebook.REPO_INSTRUCTION_CAP + 40
    assert "truncated" in out


def test_git_root_stops_walk(tmp_path):
    # AGENTS.md above the git root must NOT be picked up.
    (tmp_path / "AGENTS.md").write_text("above root")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "AGENTS.md").write_text("repo rule")
    out = _nb(tmp_path, repo=repo).repo_instructions()
    assert "repo rule" in out
    assert "above root" not in out


def test_memory_tool_cannot_target_agents(tmp_path):
    nb = _nb(tmp_path)
    assert nb.edit("add", "agents", text="x").startswith("error")
    assert nb.edit("add", "repo", text="x").startswith("error")
