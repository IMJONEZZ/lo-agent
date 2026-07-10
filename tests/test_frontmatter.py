from local_harness.agent.frontmatter import split_frontmatter


def test_scalar_and_body():
    meta, body = split_frontmatter(
        "---\ndescription: A command\nmodel: qwen3\n---\nHello $ARGUMENTS\n"
    )
    assert meta == {"description": "A command", "model": "qwen3"}
    assert body == "Hello $ARGUMENTS\n"


def test_inline_list():
    meta, _ = split_frontmatter("---\ntags: [a, b, c]\n---\nbody")
    assert meta["tags"] == ["a", "b", "c"]


def test_block_list():
    meta, _ = split_frontmatter(
        "---\nsteps:\n  - first\n  - second\n---\nbody"
    )
    assert meta["steps"] == ["first", "second"]


def test_bool_int_null_coercion():
    meta, _ = split_frontmatter(
        "---\nsubtask: true\nlimit: 5\ntemp: 0.3\nempty: null\n---\nx"
    )
    assert meta["subtask"] is True
    assert meta["limit"] == 5
    assert meta["temp"] == 0.3
    assert meta["empty"] is None


def test_block_scalar_prompt():
    meta, body = split_frontmatter(
        "---\nprompt: |\n  line one\n  line two\n---\nbody"
    )
    assert meta["prompt"] == "line one\nline two"
    assert body == "body"


def test_quoted_string_keeps_special():
    meta, _ = split_frontmatter("---\ndesc: \"a: colon, and [brackets]\"\n---\nx")
    assert meta["desc"] == "a: colon, and [brackets]"


def test_no_frontmatter_is_all_body():
    meta, body = split_frontmatter("just a body\nwith lines")
    assert meta == {}
    assert body == "just a body\nwith lines"


def test_unterminated_fence_is_body():
    text = "---\ndescription: oops no close\nmore text"
    meta, body = split_frontmatter(text)
    assert meta == {}
    assert body == text


def test_empty_and_none():
    assert split_frontmatter("") == ({}, "")
    assert split_frontmatter(None) == ({}, "")


def test_comment_lines_ignored():
    meta, _ = split_frontmatter("---\n# a comment\nmodel: x\n---\nb")
    assert meta == {"model": "x"}
