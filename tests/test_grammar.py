import pytest

from local_harness.skills.ir import Grammar, GrammarError, parse_production

SQL_RULES = {
    "select_stmt": '"SELECT " select_list " FROM " ident where_clause? ";"',
    "select_list": '"*" | ident ("," " "? ident)*',
    "where_clause": '" WHERE " ident "=" /[0-9]+/',
    "ident": "/[a-zA-Z_][a-zA-Z0-9_]*/",
}


def grammar():
    return Grammar.from_rules(SQL_RULES, root="select_stmt")


def test_validate_accepts_valid_sql():
    g = grammar()
    assert g.validate("SELECT * FROM users;")
    assert g.validate("SELECT id, name FROM users;")
    assert g.validate("SELECT id FROM users WHERE age=30;")


def test_validate_rejects_invalid():
    g = grammar()
    assert not g.validate("DELETE FROM users;")          # not derivable: spec-driven safety
    assert not g.validate("SELECT id FROM users")        # missing semicolon
    assert not g.validate("SELECT FROM users;")          # missing select list
    assert not g.validate("SELECT * FROM users; DROP TABLE users;")


def test_gbnf_emission():
    gbnf = grammar().to_gbnf()
    assert gbnf.startswith("root ::=")
    assert '"SELECT "' in gbnf
    assert "select-list ::=" in gbnf       # GBNF names use dashes
    assert "[a-zA-Z_]" in gbnf


def test_lark_emission():
    lark = grammar().to_lark()
    assert lark.startswith("start:")
    assert "select_list:" in lark          # Lark names keep underscores
    assert "/[a-zA-Z_][a-zA-Z0-9_]*/" in lark


def test_composition_merge():
    core = Grammar.from_rules({"ident": "/[a-z]+/"}, root="ident")
    main = Grammar.from_rules(
        {"stmt": '"GET " ident', "ident": "/[a-z]+/"}, root="stmt"
    )
    merged = main.merge(core)
    assert merged.validate("GET users")

    conflicting = Grammar.from_rules({"ident": "/[0-9]+/"}, root="ident")
    with pytest.raises(GrammarError, match="conflicting"):
        main.merge(conflicting)


def test_undefined_reference_caught():
    with pytest.raises(GrammarError, match="undefined"):
        Grammar.from_rules({"a": '"x" missing_rule'}, root="a")


def test_production_parser_errors():
    with pytest.raises(GrammarError):
        parse_production('"unclosed')
    with pytest.raises(GrammarError):
        parse_production('("a" | "b"')


def test_complex_regex_rejected_for_gbnf():
    g = Grammar.from_rules({"a": '/(foo|bar)+baz/'}, root="a")
    with pytest.raises(GrammarError, match="too complex"):
        g.to_gbnf()


def test_repetition_semantics():
    g = Grammar.from_rules({"a": '"x"+ "y"?'}, root="a")
    assert g.validate("x")
    assert g.validate("xxxy")
    assert not g.validate("y")
    assert not g.validate("xyy")
