"""The guidance/Outlines-style builder over the harness IR: composition, bounded
repeats, type leaves, and that everything still emits to GBNF + Lark + validates.
"""

import pytest

from local_harness.structured import (
    G, lit, regex, select, choice, seq, one_or_more, zero_or_more, optional,
    exactly, at_least, at_most, between, json_schema, INT, FLOAT, BOOL, WORD,
)
from local_harness.skills.skill import Skill


def test_select_and_operators():
    g = lit("VERDICT: ") + select(["guilty", "not guilty"])
    assert g.validate("VERDICT: guilty")
    assert g.validate("VERDICT: not guilty")
    assert not g.validate("VERDICT: maybe")

    either = lit("yes") | lit("no")
    assert either.validate("yes") and either.validate("no") and not either.validate("maybe")

    # bare strings coerce to literals on either side of + and |
    assert (("a: " + WORD)).validate("a: hello")


def test_quantifiers_and_bounded_repeats():
    assert one_or_more(regex("[0-9]")).validate("12345")
    assert not one_or_more(regex("[0-9]")).validate("")
    assert zero_or_more(lit("ab")).validate("")
    assert optional(lit("x")).validate("") and optional(lit("x")).validate("x")

    three = exactly(3, regex("[a-z]"))
    assert three.validate("abc") and not three.validate("ab") and not three.validate("abcd")

    assert at_least(2, regex("[a-z]")).validate("abcd")
    assert not at_least(2, regex("[a-z]")).validate("a")
    assert at_most(2, regex("[a-z]")).validate("") and at_most(2, regex("[a-z]")).validate("ab")
    assert not at_most(2, regex("[a-z]")).validate("abc")
    assert between(1, 3, regex("[a-z]")).validate("ab")
    assert not between(1, 3, regex("[a-z]")).validate("abcd")


def test_type_leaves():
    assert INT.validate("42") and INT.validate("-7") and not INT.validate("4.5")
    assert FLOAT.validate("4.5") and FLOAT.validate("-0.1") and not FLOAT.validate("4")
    assert BOOL.validate("true") and BOOL.validate("false") and not BOOL.validate("yes")
    assert WORD.validate("hello") and not WORD.validate("hi there")


def test_emits_to_gbnf_and_lark():
    # a structured-summary grammar like the demo's grammar intervention
    g = (lit("PARTIES: ") + one_or_more(regex("[^\n]")) + lit("\nOUTCOME: ")
         + select(["settled", "ongoing", "dismissed"]))
    gbnf = g.to_gbnf()
    assert "root ::=" in gbnf
    lark = g.to_lark()
    assert "start:" in lark
    # bounded repeats and type leaves stay GBNF-emittable (no {m,n} / no `-?`)
    assert "{" not in exactly(3, WORD).to_gbnf()
    assert INT.to_gbnf()  # does not raise


def test_skill_creation():
    skill = select(["yes", "no"]).skill("yn", description="a yes/no answer")
    assert isinstance(skill, Skill)
    assert skill.grammar is not None and skill.validate_output("yes")
    assert not skill.validate_output("maybe")


def test_json_schema_builder():
    schema = {"type": "object", "required": ["verdict"],
              "properties": {"verdict": {"type": "string"}}}
    skill = json_schema(schema, name="case")
    assert skill.json_schema == schema
    assert skill.validate_output('{"verdict": "guilty"}')
    assert not skill.validate_output('{"other": 1}')


def test_bad_compose_type():
    with pytest.raises(TypeError):
        lit("a") + 5  # not a G or str
