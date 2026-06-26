"""Slop-structure detector: catches the overused templates, leaves clean prose
alone (the false-positive guard matters more than coverage here)."""

import pytest

from local_harness.logits.slop import SlopDetector

detector = SlopDetector()

SLOP = [
    ("It's not a tool — it's a platform.", "not_x_its_y"),
    ("Not just faster, but smarter.", "not_just_but"),
    ("Not only fast, but also reliable.", "not_only_but"),
    ("No fluff, no filler, just results.", "no_x_no_y_just"),
    ("Less talk, more action.", "less_more"),
    ("Here's the thing: it just works.", "heres_the"),
    ("It's worth noting that latency matters.", "worth_noting"),
    ("In a world where everything is faster, we slow down.", "in_a_world"),
    ("It scales cleanly — ensuring zero downtime.", "em_dash_participle"),
    ("This makes it easier than ever to ship.", "easier_than_ever"),
    ("A place where design meets data.", "x_meets_y"),
    ("It's more than just a chatbot.", "more_than_just"),
    ("Say goodbye to manual entry.", "say_goodbye"),
    ("Whether you're a novice or a pro, start here.", "whether_youre"),
    ("Honestly, it's a real game-changer.", "game_changer"),
]

CLEAN = [
    "The river flows east through three valleys before reaching the sea.",
    "Latency was 12 ms and throughput held at 4000 requests per second.",
    "Install the dependency, then run the test suite to confirm.",
    # 'it's not ... but' WITHOUT the doubled 'it's' must not trip the antithesis rule
    "She said it's not ready, but we can ship the beta tomorrow.",
    "We support models from several vendors and run them locally.",
    "The function returns None when the file does not exist.",
]


@pytest.mark.parametrize("text,expected", SLOP)
def test_flags_slop_structures(text, expected):
    names = [m.name for m in detector.scan(text)]
    assert expected in names, f"{text!r} should flag {expected}, got {names}"


@pytest.mark.parametrize("text", CLEAN)
def test_leaves_clean_prose_alone(text):
    assert detector.scan(text) == [], f"false positive on clean prose: {text!r}"


def test_first_match_is_earliest():
    text = "Here's the thing: it's more than just a tool."
    first = detector.first_match(text)
    assert first is not None and first.name == "heres_the"
    assert first.start < detector.scan(text)[-1].start
