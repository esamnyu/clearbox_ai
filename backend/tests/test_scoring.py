"""Keyword refusal-detection scoring."""

from refusal_bench.scoring import is_refusal, refusal_rate


def test_is_refusal_detects_phrases():
    assert is_refusal("I'm sorry, but I cannot help with that.")
    assert is_refusal("As an AI language model, I won't do that.")


def test_is_refusal_false_on_compliance():
    assert not is_refusal("Sure! Here are the steps: first, gather your tools.")


def test_refusal_rate_fraction():
    comps = ["I cannot do that", "Sure, here you go", "I am unable to help"]
    assert refusal_rate(comps) == 2 / 3


def test_refusal_rate_empty_is_zero():
    assert refusal_rate([]) == 0.0
