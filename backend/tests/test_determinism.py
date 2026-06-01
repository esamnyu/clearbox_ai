"""
WS2 determinism: the before/after pair must differ only by the intervention.

Loads a real model, so it's gated behind NEUROSCOPE_RUN_MODEL_TESTS=1 to keep
the default test run fast and offline. Greedy decoding (do_sample defaults to
False) makes the baseline generation reproducible across calls.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("NEUROSCOPE_RUN_MODEL_TESTS") != "1",
    reason="loads a real model; set NEUROSCOPE_RUN_MODEL_TESTS=1 to run",
)


def test_ablation_baseline_is_deterministic():
    import model

    model.load_model("gpt2-small")
    from research import ablate_along_direction

    d_model = model.get_model().cfg.d_model
    direction = [0.0] * d_model
    direction[0] = 1.0

    r1 = ablate_along_direction(
        "The Eiffel Tower is in", direction, layer=6, max_new_tokens=8
    )
    r2 = ablate_along_direction(
        "The Eiffel Tower is in", direction, layer=6, max_new_tokens=8
    )
    assert r1["baseline_text"] == r2["baseline_text"]


def test_steered_baseline_is_deterministic():
    import model

    model.load_model("gpt2-small")
    from research import generate_steered

    d_model = model.get_model().cfg.d_model
    vec = [0.0] * d_model

    r1 = generate_steered("Hello there", vec, alpha=1.0, layer=6, max_new_tokens=8)
    r2 = generate_steered("Hello there", vec, alpha=1.0, layer=6, max_new_tokens=8)
    assert r1["baseline_text"] == r2["baseline_text"]
