"""
Uncertainty estimators for the Refusal Bench (WS6 — stats hardening).

These back the confidence intervals and the "post-ablation AUC beats chance"
p-value the leaderboard reports. The point of the bench is a small-n claim
(0.96 → 0.76 AUC on ~10–26 eval points), so the estimators that quantify "is
that distinguishable from noise?" need their own tests.

Also covers the runner wiring: that run_bench actually threads the CI /
p-value / n fields onto TechniqueResult rows (success and error paths), and
that serialize() output is strictly-valid JSON (no NaN tokens).
"""

import json
import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest
import torch

import refusal_bench.runner as runner
from refusal_bench.harmfulness_probe import auc_permutation_p, bootstrap_auc_ci
from refusal_bench.scoring import wilson_ci


# ---------------------------------------------------------------------------
# bootstrap_auc_ci
# ---------------------------------------------------------------------------

def _separable(n_per_class: int = 10):
    """Labels + scores that rank perfectly (AUC = 1.0)."""
    labels = [0] * n_per_class + [1] * n_per_class
    scores = list(range(2 * n_per_class))  # strictly increasing == label order
    return labels, scores


def test_bootstrap_ci_zero_width_boundary_degeneracy_on_separable():
    """
    KNOWN BOUNDARY DEGENERACY (documented in bootstrap_auc_ci's docstring,
    deliberately not "fixed"): at observed AUC == 1.0 every both-class
    resample is also perfectly ranked, so the percentile interval collapses
    to zero width — claiming zero uncertainty at small n. This test pins the
    behavior so a silent estimator change is caught; the frontend suppresses
    zero-width CIs rather than displaying them as confidence.
    """
    labels, scores = _separable(12)
    lo, hi = bootstrap_auc_ci(labels, scores, seed=42)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_brackets_half_on_noise():
    # Balanced labels, scores independent of class -> AUC ~ 0.5, wide band.
    rng = np.random.default_rng(0)
    labels = [0, 1] * 30
    scores = rng.normal(size=len(labels)).tolist()
    lo, hi = bootstrap_auc_ci(labels, scores, seed=42)
    assert lo < 0.5 < hi
    assert hi - lo > 0.15  # genuinely uncertain at this n


def test_bootstrap_ci_deterministic_for_seed():
    labels, scores = _separable(8)
    assert bootstrap_auc_ci(labels, scores, seed=7) == bootstrap_auc_ci(
        labels, scores, seed=7
    )


def test_bootstrap_ci_single_class_is_nan():
    lo, hi = bootstrap_auc_ci([1, 1, 1, 1], [0.2, 0.4, 0.6, 0.8])
    assert math.isnan(lo) and math.isnan(hi)


def test_bootstrap_ci_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_auc_ci([0, 1, 0], [0.1, 0.9])


# ---------------------------------------------------------------------------
# auc_permutation_p
# ---------------------------------------------------------------------------

def test_permutation_p_small_on_signal():
    labels, scores = _separable(12)
    # A perfectly-ranked probe should be all-but-impossible under label shuffles.
    assert auc_permutation_p(labels, scores, seed=42) < 0.05


def test_permutation_p_large_on_noise():
    # Equal scores -> every permutation yields AUC 0.5 == observed -> p == 1.0.
    labels = [0, 1] * 10
    scores = [0.5] * len(labels)
    assert auc_permutation_p(labels, scores, seed=42) == pytest.approx(1.0)


def test_permutation_p_floored_by_add_one():
    labels, scores = _separable(12)
    p = auc_permutation_p(labels, scores, n_perm=500, seed=1)
    # add-one correction: p can never drop below 1/(n_perm+1).
    assert p >= 1.0 / 501


def test_permutation_p_single_class_is_nan():
    assert math.isnan(auc_permutation_p([0, 0, 0], [0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def test_wilson_ci_known_interval():
    lo, hi = wilson_ci(3, 5)  # p = 0.6, 95%
    assert lo == pytest.approx(0.2306, abs=2e-3)
    assert hi == pytest.approx(0.8823, abs=2e-3)
    assert lo < 0.6 < hi


def test_wilson_ci_stays_in_unit_interval_at_extremes():
    # k = 0: lower bound clamps to 0 but upper bound is informative (unlike the
    # normal approximation, which collapses to a zero-width [0, 0]).
    lo0, hi0 = wilson_ci(0, 5)
    assert lo0 == pytest.approx(0.0, abs=1e-9)
    assert hi0 > 0.3
    # k = n: mirror case.
    lo1, hi1 = wilson_ci(5, 5)
    assert hi1 == pytest.approx(1.0, abs=1e-9)
    assert lo1 < 1.0


def test_wilson_ci_empty_eval_set():
    assert wilson_ci(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# run_bench wiring — the CI / p / n fields must land on TechniqueResult rows
# ---------------------------------------------------------------------------
# Runs the full runner orchestration with the model, generation, and residual
# extraction stubbed out (no real model load), mirroring the conftest stub
# philosophy: exercise the wiring, not the transformer.

_D_MODEL = 8
_N_PROMPTS = 12
_TEST_FRACTION = 0.25
# With 12 prompts at 0.25 the runner holds out max(2, 3) = 3 per class.
_EXPECTED_N_REFUSAL_EVAL = 3
_EXPECTED_N_AUC_EVAL = 6


def _stub_residuals(prompts: List[str]) -> torch.Tensor:
    """Linearly separable residuals keyed off the prompt text."""
    rows: List[torch.Tensor] = []
    for i, prompt in enumerate(prompts):
        base = 1.0 if "harmful" in prompt else -1.0
        row = torch.full((_D_MODEL,), base)
        row[1] = 0.01 * i  # break exact ties so the probe sees variation
        rows.append(row)
    return torch.stack(rows, dim=0)


class _StubTechnique:
    name = "stub"
    paper_url = "https://example.com/stub"

    def __init__(self) -> None:
        self._fitted = False
        self._layer: Optional[int] = None

    def fit(
        self,
        model: object,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
    ) -> None:
        self._layer = layer
        self._fitted = True

    def make_ablation_hook(self) -> Tuple[str, Callable]:
        return ("blocks.0.hook_resid_post", lambda act, hook: act)

    def unit_direction(self) -> None:
        return None


class _ExplodingTechnique(_StubTechnique):
    name = "exploding"

    def fit(
        self,
        model: object,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
    ) -> None:
        raise RuntimeError("synthetic fit failure")


@pytest.fixture()
def stubbed_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner, "get_model", lambda: object())
    monkeypatch.setattr(runner, "get_model_name", lambda: "stub-model")
    monkeypatch.setattr(
        runner,
        "extract_last_token_residuals",
        lambda prompts, layer: _stub_residuals(prompts),
    )
    monkeypatch.setattr(
        runner,
        "_extract_residuals_with_hook",
        lambda prompts, layer, hook_name, hook_fn: _stub_residuals(prompts),
    )

    def fake_generate(
        prompt: str,
        hook_name: Optional[str],
        hook_fn: Optional[Callable],
        max_new_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
    ) -> str:
        # Baseline (no hook) refuses; ablated complies -> a clean delta.
        return "I cannot help with that" if hook_name is None else "Sure, here you go"

    monkeypatch.setattr(runner, "_generate_with_hook", fake_generate)
    monkeypatch.setattr(
        runner,
        "TECHNIQUES",
        {"stub": _StubTechnique, "exploding": _ExplodingTechnique},
    )
    return runner


def _run_stub_bench(technique_names: List[str]):
    harmful = [f"harmful question {i}" for i in range(_N_PROMPTS)]
    harmless = [f"benign question {i}" for i in range(_N_PROMPTS)]
    return runner.run_bench(
        technique_names=technique_names,
        layer=3,
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        test_fraction=_TEST_FRACTION,
        max_new_tokens=4,
        temperature=0.7,
        seed=42,
    )


def _assert_valid_ci(ci: Optional[Tuple[float, float]]) -> None:
    assert ci is not None
    lo, hi = ci
    assert math.isfinite(lo) and math.isfinite(hi)
    assert 0.0 <= lo <= hi <= 1.0


def test_run_bench_success_row_carries_all_stats_fields(stubbed_runner):
    result = _run_stub_bench(["stub"])
    assert len(result.results) == 1
    row = result.results[0]
    assert row.error is None

    _assert_valid_ci(row.refusal_rate_baseline_ci)
    _assert_valid_ci(row.refusal_rate_ablated_ci)
    _assert_valid_ci(row.harmfulness_auc_pre_ci)
    _assert_valid_ci(row.harmfulness_auc_post_ci)

    assert row.harmfulness_auc_post_p is not None
    assert 0.0 < row.harmfulness_auc_post_p <= 1.0

    assert row.n_refusal_eval == _EXPECTED_N_REFUSAL_EVAL
    assert row.n_auc_eval == _EXPECTED_N_AUC_EVAL


def test_run_bench_error_row_keeps_baseline_only_cis(stubbed_runner):
    result = _run_stub_bench(["exploding"])
    row = result.results[0]
    assert row.error is not None and "synthetic fit failure" in row.error

    # Baseline stats are technique-independent, so they survive the failure...
    _assert_valid_ci(row.refusal_rate_baseline_ci)
    _assert_valid_ci(row.harmfulness_auc_pre_ci)
    assert row.n_refusal_eval == _EXPECTED_N_REFUSAL_EVAL
    assert row.n_auc_eval == _EXPECTED_N_AUC_EVAL

    # ...while everything post-ablation is absent (None or NaN pre-serialize).
    assert row.refusal_rate_ablated_ci is None
    assert row.harmfulness_auc_post_ci is None
    assert row.harmfulness_auc_post_p is None
    assert math.isnan(row.refusal_rate_ablated)
    assert math.isnan(row.harmfulness_auc_post)
    assert math.isnan(row.delta_refusal_rate)
    assert math.isnan(row.delta_auc)


def test_serialize_is_nan_free_even_with_error_rows(stubbed_runner):
    result = _run_stub_bench(["stub", "exploding", "no_such_technique"])
    payload = runner.serialize(result)

    # Starlette's JSONResponse uses json.dumps(allow_nan=False); this
    # round-trip is the exact failure mode the sanitizer guards against.
    parsed = json.loads(json.dumps(payload, allow_nan=False))

    error_rows = [r for r in parsed["results"] if r["error"]]
    assert len(error_rows) == 2
    for row in error_rows:
        assert row["refusal_rate_ablated"] is None
        assert row["harmfulness_auc_post"] is None
        assert row["delta_refusal_rate"] is None
        assert row["delta_auc"] is None


def test_json_safe_maps_non_finite_to_none():
    obj = {
        "a": float("nan"),
        "b": [float("inf"), 1.0],
        "c": (float("-inf"), 0.5),
    }
    assert runner.json_safe(obj) == {"a": None, "b": [None, 1.0], "c": [None, 0.5]}


# -----------------------------------------------------------------------------
# Discriminability: |AUC - 0.5|
#
# AUC's no-information point is 0.5, not 0. A probe at AUC 0.05 discriminates
# almost perfectly and simply reads backwards — flip its sign and it is a 0.95
# probe. Scoring "did the harmfulness signal survive ablation?" on raw AUC
# therefore reports a strongly-inverted probe as though it were signal-free.
#
# This is live in the n=50 run: Arditi post-AUC 0.35, Wollschlager 0.32, both
# below chance, both returning a LARGE one-sided p (~0.88, ~0.93).
# -----------------------------------------------------------------------------

from refusal_bench.harmfulness_probe import (  # noqa: E402
    bootstrap_discriminability_ci,
    discriminability,
    discriminability_permutation_p,
)


def _perfect(n=12):
    """Scores that rank the labels perfectly (AUC 1.0)."""
    labels = [1] * n + [0] * n
    scores = [0.9] * n + [0.1] * n
    return labels, scores


def _inverted(n=12):
    """The same separation, read backwards (AUC 0.0)."""
    labels = [1] * n + [0] * n
    scores = [0.1] * n + [0.9] * n
    return labels, scores


def test_discriminability_is_symmetric_about_chance():
    # The core property: equally-informative probes score equally, regardless
    # of which way they point.
    assert discriminability(0.95) == pytest.approx(0.45)
    assert discriminability(0.05) == pytest.approx(0.45)
    assert discriminability(1.0) == pytest.approx(0.5)
    assert discriminability(0.0) == pytest.approx(0.5)


def test_discriminability_is_zero_at_chance():
    assert discriminability(0.5) == pytest.approx(0.0)


def test_discriminability_handles_missing_and_non_finite():
    assert discriminability(None) is None
    assert discriminability(float("nan")) is None
    assert discriminability(float("inf")) is None


def test_two_sided_p_is_significant_for_an_INVERTED_probe():
    """
    The whole reason this statistic exists. A perfectly inverted probe is
    highly significant two-sided, while the one-sided test — which only ever
    rewards AUC > 0.5 — reports it as maximally UNsurprising.
    """
    labels, scores = _inverted()
    two_sided = discriminability_permutation_p(labels, scores, n_perm=500)
    one_sided = auc_permutation_p(labels, scores, n_perm=500)

    assert two_sided < 0.05, "inverted probe must register as discriminative"
    assert one_sided > 0.9, "one-sided test is blind to inversion (by design)"


def test_two_sided_p_matches_one_sided_significance_for_a_forward_probe():
    labels, scores = _perfect()
    assert discriminability_permutation_p(labels, scores, n_perm=500) < 0.05
    assert auc_permutation_p(labels, scores, n_perm=500) < 0.05


def test_two_sided_p_is_large_on_noise():
    rng = np.random.default_rng(7)
    labels = [1] * 15 + [0] * 15
    scores = list(rng.normal(size=30))
    assert discriminability_permutation_p(labels, scores, n_perm=500) > 0.1


def test_two_sided_p_is_floored_by_add_one():
    labels, scores = _perfect()
    p = discriminability_permutation_p(labels, scores, n_perm=100)
    assert p >= 1.0 / 101


def test_two_sided_p_single_class_is_nan():
    assert math.isnan(discriminability_permutation_p([1, 1, 1], [0.1, 0.2, 0.3]))


def test_discriminability_ci_is_identical_for_mirrored_probes():
    """Folding about chance must make the interval direction-blind."""
    fwd = bootstrap_discriminability_ci(*_perfect(), n_boot=300)
    inv = bootstrap_discriminability_ci(*_inverted(), n_boot=300)
    assert fwd == pytest.approx(inv)


def test_discriminability_ci_collapses_at_the_boundary():
    # AUC pinned at 1.0 → every resample reproduces it → zero-width at 0.5.
    # Documented degeneracy; consumers must not read it as certainty.
    lo, hi = bootstrap_discriminability_ci(*_perfect(), n_boot=300)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_discriminability_ci_lower_bound_reaches_zero_on_noise():
    # Near chance the folded distribution piles up against 0. A lower bound of
    # 0.0 is the correct "consistent with no discrimination" reading, NOT the
    # boundary degeneracy above.
    rng = np.random.default_rng(11)
    labels = [1] * 15 + [0] * 15
    scores = list(rng.normal(size=30))
    lo, hi = bootstrap_discriminability_ci(labels, scores, n_boot=300)
    assert lo == pytest.approx(0.0, abs=0.05)
    assert hi > lo


def test_discriminability_ci_is_deterministic_for_seed():
    a = bootstrap_discriminability_ci(*_perfect(), n_boot=200, seed=3)
    b = bootstrap_discriminability_ci(*_perfect(), n_boot=200, seed=3)
    assert a == b


def test_discriminability_ci_single_class_is_nan():
    lo, hi = bootstrap_discriminability_ci([1, 1, 1], [0.1, 0.2, 0.3])
    assert math.isnan(lo) and math.isnan(hi)


def test_discriminability_ci_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_discriminability_ci([1, 0, 1], [0.1, 0.2])
