"""
Harmfulness probe — Zhao 2507.11878 replication primitive.

Pipeline (called from main.py /harmfulness-probe):

    1. Gather last-token residuals at `layer` for harmful + harmless prompts.
    2. Train a logistic-regression probe (harmful=1, harmless=0) on an
       80/20 stratified split.
    3. Optionally re-gather residuals for the same harmful prompts WITH the
       refusal direction ablated at `layer`, then re-evaluate the probe.
    4. Compare mean P(harmful) pre- vs post-ablation: if it stays high, the
       residual-stream still encodes harmfulness even though the surface
       refusal behavior is gone (the Zhao finding).

The ablation hook is imported from research.make_ablation_hook so the bench
and the UI ablation share a single source of truth for h' = h − (h · d̂) d̂.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from model import get_model
from research import make_ablation_hook


# -----------------------------------------------------------------------------
# Residual extraction
# -----------------------------------------------------------------------------

def extract_last_token_residuals(prompts: List[str], layer: int) -> torch.Tensor:
    """
    Run each prompt through the model and grab the last-token residual at `layer`.

    Returns shape [n_prompts, d_model] on CPU. CPU because the probe is
    trained on numpy and we don't want to hold GPU memory across many calls.
    """
    if not prompts:
        raise ValueError("prompts list is empty")

    model = get_model()
    hook_name = f"blocks.{layer}.hook_resid_post"

    residuals: List[torch.Tensor] = []
    for prompt in prompts:
        _logits, cache = model.run_with_cache(prompt)
        last = cache[hook_name][:, -1, :].squeeze(0).detach().cpu()  # [d_model]
        residuals.append(last)

    return torch.stack(residuals, dim=0)  # [n, d_model]


# -----------------------------------------------------------------------------
# Probe training and evaluation
# -----------------------------------------------------------------------------

def train_probe(
    harmful_residuals: torch.Tensor,
    harmless_residuals: torch.Tensor,
) -> Dict:
    """
    Fit a logistic-regression probe to discriminate harmful from harmless residuals.

    class_weight="balanced" so unequal class sizes don't bias the probe.
    random_state=42 for reproducibility. Stratified 80/20 split keeps both
    classes in both folds — required for the AUC to be defined.
    """
    if harmful_residuals.ndim != 2 or harmless_residuals.ndim != 2:
        raise ValueError("residuals must be 2D [n, d_model]")
    if harmful_residuals.shape[1] != harmless_residuals.shape[1]:
        raise ValueError(
            f"d_model mismatch: harmful={harmful_residuals.shape[1]} "
            f"harmless={harmless_residuals.shape[1]}"
        )

    # .float() before .numpy(): torch cannot convert bfloat16 to numpy at all
    # ("Got unsupported ScalarType BFloat16"), which broke every CPU bench run
    # — run_bench_local.py casts the model to bfloat16 on CPU (float16 on MPS,
    # which is why MPS runs never hit this). sklearn needs float32/64 anyway,
    # so upcasting here is the correct conversion, not a workaround.
    X = (
        torch.cat([harmful_residuals, harmless_residuals], dim=0)
        .float()
        .cpu()
        .numpy()
    )
    y = np.concatenate(
        [
            np.ones(harmful_residuals.shape[0], dtype=np.int64),
            np.zeros(harmless_residuals.shape[0], dtype=np.int64),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Stronger regularization than the original C=1.0. With n_samples << d_model
    # the probe can linearly separate ANY labeling, so the single-split
    # train_auc saturates at 1.0 and is uninformative; we lean on regularization
    # and treat cross-validated AUC (below) as the number to trust.
    C = 0.5
    probe = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    probe.fit(X_train, y_train)

    train_scores = probe.predict_proba(X_train)[:, 1]
    test_scores = probe.predict_proba(X_test)[:, 1]

    # Cross-validated AUC on the full set — the honest metric when one split's
    # train_auc is degenerate. Stratified so both classes appear in each fold;
    # guard datasets too small for CV (leaves the fields None).
    cv_auc_mean = None
    cv_auc_std = None
    min_class = int(min((y == 1).sum(), (y == 0).sum()))
    if min_class >= 2:
        n_splits = min(5, min_class)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(
                LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
                X,
                y,
                cv=skf,
                scoring="roc_auc",
            )
            cv_auc_mean = float(cv_scores.mean())
            cv_auc_std = float(cv_scores.std())
        except ValueError:
            pass  # degenerate (single-class) fold — leave as None

    return {
        "model": probe,
        "train_auc": float(roc_auc_score(y_train, train_scores)),
        "test_auc": float(roc_auc_score(y_test, test_scores)),
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


def evaluate_probe(
    probe: LogisticRegression,
    residuals: torch.Tensor,
    labels: Optional[List[int]] = None,
) -> Dict:
    """
    Score residuals with a trained probe. Returns per-example P(harmful),
    the mean, and AUC if ground-truth labels are supplied (and both classes
    are present).
    """
    if residuals.ndim != 2:
        raise ValueError("residuals must be 2D [n, d_model]")

    # .float() for the same bfloat16 reason as in train_probe above.
    p_harm = probe.predict_proba(residuals.float().cpu().numpy())[:, 1]

    auc: Optional[float] = None
    if labels is not None:
        if len(labels) != p_harm.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} != residuals count {p_harm.shape[0]}"
            )
        if len(set(labels)) >= 2:
            auc = float(roc_auc_score(labels, p_harm))

    return {
        "p_harm": [float(x) for x in p_harm.tolist()],
        "mean_p_harm": float(p_harm.mean()),
        "auc": auc,
    }


# -----------------------------------------------------------------------------
# Uncertainty estimation
#
# The eval set is tiny (n ~ 5–15 per class), so a point AUC like 0.76 is a
# single draw from a wide sampling distribution — on ~26 points ROC-AUC only
# takes values on a grid of step ~1/169. Reporting it bare invites the obvious
# reviewer rebuttal ("0.96 → 0.76 is within noise at this n"). These two
# estimators answer that rebuttal directly:
#   * bootstrap_auc_ci  → how wide is the band around the point AUC?
#   * auc_permutation_p → is the post-ablation AUC ABOVE chance? (one-sided)
# Both seed an explicit Generator so a given (labels, scores) → identical CI/p
# across runs, matching the rest of the bench's determinism contract.
#
# ── Why AUC alone is the wrong "did the signal survive?" statistic ───────────
#
# AUC measures ranking agreement, and 0.5 — not 0 — is the no-information
# point. An AUC of 0.05 is not "signal destroyed"; it is a probe that
# discriminates almost perfectly and reads BACKWARDS. Flip its sign and it is a
# 0.95 probe. The harmfulness information is fully present in the residual
# stream either way, which is precisely what the dissociation question asks
# about.
#
# This is not hypothetical here. At n=50, Arditi ablation drives post-AUC to
# 0.35 and Wollschlager to 0.32 — both BELOW chance. The one-sided
# auc_permutation_p correctly returns ~0.88 and ~0.93 for those ("no evidence
# AUC > 0.5"), but read casually that looks like "no signal", when the honest
# reading is "signal, pointing the other way, and this eval is too small to
# resolve which".
#
# So the discriminability functions below score |AUC − 0.5| instead: distance
# from chance, in [0, 0.5], sign-agnostic. 0 means no information; 0.5 means
# perfect separation in one direction or the other. Raw AUC is still reported
# alongside, because the SIGN is what tells you the direction — you need both.
#
# These are additive: auc_permutation_p and bootstrap_auc_ci keep their exact
# original meaning so artifacts written before this change are not silently
# reinterpreted. See docs/bench_partials/README.md for why this repo is strict
# about that.
# -----------------------------------------------------------------------------


def discriminability(auc: Optional[float]) -> Optional[float]:
    """
    |AUC − 0.5| — how far the probe is from uninformative, ignoring direction.

    Range [0, 0.5]. Returns None for a None/NaN AUC so callers can distinguish
    "undefined" from "no discrimination".
    """
    if auc is None:
        return None
    a = float(auc)
    if not np.isfinite(a):
        return None
    return abs(a - 0.5)

def bootstrap_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for ROC-AUC.

    Resamples (label, score) pairs with replacement n_boot times and takes the
    central `ci` mass of the resampled AUCs. Resamples that collapse to a
    single class (AUC undefined) are skipped. Returns (nan, nan) if the input
    is single-class or every resample degenerated.

    Known failure mode — boundary degeneracy: when the observed AUC is exactly
    0.0 or 1.0 (scores perfectly rank the labels), every bootstrap resample
    that retains both classes is also perfectly ranked, so the interval
    collapses to zero width — (1.0, 1.0) or (0.0, 0.0). At small n this reads
    as "zero uncertainty" at exactly the point where sampling uncertainty is
    large — the overclaim this CI exists to prevent. Downstream consumers must
    not treat a zero-width boundary interval as a confident estimate; the
    frontend suppresses zero-width CIs for this reason. We keep the percentile
    bootstrap (rather than switching estimators) so the CI stays comparable
    with the rest of the bench's bootstrap machinery.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"labels ({y.shape[0]}) and scores ({s.shape[0]}) differ")
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = y.shape[0]
    aucs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, s[idx])))

    if not aucs:
        return (float("nan"), float("nan"))
    tail = (1.0 - ci) / 2.0
    lo = float(np.percentile(aucs, 100.0 * tail))
    hi = float(np.percentile(aucs, 100.0 * (1.0 - tail)))
    return (lo, hi)


def auc_permutation_p(
    labels: Sequence[int],
    scores: Sequence[float],
    n_perm: int = 2000,
    seed: int = 42,
) -> float:
    """
    One-sided permutation p-value for H0: AUC = 0.5 vs H1: AUC > 0.5.

    Shuffles the labels n_perm times (breaking any score↔class association) and
    measures how often a permuted AUC reaches the observed AUC. Uses the
    add-one correction p = (1 + #{perm ≥ observed}) / (n_perm + 1), so p is
    never exactly 0 and is bounded below by 1/(n_perm+1). A small p means the
    probe still reads class above chance *after* ablation — i.e. the
    harmfulness representation genuinely survived, not an artifact. Returns nan
    if the input is single-class.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"labels ({y.shape[0]}) and scores ({s.shape[0]}) differ")
    if len(np.unique(y)) < 2:
        return float("nan")

    observed = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        if float(roc_auc_score(rng.permutation(y), s)) >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def bootstrap_discriminability_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for |AUC − 0.5|.

    Same resampling scheme as bootstrap_auc_ci — identical seed and draw order,
    so the two are directly comparable — but folds each resampled AUC about
    chance before taking percentiles.

    Folding makes the statistic non-negative, which has one consequence worth
    knowing: when the true AUC sits near 0.5, the folded distribution piles up
    against 0 and the interval's lower bound goes to 0.0. That is the correct
    reading ("consistent with no discrimination"), not a degenerate interval.

    The genuine degeneracy is at the other end: an observed AUC pinned at
    exactly 0.0 or 1.0 gives discriminability 0.5 in every resample that keeps
    both classes, collapsing the interval to (0.5, 0.5). Callers must treat a
    zero-width interval as an artifact of the estimator at small n, not as
    certainty — the frontend suppresses it for that reason.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"labels ({y.shape[0]}) and scores ({s.shape[0]}) differ")
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = y.shape[0]
    values: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        values.append(abs(float(roc_auc_score(yb, s[idx])) - 0.5))

    if not values:
        return (float("nan"), float("nan"))
    tail = (1.0 - ci) / 2.0
    lo = float(np.percentile(values, 100.0 * tail))
    hi = float(np.percentile(values, 100.0 * (1.0 - tail)))
    return (lo, hi)


def discriminability_permutation_p(
    labels: Sequence[int],
    scores: Sequence[float],
    n_perm: int = 2000,
    seed: int = 42,
) -> float:
    """
    Two-sided permutation p-value for H0: AUC = 0.5 vs H1: AUC ≠ 0.5.

    The test statistic is |AUC − 0.5|, so a probe that reads perfectly
    backwards is as significant as one that reads perfectly forwards — which is
    the whole point. Compare with auc_permutation_p, which only ever rewards
    AUC > 0.5 and therefore reports a *large* p for a strongly inverted probe,
    a result easily misread as "no signal".

    Same add-one correction as the one-sided test:
    p = (1 + #{|AUC_perm − 0.5| ≥ |AUC_obs − 0.5|}) / (n_perm + 1), so p is
    bounded below by 1/(n_perm+1) and never exactly 0. Returns nan if the input
    is single-class.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"labels ({y.shape[0]}) and scores ({s.shape[0]}) differ")
    if len(np.unique(y)) < 2:
        return float("nan")

    observed = abs(float(roc_auc_score(y, s)) - 0.5)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        if abs(float(roc_auc_score(rng.permutation(y), s)) - 0.5) >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


# -----------------------------------------------------------------------------
# Ablation-aware extraction
# -----------------------------------------------------------------------------

def extract_with_ablation(
    prompts: List[str],
    layer_extract: int,
    ablation_direction: List[float],
    ablation_layer: int,
) -> torch.Tensor:
    """
    Gather last-token residuals at `layer_extract` while ablating
    `ablation_direction` at `ablation_layer`. Typical use: extract == ablation,
    but we keep them split so callers can probe DOWNSTREAM of the ablation
    site too (useful for tracing how the harmfulness signal propagates).
    """
    if not prompts:
        raise ValueError("prompts list is empty")

    model = get_model()

    direction_tensor = torch.tensor(
        ablation_direction, dtype=torch.float32, device=model.cfg.device
    )
    norm = direction_tensor.norm()
    if norm.item() < 1e-8:
        raise RuntimeError("ablation_direction has near-zero norm; cannot ablate")
    unit_direction = direction_tensor / norm

    extract_hook_name = f"blocks.{layer_extract}.hook_resid_post"
    ablation_hook_name = f"blocks.{ablation_layer}.hook_resid_post"
    ablation_hook = make_ablation_hook(unit_direction)

    residuals: List[torch.Tensor] = []
    for prompt in prompts:
        with model.hooks(fwd_hooks=[(ablation_hook_name, ablation_hook)]):
            _logits, cache = model.run_with_cache(prompt)
        last = cache[extract_hook_name][:, -1, :].squeeze(0).detach().cpu()
        residuals.append(last)

    return torch.stack(residuals, dim=0)
