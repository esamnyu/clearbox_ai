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

from typing import Dict, List, Optional

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

    X = torch.cat([harmful_residuals, harmless_residuals], dim=0).cpu().numpy()
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

    p_harm = probe.predict_proba(residuals.cpu().numpy())[:, 1]

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
