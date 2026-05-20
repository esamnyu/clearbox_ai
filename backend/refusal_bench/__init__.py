"""
Refusal Bench — head-to-head benchmark of refusal-ablation techniques.

Modules:
    harmfulness_probe — Zhao 2507.11878 probe: a linear classifier on residual
        streams that detects harmful intent. Used as the ground-truth signal
        for scoring whether a refusal-ablation technique merely suppresses
        verbal refusal or actually removes the model's internal harmfulness
        representation.
"""

from .harmfulness_probe import (
    evaluate_probe,
    extract_last_token_residuals,
    extract_with_ablation,
    train_probe,
)
from .runner import BenchResult, TechniqueResult, run_bench, serialize
from .scoring import REFUSAL_PHRASES, is_refusal, refusal_rate

__all__ = [
    # Probe primitives
    "evaluate_probe",
    "extract_last_token_residuals",
    "extract_with_ablation",
    "train_probe",
    # Runner
    "BenchResult",
    "TechniqueResult",
    "run_bench",
    "serialize",
    # Scoring
    "REFUSAL_PHRASES",
    "is_refusal",
    "refusal_rate",
]
