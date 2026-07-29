"""Harmfulness probe: cross-validated AUC fields (WS3) + dtype handling."""

import pytest
import torch

from refusal_bench.harmfulness_probe import evaluate_probe, train_probe


def _separable(n: int, d: int, sep: float = 8.0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    harmful = torch.randn(n, d, generator=g) + sep
    harmless = torch.randn(n, d, generator=g) - sep
    return harmful, harmless


def test_train_probe_returns_cv_fields():
    harmful, harmless = _separable(20, 16)
    out = train_probe(harmful, harmless)
    assert "cv_auc_mean" in out and "cv_auc_std" in out
    assert out["cv_auc_mean"] is not None


def test_train_probe_separable_high_cv_auc():
    harmful, harmless = _separable(20, 16, sep=8.0)
    out = train_probe(harmful, harmless)
    # Well-separated classes -> cross-validated AUC should be near 1.
    assert out["cv_auc_mean"] > 0.9


def test_train_probe_dmodel_mismatch_raises():
    with pytest.raises(ValueError):
        train_probe(torch.randn(4, 8), torch.randn(4, 16))


# -----------------------------------------------------------------------------
# Reduced-precision residuals
#
# run_bench_local.py casts the model to bfloat16 on CPU and float16 on MPS to
# fit Llama-3.2-1B in memory, so residuals reach the probe in those dtypes.
# torch cannot convert bfloat16 to numpy at all — `.numpy()` raises
# "TypeError: Got unsupported ScalarType BFloat16" — which made every CPU bench
# run die at probe training while MPS runs (float16 converts fine) passed. The
# probe now upcasts with .float() before handing off to sklearn.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_train_probe_accepts_reduced_precision_residuals(dtype):
    harmful, harmless = _separable(20, 16)
    out = train_probe(harmful.to(dtype), harmless.to(dtype))
    assert out["cv_auc_mean"] > 0.9


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_evaluate_probe_accepts_reduced_precision_residuals(dtype):
    harmful, harmless = _separable(20, 16)
    out = train_probe(harmful, harmless)
    residuals = torch.cat([harmful, harmless], dim=0).to(dtype)
    labels = [1] * harmful.shape[0] + [0] * harmless.shape[0]

    ev = evaluate_probe(out["model"], residuals, labels=labels)

    assert ev["auc"] is not None and ev["auc"] > 0.9
    assert len(ev["p_harm"]) == residuals.shape[0]
