"""Harmfulness probe: cross-validated AUC fields (WS3)."""

import pytest
import torch

from refusal_bench.harmfulness_probe import train_probe


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
