"""probe_direction_cosine: the dissociation confound diagnostic (WS3)."""

import numpy as np
import torch

from refusal_bench.runner import probe_direction_cosine


class _FakeProbe:
    def __init__(self, w):
        self.coef_ = np.asarray(w, dtype=float).reshape(1, -1)


def test_cosine_parallel_is_one():
    probe = _FakeProbe([1.0, 0.0, 0.0])
    d = torch.tensor([2.0, 0.0, 0.0])
    assert probe_direction_cosine(probe, d) == 1.0  # absolute cosine


def test_cosine_orthogonal_is_zero():
    probe = _FakeProbe([1.0, 0.0, 0.0])
    d = torch.tensor([0.0, 1.0, 0.0])
    assert probe_direction_cosine(probe, d) < 1e-6


def test_cosine_none_when_no_direction():
    assert probe_direction_cosine(_FakeProbe([1.0, 0.0]), None) is None


def test_cosine_none_on_shape_mismatch():
    probe = _FakeProbe([1.0, 0.0, 0.0])
    assert probe_direction_cosine(probe, torch.tensor([1.0, 0.0])) is None
