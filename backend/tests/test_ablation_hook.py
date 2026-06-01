"""make_ablation_hook removes exactly the projection along the direction."""

import torch

from research import make_ablation_hook


def test_ablation_hook_zeroes_projection_along_direction():
    d = 8
    direction = torch.zeros(d)
    direction[0] = 1.0  # unit e_0
    hook_fn = make_ablation_hook(direction)

    act = torch.randn(2, 3, d)
    out = hook_fn(act.clone(), None)

    # Component along e_0 is removed everywhere ...
    assert torch.allclose(out[..., 0], torch.zeros(2, 3), atol=1e-5)
    # ... and the orthogonal components are untouched.
    assert torch.allclose(out[..., 1:], act[..., 1:], atol=1e-5)


def test_ablation_hook_idempotent():
    d = 6
    direction = torch.randn(d)
    direction = direction / direction.norm()
    hook_fn = make_ablation_hook(direction)

    act = torch.randn(1, 4, d)
    once = hook_fn(act.clone(), None)
    twice = hook_fn(once.clone(), None)
    # Removing an already-removed direction changes nothing further.
    assert torch.allclose(once, twice, atol=1e-5)
