"""
Cheng compressed-direction refusal ablation.

Cheng et al. (April 2026):
"What Drives Representation Steering?"
https://arxiv.org/abs/2604.08524

The headline finding: the Arditi-style refusal direction has thousands of
nonzero coefficients (d_model = 2048 for Llama-3.2-1B), but its behavioral
effect is dominated by a small subset of them. Sparsifying d_L to its top
1–10% of coefficients by magnitude and re-normalizing preserves most of
the ablation effect — a representation-sparsity result.

Cheng = "Arditi + a compression step". We reuse Arditi's difference-of-
means extraction at layer L, then keep the top-k coefficients by magnitude
and zero the rest. The compressed vector is re-normalized so the standard
projection-removal hook math (h - (h·d̂)d̂ with ‖d̂‖ = 1) is unchanged.

Compression method choice
-------------------------
The paper hints at two routes (top-k magnitude thresholding and SVD-of-
harmful-residuals projection). We ship the top-k-magnitude variant
because:

    - It's a single line of tensor code, fully reproducible with no
      hyperparameter beyond α and no extra forward passes.
    - It needs no additional residual stack (SVD would require caching
      the harmful-class residuals separately and a second decomposition,
      doubling memory).
    - It's the natural thing to defend in a writeup: "we kept the largest
      coefficients of the Arditi direction." The SVD variant introduces
      an extra basis-choice question that the bench isn't designed to
      arbitrate.

If a future ablation shows the SVD variant changes the harmfulness-probe
AUC story, add it as a second class (e.g. ChengSVD) rather than a flag.
"""

from typing import Callable, List, Tuple

import torch

from model import get_model  # noqa: F401  — imported for parity with sibling techniques
from research import apply_chat_template, make_ablation_hook

from ..technique import Technique


class Cheng(Technique):
    """Sparsified single-direction projection ablation (Arditi + top-k compression)."""

    name = "Cheng (compressed direction)"
    paper_url = "https://arxiv.org/abs/2604.08524"

    def __init__(self, sparsity_fraction: float = 0.05) -> None:
        """
        Parameters
        ----------
        sparsity_fraction
            Fraction α ∈ (0, 1] of d_model coefficients to keep. The default
            α = 0.05 (top 5%) sits in the middle of the paper's "1–10%
            retains most behavior" claim. Override for sweeps.
        """
        super().__init__()
        if not (0.0 < sparsity_fraction <= 1.0):
            raise ValueError(
                f"Cheng: sparsity_fraction must be in (0, 1], got {sparsity_fraction}"
            )

        self._sparsity_fraction: float = sparsity_fraction
        self._unit_direction: torch.Tensor | None = None
        self._direction_norm: float = 0.0  # norm of the dense Arditi direction (pre-compression)
        self._compressed_norm: float = 0.0  # norm of the sparse direction (pre-renormalization)
        self._n_kept: int = 0
        self._n_total: int = 0
        self._kept_indices: torch.Tensor | None = None

    def fit(
        self,
        model,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
    ) -> None:
        """
        Extract the Arditi direction, then compress to its top-k coefficients
        by magnitude and re-normalize.

        The compressed vector lives in the same R^d_model space as Arditi's;
        only its sparsity pattern changes. The ablation hook (projection
        removal along a unit vector) is identical.
        """
        if not harmful_prompts or not harmless_prompts:
            raise ValueError("Cheng.fit: need at least 1 prompt per class")

        # --- Stage 1: Arditi extraction (difference-of-means, last token) ---
        def last_token_resid(prompt: str) -> torch.Tensor:
            formatted = apply_chat_template(prompt)
            _logits, cache = model.run_with_cache(formatted)
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            return resid[:, -1, :].squeeze(0).detach()

        harmful_resid = torch.stack([last_token_resid(p) for p in harmful_prompts])
        harmless_resid = torch.stack([last_token_resid(p) for p in harmless_prompts])

        direction = harmful_resid.mean(dim=0) - harmless_resid.mean(dim=0)
        dense_norm = direction.norm()
        if dense_norm.item() < 1e-8:
            raise RuntimeError("Cheng: extracted (pre-compression) direction has near-zero norm")

        # --- Stage 2: top-k compression by coefficient magnitude ---
        d_model = direction.numel()
        # Always keep at least one coefficient — guards against α * d_model rounding to 0
        # for pathologically small models or sparsity fractions.
        n_keep = max(1, round(self._sparsity_fraction * d_model))
        n_keep = min(n_keep, d_model)

        # topk on absolute values, then build a sparse copy of the original
        # signed coefficients at those indices. Keeping signs matters: the
        # ablation hook removes a signed projection, not a magnitude.
        magnitudes = direction.abs()
        _topk_vals, topk_indices = torch.topk(magnitudes, k=n_keep, largest=True, sorted=False)
        compressed = torch.zeros_like(direction)
        compressed[topk_indices] = direction[topk_indices]

        compressed_norm = compressed.norm()
        if compressed_norm.item() < 1e-8:
            # Should be impossible if dense_norm is finite and n_keep >= 1, but guard anyway.
            raise RuntimeError("Cheng: compressed direction has near-zero norm")

        # --- Stage 3: re-normalize so the ablation hook math stays valid ---
        unit_direction = (compressed / compressed_norm).to(model.cfg.device)

        # --- Store ---
        self._unit_direction = unit_direction
        self._direction_norm = float(dense_norm.item())
        self._compressed_norm = float(compressed_norm.item())
        self._n_kept = int(n_keep)
        self._n_total = int(d_model)
        self._kept_indices = topk_indices.detach().cpu()
        self._layer = layer
        self._fitted = True

    def make_ablation_hook(self) -> Tuple[str, Callable]:
        if not self._fitted or self._unit_direction is None or self._layer is None:
            raise RuntimeError("Cheng.make_ablation_hook called before fit()")
        return (
            f"blocks.{self._layer}.hook_resid_post",
            make_ablation_hook(self._unit_direction),
        )
