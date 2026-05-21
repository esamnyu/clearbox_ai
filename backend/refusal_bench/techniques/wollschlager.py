"""
Wollschlager polyhedral-cone refusal ablation.

Wollschlager, Hubinger et al. (ICML 2025):
"Concept Cones: Refusal in LLMs Is Multi-Directional."
https://arxiv.org/abs/2502.17420

Arditi 2024 isolated refusal as a single direction via difference-of-means.
Wollschlager showed this is incomplete on many models — particularly the
Llama family — where the refusal subspace is rank > 1: a polyhedral cone
of mechanistically independent directions. Single-direction ablation
suppresses one face of the cone; the rest remain available to recover the
refusal behavior.

Extraction is iterative. At each step k we:
    1. Take diff-of-means on the CURRENT cached residuals (which have
       d_1..d_{k-1} already projected out).
    2. Gram-Schmidt the candidate against the directions extracted so far
       (defensive: in floating point, repeated diff-of-means after
       projection won't lie perfectly in the orthogonal complement).
    3. Renormalize; if the post-orthogonalization norm collapses below a
       tolerance, the cone has lower effective rank than the requested K
       and we stop early.
    4. Project the new direction out of every cached residual before the
       next iteration.

Ablation projects ALL K (or k_effective) directions out simultaneously.
Because the stored directions are orthonormal, the per-direction sequential
projection in the hook is mathematically equivalent to a single batched
projection h - (h D^T) D, just clearer to read.

Default K = 5 mirrors Wollschlager's headline rank for the Llama family;
expose via `__init__(rank=...)` for the bench's rank-k sweep.
"""

from typing import Callable, List, Tuple

import torch

from model import get_model  # noqa: F401 — imported for parity with sibling techniques
from research import apply_chat_template

from ..technique import Technique


class Wollschlager(Technique):
    """Polyhedral-cone projection ablation (iterative diff-of-means with Gram-Schmidt)."""

    name = "Wollschlager (polyhedral cone)"
    paper_url = "https://arxiv.org/abs/2502.17420"

    # Norm below which an orthogonalized candidate direction is treated as
    # collapsed — i.e. the cone has been exhausted before reaching K.
    _ORTHO_TOL: float = 1e-8

    def __init__(self, rank: int = 5) -> None:
        """
        Parameters
        ----------
        rank
            Maximum number of cone directions K to extract. Default 5,
            matching the paper's headline rank for the Llama family. The
            effective rank attained at fit() time is exposed as
            `self._effective_rank` and may be less than K if a direction
            collapses to zero norm after Gram-Schmidt orthogonalization.
        """
        super().__init__()
        if rank < 1:
            raise ValueError(f"Wollschlager: rank must be >= 1, got {rank}")

        self._target_rank: int = int(rank)
        self._effective_rank: int = 0
        self._directions: torch.Tensor | None = None  # [k_effective, d_model], orthonormal

    def fit(
        self,
        model,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
    ) -> None:
        """
        Iteratively extract up to `self._target_rank` orthonormal cone
        directions from the contrastive sets at the given layer.

        We cache the last-token residuals once and then mutate the cache
        in place (per-prompt projection removal). That keeps total work at
        O(n_prompts * 1) forward passes regardless of K, instead of O(K)
        re-runs.
        """
        if not harmful_prompts or not harmless_prompts:
            raise ValueError("Wollschlager.fit: need at least 1 prompt per class")

        device = model.cfg.device

        def last_token_resid(prompt: str) -> torch.Tensor:
            formatted = apply_chat_template(prompt)
            _logits, cache = model.run_with_cache(formatted)
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            return resid[:, -1, :].squeeze(0).detach()

        # Stage 1: cache initial last-token residuals on the model's device.
        # Shape: [n_harmful, d_model] and [n_harmless, d_model].
        harmful_resid = torch.stack([last_token_resid(p) for p in harmful_prompts]).to(device)
        harmless_resid = torch.stack([last_token_resid(p) for p in harmless_prompts]).to(device)

        # Stage 2: iterative extraction with Gram-Schmidt + early stopping.
        directions: List[torch.Tensor] = []

        for _k in range(self._target_rank):
            # (a) raw diff-of-means on the CURRENT cached residuals.
            d_raw = harmful_resid.mean(dim=0) - harmless_resid.mean(dim=0)

            # (b) Gram-Schmidt against previously extracted directions.
            #     Defensive: if the cache update step has been doing its job,
            #     d_raw should already be (numerically) orthogonal to all of
            #     them, but re-orthogonalizing avoids floating-point drift
            #     accumulating across iterations.
            for d_prev in directions:
                d_raw = d_raw - (d_raw @ d_prev) * d_prev

            # (c) renormalize, with early stopping if the direction collapsed.
            norm = d_raw.norm()
            if norm.item() < self._ORTHO_TOL:
                # Cone has lower effective rank than the requested K — stop.
                break

            d_k = (d_raw / norm).to(device)
            directions.append(d_k)

            # (d) project d_k out of every cached residual so the next
            #     diff-of-means is computed in the orthogonal complement.
            #     coeffs: [n, 1]; d_k: [d_model]; broadcasts cleanly.
            h_coeffs = (harmful_resid * d_k).sum(dim=-1, keepdim=True)
            harmful_resid = harmful_resid - h_coeffs * d_k

            l_coeffs = (harmless_resid * d_k).sum(dim=-1, keepdim=True)
            harmless_resid = harmless_resid - l_coeffs * d_k

        if not directions:
            # Pathological: even the first diff-of-means collapsed. Mirror
            # the message from Arditi for consistency.
            raise RuntimeError(
                "Wollschlager: first cone direction has near-zero norm; "
                "contrastive sets are too similar to extract a direction."
            )

        # Stage 3: stack into [k_effective, d_model] on the model's device.
        self._directions = torch.stack(directions, dim=0).to(device)
        self._effective_rank = len(directions)
        self._layer = layer
        self._fitted = True

    def _make_cone_hook(self, directions: torch.Tensor) -> Callable:
        """Closure that projects every stored direction out of the residual stream."""

        # TransformerLens passes the hook point as keyword arg `hook=...`,
        # so the parameter MUST be named `hook` (not `hook_`). Matching the
        # convention in research.make_ablation_hook.
        def cone_ablation_hook(activation, hook):
            # activation: [batch, seq_len, d_model]
            # directions: [k, d_model], orthonormal
            h = activation
            for d in directions:
                coeffs = (h * d).sum(dim=-1, keepdim=True)
                h = h - coeffs * d
            activation[:, :, :] = h
            return activation

        return cone_ablation_hook

    def make_ablation_hook(self) -> Tuple[str, Callable]:
        if not self._fitted or self._directions is None or self._layer is None:
            raise RuntimeError("Wollschlager.make_ablation_hook called before fit()")
        return (
            f"blocks.{self._layer}.hook_resid_post",
            self._make_cone_hook(self._directions),
        )
