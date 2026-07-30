"""
COSMIC: COsine Similarity Metric for Identifying Concepts.

Luu, Wei, Yang, Wang (ACL 2025 Findings):
"COSMIC: Cosine Similarity Metric for Identifying Concepts."
https://arxiv.org/abs/2506.00085

Where Arditi (2406.11717) extracts ONE direction at a user-picked layer and
ablates it there, COSMIC asks: which (layer, direction) pair best separates
the contrastive classes — judged by cosine similarity of the post-intervention
activations themselves, with no reference to model output text?

That self-contained "output-free" character is COSMIC's selling point on
weakly-aligned / small models, where output-string refusal classifiers
(Arditi's selection rule) misfire because the model refuses ambiguously
or not at all.

Objective (paper, last-token activations at the candidate layer L):

    S_refuse(d, L) = cos( ā_+(d, L),  b̄(L) )
    S_comply(d, L) = cos( ā(L),       b̄_-(d, L) )
    score(d, L)    = S_refuse + S_comply

    where
      b̄(L)         = mean residual at L on the harmful set,        no intervention
      ā(L)         = mean residual at L on the harmless set,       no intervention
      ā_+(d, L)    = mean residual at L on the harmless set, with d ADDED at L
                     (refusal induction; intuition: harmless inputs should look
                      more like harmful baselines once d is injected)
      b̄_-(d, L)    = mean residual at L on the harmful set, with d ABLATED at L
                     (refusal ablation; intuition: harmful inputs should look
                      more like harmless baselines once d is projected out)

In our scaffold we have one candidate direction per layer (diff-of-means at
that L), so the search effectively reduces to layer selection — which still
captures the paper's main practical claim ("automated layer choice").

Cost: 4 forward-cache passes per prompt per candidate layer, capped by the
7-layer search window mandated by the bench harness.
"""

from typing import Callable, List, Tuple

import torch

from model import get_model
from research import apply_chat_template, make_ablation_hook

from ..technique import Technique


class Cosmic(Technique):
    """
    Layer + direction selection via the COSMIC cosine objective.

    Implements the full paper objective (refusal-induction + refusal-ablation
    cosine sum on mean last-token residuals), with the following scoping
    decisions documented honestly:

      * Candidate directions are diff-of-means at each candidate layer
        (one direction per L). The original paper considers multiple
        candidate directions per layer; we constrain to one per L to fit the
        7-layer search budget. The layer-search advantage over Arditi is
        preserved; the multi-direction-per-layer sweep is not.
      * Refusal-induction strength (α in `h ← h + α·d̂`) is set to the
        diff-of-means norm at L. The paper does not explicitly pin α; this
        is a natural-scale default consistent with the magnitude of the
        contrastive signal at that layer.
      * Means are taken over the contrastive prompt sets passed to fit().
        The paper holds out a validation split; we deliberately match
        Arditi's footprint (no extra split) so the bench compares
        techniques on identical fit-time data.
    """

    name = "COSMIC (cosine layer + direction selection)"
    paper_url = "https://arxiv.org/abs/2506.00085"

    def __init__(self) -> None:
        super().__init__()
        self._unit_direction: torch.Tensor | None = None
        self._direction_norm: float = 0.0
        self._selected_score: float = 0.0
        self._search_window: Tuple[int, int] = (0, 0)
        self._per_layer_scores: List[Tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_token_resid(model, prompt: str, layer: int) -> torch.Tensor:
        """Run the prompt, return the last-token residual at `layer` (detached)."""
        formatted = apply_chat_template(prompt)
        _logits, cache = model.run_with_cache(formatted)
        return cache[f"blocks.{layer}.hook_resid_post"][:, -1, :].squeeze(0).detach()

    @staticmethod
    def _last_token_resid_with_hook(
        model, prompt: str, layer: int, hook_fn: Callable
    ) -> torch.Tensor:
        """Same as _last_token_resid but with a fwd hook installed at `layer`."""
        formatted = apply_chat_template(prompt)
        hook_name = f"blocks.{layer}.hook_resid_post"
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            _logits, cache = model.run_with_cache(formatted)
        return cache[hook_name][:, -1, :].squeeze(0).detach()

    @staticmethod
    def _mean_resid(model, prompts: List[str], layer: int) -> torch.Tensor:
        return torch.stack(
            [Cosmic._last_token_resid(model, p, layer) for p in prompts]
        ).mean(dim=0)

    @staticmethod
    def _mean_resid_with_hook(
        model, prompts: List[str], layer: int, hook_fn: Callable
    ) -> torch.Tensor:
        return torch.stack(
            [
                Cosmic._last_token_resid_with_hook(model, p, layer, hook_fn)
                for p in prompts
            ]
        ).mean(dim=0)

    @staticmethod
    def _make_addition_hook(alpha_times_unit: torch.Tensor) -> Callable:
        """Build a hook that ADDS `alpha_times_unit` to every position at the
        installed layer. Mirrors the steering primitive in research.generate_steered
        but kept local so we don't import a generation-shaped helper here."""

        def addition_hook(activation, hook):
            # activation: [batch, seq, d_model]
            activation[:, :, :] = activation + alpha_times_unit
            return activation

        return addition_hook

    @staticmethod
    def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
        # torch.cosine_similarity needs 1D->2D; use a manual computation for clarity.
        denom = (u.norm() * v.norm()).clamp_min(1e-12)
        return float((u @ v) / denom)

    def _diff_of_means_direction(
        self,
        model,
        harmful: List[str],
        harmless: List[str],
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return (unit_direction, raw_direction, norm) at `layer`."""
        harmful_mean = self._mean_resid(model, harmful, layer)
        harmless_mean = self._mean_resid(model, harmless, layer)
        raw = harmful_mean - harmless_mean
        norm = float(raw.norm().item())
        if norm < 1e-8:
            raise RuntimeError(
                f"COSMIC: diff-of-means at L={layer} has near-zero norm; "
                "contrastive sets are too similar to extract a direction."
            )
        return raw / norm, raw, norm

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------

    def fit(
        self,
        model,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
    ) -> None:
        if not harmful_prompts or not harmless_prompts:
            raise ValueError("Cosmic.fit: need at least 1 prompt per class")

        n_layers = model.cfg.n_layers
        lo = max(0, layer - 3)
        hi = min(n_layers, layer + 4)  # +4 → end-exclusive gives 7-layer span
        if lo >= hi:
            raise RuntimeError(
                f"COSMIC: search window collapsed (lo={lo}, hi={hi}); "
                f"n_layers={n_layers}, suggested layer={layer}."
            )
        self._search_window = (lo, hi)

        device = model.cfg.device
        best_L: int | None = None
        best_score = -float("inf")
        best_unit: torch.Tensor | None = None
        best_norm: float = 0.0
        per_layer: List[Tuple[int, float]] = []

        for L in range(lo, hi):
            # 1. Candidate direction at L (diff-of-means)
            unit, _raw, norm = self._diff_of_means_direction(
                model, harmful_prompts, harmless_prompts, L
            )
            unit = unit.to(device)

            # 2. Baselines at L (no intervention)
            harmful_mean = self._mean_resid(model, harmful_prompts, L)
            harmless_mean = self._mean_resid(model, harmless_prompts, L)

            # 3. Refusal-induction: ADD α·d̂ to harmless prompts at L.
            #    α = ‖raw diff-of-means‖ (natural scale of the contrastive signal).
            addition = self._make_addition_hook((norm * unit).to(device))
            harmless_induced_mean = self._mean_resid_with_hook(
                model, harmless_prompts, L, addition
            )

            # 4. Refusal-ablation: project d̂ OUT of harmful prompts at L.
            ablation = make_ablation_hook(unit)
            harmful_ablated_mean = self._mean_resid_with_hook(
                model, harmful_prompts, L, ablation
            )

            # 5. COSMIC objective
            s_refuse = self._cos(harmless_induced_mean, harmful_mean)
            s_comply = self._cos(harmless_mean, harmful_ablated_mean)
            score = s_refuse + s_comply

            per_layer.append((L, score))
            if score > best_score:
                best_score = score
                best_L = L
                best_unit = unit
                best_norm = norm

        if best_L is None or best_unit is None:
            raise RuntimeError("COSMIC: no candidate layer scored; this is a bug.")

        self._layer = best_L
        self._unit_direction = best_unit
        self._direction_norm = best_norm
        self._selected_score = best_score
        self._per_layer_scores = per_layer
        self._fitted = True

    def make_ablation_hook(self) -> Tuple[str, Callable]:
        if not self._fitted or self._unit_direction is None or self._layer is None:
            raise RuntimeError("Cosmic.make_ablation_hook called before fit()")
        return (
            f"blocks.{self._layer}.hook_resid_post",
            make_ablation_hook(self._unit_direction),
        )
