"""
Arditi single-direction refusal ablation.

Arditi, Obeso, Syed, Conmy, Turner, Belrose, Schwettmann (NeurIPS 2024):
"Refusal in Language Models Is Mediated by a Single Direction."
https://arxiv.org/abs/2406.11717

The claim: refusal in chat models can be isolated as ONE direction in
activation space. Extracted via difference-of-means on contrastive
(harmful, harmless) prompt pairs at the last-token position of layer L.
Ablation projects the residual stream's component along this unit
direction out at layer L:

    h' = h - (h · d̂) d̂

This is the baseline that the other five techniques in the bench compare
against. If a technique can't beat Arditi on Llama-3.2-1B, it doesn't
belong in the bench.
"""

from typing import Callable, List, Tuple

import torch

from model import get_model
from research import apply_chat_template, make_ablation_hook

from ..technique import Technique


class Arditi(Technique):
    """Single-direction projection ablation, difference-of-means extraction."""

    name = "Arditi (single direction)"
    paper_url = "https://arxiv.org/abs/2406.11717"

    def __init__(self) -> None:
        super().__init__()
        self._unit_direction: torch.Tensor | None = None
        self._direction_norm: float = 0.0

    def fit(
        self,
        model,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
    ) -> None:
        """
        Extract the refusal direction via difference-of-means on last-token
        residuals at the given layer.

        Prompts are wrapped in the chat template (no-op for non-instruct
        models) so refusal behaves realistically on Llama-3.2-*-Instruct.
        """
        if not harmful_prompts or not harmless_prompts:
            raise ValueError("Arditi.fit: need at least 1 prompt per class")

        def last_token_resid(prompt: str) -> torch.Tensor:
            formatted = apply_chat_template(prompt)
            _logits, cache = model.run_with_cache(formatted)
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            return resid[:, -1, :].squeeze(0).detach()

        harmful_resid = torch.stack([last_token_resid(p) for p in harmful_prompts])
        harmless_resid = torch.stack([last_token_resid(p) for p in harmless_prompts])

        # difference of means; points from "harmless" to "harmful"
        direction = harmful_resid.mean(dim=0) - harmless_resid.mean(dim=0)
        norm = direction.norm()
        if norm.item() < 1e-8:
            raise RuntimeError("Arditi: extracted direction has near-zero norm")

        self._unit_direction = (direction / norm).to(model.cfg.device)
        self._direction_norm = float(norm.item())
        self._layer = layer
        self._fitted = True

    def make_ablation_hook(self) -> Tuple[str, Callable]:
        if not self._fitted or self._unit_direction is None or self._layer is None:
            raise RuntimeError("Arditi.make_ablation_hook called before fit()")
        return (
            f"blocks.{self._layer}.hook_resid_post",
            make_ablation_hook(self._unit_direction),
        )
