"""
Refusal-ablation technique base class.

Every technique in the bench implements this minimal interface:
    1. fit(model, harmful_prompts, harmless_prompts, layer)
       — derive whatever the technique needs from contrastive data.
    2. make_ablation_hook() -> (hook_name, hook_fn)
       — return a TransformerLens hook that the bench can install during
         generation or residual extraction.

The bench harness (refusal_bench/runner.py) loops over techniques, calls
fit + make_ablation_hook for each, and scores each with two metrics:
    - refusal rate (keyword + classifier based)
    - harmfulness-probe AUC after ablation (the Zhao 2507.11878 signal)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple


class Technique:
    """
    Base class. Subclass and override `fit` and `make_ablation_hook`.

    Conventions every subclass follows:
        - `name`: short human-readable identifier (e.g. "Arditi single").
        - `paper_url`: arxiv URL for the paper this technique replicates.
        - `_layer`: the layer at which the ablation will be applied. Set in fit().
        - `_fitted`: bool flag. make_ablation_hook() must raise if False.
    """

    name: str = "unnamed"
    paper_url: str = ""

    def __init__(self) -> None:
        self._fitted: bool = False
        self._layer: Optional[int] = None

    # --- Contract ---------------------------------------------------------

    def fit(
        self,
        model,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
    ) -> None:
        """
        Derive the technique's ablation spec from contrastive data.

        After fit() the technique should be ready to produce an ablation
        hook via make_ablation_hook(). The exact data the technique stores
        depends on the method (a unit direction for Arditi/COSMIC/Cheng,
        a stack of directions for Wollschlager, a neuron set for Herring,
        etc.).
        """
        raise NotImplementedError

    def make_ablation_hook(self) -> Tuple[str, Callable]:
        """
        Return (hook_name, hook_fn) for TransformerLens to install.

        Conventional hook names:
            - residual-stream techniques: f"blocks.{layer}.hook_resid_post"
            - neuron-level techniques (Herring CNA): f"blocks.{layer}.mlp.hook_post"
        """
        raise NotImplementedError

    # --- Diagnostics ------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        layer = f"@L{self._layer}" if self._layer is not None else ""
        return f"<{self.__class__.__name__} {self.name!r} {status}{layer}>"
