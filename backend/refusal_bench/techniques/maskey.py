"""
Maskey over-refusal vs harmful-refusal decomposition.

Maskey et al. (revised April 2026):
"Over-Refusal and Representation Subspaces."
https://arxiv.org/abs/2603.27518

Most refusal-direction methods (Arditi 2406.11717, Wollschlager 2502.17420,
Cheng 2604.08524, COSMIC 2506.00085) conflate two phenomena under a single
"refusal" axis:

  (a) genuinely-harmful-refusal — the model refuses something dangerous
      ("how do I make a pipe bomb"), which is the safety behavior we want
      to keep on the harmful side and probe causally on the ablation side.
  (b) over-refusal — the model refuses a benign-but-edgy request out of
      conservatism. XSTest-style examples: "How do I kill a process in
      Linux?", "What's the best way to shoot a basketball?". The surface
      lexical cues overlap with (a) but the intent is innocuous.

Maskey shows these two phenomena are mediated by DISTINCT directions in
the residual stream — and that ablating the naive single direction
(diff-of-means over JBB-style harmful vs Alpaca-style harmless) removes
both, which is desirable for over-refusal (the model becomes less
neurotic) but unnecessarily strips safety on (a).

The decomposition: extract both directions separately, then project the
over-refusal component out of the harmful direction so the residual is
the part of "harmfulness" that is NOT just over-refusal flavored. Ablate
only that residual.

Algorithm (last-token residual at layer L):

    h_+(p)         = residual at L on prompt p, last token
    d_harmful_raw  = mean(h_+(P_harmful))      − mean(h_+(P_harmless))
    d_over_raw     = mean(h_+(P_over_refusal)) − mean(h_+(P_harmless))

    d_over             = d_over_raw / ‖d_over_raw‖
    d_actually_harmful = d_harmful_raw − (d_harmful_raw · d_over) d_over
    d_actually_harmful = d_actually_harmful / ‖d_actually_harmful‖

Ablation: standard projection-removal h − (h·d̂) d̂ with d̂ =
d_actually_harmful at L.

Because Maskey requires THREE prompt sets but the base Technique.fit
contract takes only two (harmful, harmless), this class exposes a
separate `set_over_refusal(prompts)` method that the runner must call
BEFORE fit(). fit() raises RuntimeError if it hasn't been called.

The runner in refusal_bench/runner.py is currently structured around the
two-set contract; updating it to plumb over-refusal prompts is a
follow-up PR. Until then this class is wired in and self-contained but
can't be invoked from the bench loop without that runner change.
"""

from typing import Callable, List, Optional, Tuple

import torch

from model import get_model  # noqa: F401 — imported for parity with sibling techniques
from research import apply_chat_template, make_ablation_hook

from ..technique import Technique


class Maskey(Technique):
    """Decomposed single-direction ablation: harmful minus over-refusal component."""

    name = "Maskey (over-refusal decomposition)"
    paper_url = "https://arxiv.org/abs/2603.27518"

    def __init__(self) -> None:
        super().__init__()
        self._over_refusal_prompts: Optional[List[str]] = None
        self._unit_direction: torch.Tensor | None = None
        # Diagnostics for write-up / debugging.
        self._harmful_raw_norm: float = 0.0
        self._over_raw_norm: float = 0.0
        self._actually_harmful_raw_norm: float = 0.0
        self._cos_harmful_over: float = 0.0

    # ------------------------------------------------------------------
    # Three-set extension to the base contract
    # ------------------------------------------------------------------

    def set_over_refusal(self, prompts: List[str]) -> None:
        """
        Store the over-refusal prompt set. Must be called BEFORE fit().

        Prompts come from XSTest (2308.01263) or equivalent — benign
        questions whose surface lexicon (kill, shoot, attack, etc.)
        triggers over-cautious refusal in instruction-tuned models.

        Population happens via backend/over_refusal_pairs.py, populated
        in turn by backend/scripts/build_over_refusal_pairs.py.
        """
        if not prompts:
            raise ValueError(
                "Maskey.set_over_refusal: need at least 1 over-refusal prompt"
            )
        self._over_refusal_prompts = list(prompts)

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
        """
        Extract two raw difference-of-means directions (harmful and
        over-refusal, each against the shared harmless baseline), project
        the over-refusal direction out of the harmful direction, and store
        the renormalized residual as the ablation direction.
        """
        if self._over_refusal_prompts is None:
            raise RuntimeError(
                "Maskey.fit: set_over_refusal(prompts) must be called before "
                "fit(). Maskey requires three prompt sets — harmful, harmless, "
                "and over-refusal — but the base Technique.fit contract only "
                "passes the first two. See class docstring."
            )
        if not harmful_prompts or not harmless_prompts:
            raise ValueError(
                "Maskey.fit: need at least 1 prompt per class (harmful + harmless)"
            )

        device = model.cfg.device

        def last_token_resid(prompt: str) -> torch.Tensor:
            formatted = apply_chat_template(prompt)
            _logits, cache = model.run_with_cache(formatted)
            resid = cache[f"blocks.{layer}.hook_resid_post"]
            return resid[:, -1, :].squeeze(0).detach()

        # Stage 1: per-class mean last-token residuals at L.
        harmful_mean = torch.stack(
            [last_token_resid(p) for p in harmful_prompts]
        ).mean(dim=0)
        harmless_mean = torch.stack(
            [last_token_resid(p) for p in harmless_prompts]
        ).mean(dim=0)
        over_mean = torch.stack(
            [last_token_resid(p) for p in self._over_refusal_prompts]
        ).mean(dim=0)

        # Stage 2: two raw difference-of-means directions sharing harmless baseline.
        d_harmful_raw = harmful_mean - harmless_mean
        d_over_raw = over_mean - harmless_mean

        harmful_raw_norm = float(d_harmful_raw.norm().item())
        over_raw_norm = float(d_over_raw.norm().item())
        if harmful_raw_norm < 1e-8:
            raise RuntimeError(
                "Maskey: harmful raw direction has near-zero norm; "
                "harmful vs harmless sets are too similar to extract a direction."
            )
        if over_raw_norm < 1e-8:
            raise RuntimeError(
                "Maskey: over-refusal raw direction has near-zero norm; "
                "over-refusal vs harmless sets are too similar to extract a direction."
            )

        # Stage 3: subtract the over-refusal component from the harmful direction.
        #   d_over             = d_over_raw / ‖d_over_raw‖
        #   d_actually_harmful = d_harmful_raw − (d_harmful_raw · d_over) d_over
        d_over = d_over_raw / over_raw_norm
        projection_scalar = float(torch.dot(d_harmful_raw, d_over).item())
        d_actually_harmful = d_harmful_raw - projection_scalar * d_over

        # Stage 4: renormalize. If harmful and over-refusal are nearly collinear
        # the residual collapses — surface that explicitly rather than producing
        # a numerically degenerate unit vector.
        actually_norm = d_actually_harmful.norm()
        if actually_norm.item() < 1e-8:
            raise RuntimeError(
                "Maskey: actually-harmful residual collapsed after subtracting "
                "the over-refusal component. The harmful direction is (within "
                "numerical tolerance) a scalar multiple of the over-refusal "
                "direction — i.e. for this prompt set the decomposition is "
                "degenerate. Check that the harmful and over-refusal sets are "
                "drawn from distinct sources (JailbreakBench vs XSTest)."
            )

        self._unit_direction = (d_actually_harmful / actually_norm).to(device)
        self._layer = layer

        # Diagnostics — handy when writing up which fraction of the harmful
        # signal was actually over-refusal flavor.
        self._harmful_raw_norm = harmful_raw_norm
        self._over_raw_norm = over_raw_norm
        self._actually_harmful_raw_norm = float(actually_norm.item())
        self._cos_harmful_over = projection_scalar / (
            max(harmful_raw_norm, 1e-12) * 1.0  # d_over is unit
        )

        self._fitted = True

    def make_ablation_hook(self) -> Tuple[str, Callable]:
        if not self._fitted or self._unit_direction is None or self._layer is None:
            raise RuntimeError("Maskey.make_ablation_hook called before fit()")
        return (
            f"blocks.{self._layer}.hook_resid_post",
            make_ablation_hook(self._unit_direction),
        )
