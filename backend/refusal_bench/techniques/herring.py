"""
Herring Contrastive Neuron Ablation (CNA).

Herring, Naviasky, Malhotra (May 12, 2026):
"Targeted Neuron Modulation via Contrastive Pair Search."
https://arxiv.org/abs/2605.12290

Arditi / Wollschlager / COSMIC / Cheng all operate on the residual stream
— they isolate a direction (or stack of directions) in d_model-dimensional
space and project it out. Herring CNA is structurally different: it
identifies the sparse set of MLP neurons that fire most differentially
between harmful and harmless prompts at layer L, and zeros those neurons'
post-activations during the forward pass.

Why this is interesting for the bench: a successful neuron-ablation result
is a sufficient condition counter-example to the strong reading of the
"single-direction" claim. If zeroing ~0.1% of MLP neurons matches a
residual-stream projection's refusal-rate drop, refusal is at least
partially mediated by a localized neuron set inside the MLP, not just a
linear feature in the residual.

Hook point: `f"blocks.{layer}.mlp.hook_post"` — post-MLP activations of
shape [batch, seq_len, d_mlp]. d_mlp = 4 · d_model for GPT-2 (3072 for
gpt2-small) and 8192 for Llama-3.2-1B (so the same α = 0.001 selects ~3
neurons on GPT-2 vs ~8 on Llama-1B; α is the lever to keep neuron-count
roughly comparable across models if needed).

Algorithm (per the paper, condensed to the bench's data footprint):
    1. For each prompt p, capture cache[f"blocks.{L}.mlp.hook_post"]
       and take the LAST TOKEN row → vector in R^{d_mlp}.
    2. Stack into harmful_mlp [n, d_mlp] and harmless_mlp [n, d_mlp].
    3. Per-neuron score:
           s[i] = mean(harmful_mlp[:, i]) - mean(harmless_mlp[:, i])
       Sign tells us which class the neuron activates *for*, but for
       ablation we care about *differential* activation regardless of
       direction — so we select by |s|.
    4. Select top-k by |s|, k = round(α · d_mlp).
    5. Store the index set; the ablation hook zeros those columns of
       every MLP-post activation at L (all positions, not just last
       token — the activation is part of the forward pass for whatever
       token is currently being decoded).
"""

from typing import Callable, List, Tuple

import torch

from model import get_model  # noqa: F401 — imported for parity with sibling techniques
from research import apply_chat_template

from ..technique import Technique


class HerringCNA(Technique):
    """Sparse MLP-neuron ablation via contrastive pair search."""

    name = "Herring (contrastive neuron ablation)"
    paper_url = "https://arxiv.org/abs/2605.12290"

    def __init__(self, sparsity_fraction: float = 0.001) -> None:
        """
        Parameters
        ----------
        sparsity_fraction
            Fraction α of MLP neurons to ablate at layer L. Default 0.001
            (top 0.1%). Paper sweeps α ∈ [0.001, 0.01]. Must satisfy
            0 < α <= 1.

            d_mlp implications by model:
              - gpt2-small:    d_mlp = 3072  → α=0.001 picks  3 neurons
              - Llama-3.2-1B:  d_mlp = 8192  → α=0.001 picks  8 neurons
              - Llama-3.2-3B:  d_mlp = 8192  → α=0.001 picks  8 neurons
            On GPT-2 the floor is brutally coarse: a single neuron is
            ~0.033% of d_mlp, so α below ~0.0007 rounds to zero. The
            constructor doesn't enforce a minimum-k; fit() raises if
            round(α · d_mlp) < 1.
        """
        super().__init__()
        if not (0.0 < sparsity_fraction <= 1.0):
            raise ValueError(
                f"HerringCNA: sparsity_fraction must be in (0, 1], "
                f"got {sparsity_fraction}"
            )

        self._sparsity_fraction: float = float(sparsity_fraction)
        self._neuron_indices: torch.Tensor | None = None  # Long tensor, shape [k]
        self._d_mlp: int = 0
        # Diagnostics exposed for the bench / inspection UI.
        self._n_neurons_kept: int = 0
        self._d_mlp_total: int = 0

    def fit(
        self,
        model,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
    ) -> None:
        """
        Identify the top-k differentially-activating MLP neurons at `layer`.

        Steps:
          1. Cache last-token MLP-post activations on each contrastive set.
          2. Score each neuron by the SIGNED difference of class means,
             then rank by |score|.
          3. Store the top-k indices for the hook to zero out.
        """
        if not harmful_prompts or not harmless_prompts:
            raise ValueError("HerringCNA.fit: need at least 1 prompt per class")

        device = model.cfg.device
        d_mlp = int(model.cfg.d_mlp)
        if d_mlp <= 0:
            raise RuntimeError(
                f"HerringCNA: model.cfg.d_mlp is {d_mlp}; can't size MLP neuron set."
            )

        k = int(round(self._sparsity_fraction * d_mlp))
        if k < 1:
            raise RuntimeError(
                f"HerringCNA: sparsity_fraction={self._sparsity_fraction} on "
                f"d_mlp={d_mlp} rounds to 0 neurons; raise α."
            )

        hook_name = f"blocks.{layer}.mlp.hook_post"

        def last_token_mlp_post(prompt: str) -> torch.Tensor:
            formatted = apply_chat_template(prompt)
            _logits, cache = model.run_with_cache(formatted, names_filter=[hook_name])
            act = cache[hook_name]  # [1, seq_len, d_mlp]
            return act[:, -1, :].squeeze(0).detach()  # [d_mlp]

        # [n_harmful, d_mlp] and [n_harmless, d_mlp]
        harmful_mlp = torch.stack([last_token_mlp_post(p) for p in harmful_prompts])
        harmless_mlp = torch.stack([last_token_mlp_post(p) for p in harmless_prompts])

        # Signed per-neuron contrastive score; |·| for ranking only.
        # We rank on absolute value because a neuron that strongly fires for
        # *harmless* and weakly for *harmful* is just as informative about the
        # class boundary as the reverse — and the hook zeros it either way,
        # which mirrors the paper's framing of "differentially active" neurons.
        score = harmful_mlp.mean(dim=0) - harmless_mlp.mean(dim=0)  # [d_mlp]
        abs_score = score.abs()

        # torch.topk on the absolute scores gives the k most differentially
        # active neurons regardless of sign.
        _topk_vals, topk_idx = torch.topk(abs_score, k=k, largest=True, sorted=False)
        self._neuron_indices = topk_idx.to(device=device, dtype=torch.long)

        self._d_mlp = d_mlp
        self._d_mlp_total = d_mlp
        self._n_neurons_kept = int(k)
        self._layer = layer
        self._fitted = True

    def _make_neuron_zero_hook(
        self, neuron_indices: torch.Tensor
    ) -> Callable:
        """Closure that zeros the selected MLP-post neuron columns in place."""

        def hook(activation, hook_):
            # activation: [batch, seq_len, d_mlp]
            # neuron_indices: [k], Long
            activation[..., neuron_indices] = 0
            return activation

        return hook

    def make_ablation_hook(self) -> Tuple[str, Callable]:
        if (
            not self._fitted
            or self._neuron_indices is None
            or self._layer is None
        ):
            raise RuntimeError("HerringCNA.make_ablation_hook called before fit()")
        return (
            f"blocks.{self._layer}.mlp.hook_post",
            self._make_neuron_zero_hook(self._neuron_indices),
        )
