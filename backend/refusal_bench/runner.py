"""
Refusal Bench runner — orchestrate a head-to-head comparison of techniques.

Pipeline:
    1. Split contrastive prompts into extraction set (80%) and eval set (20%).
    2. Train ONE harmfulness probe on baseline residuals from the extraction
       set. The probe is shared across techniques so AUC numbers are
       directly comparable.
    3. Compute baseline refusal-rate + AUC on the eval set (no ablation).
    4. For each technique:
         a. Instantiate; fit on the extraction set.
         b. Get its ablation hook.
         c. Generate completions on eval-harmful with the hook active;
            compute post-ablation refusal rate.
         d. Extract residuals on eval-harmful + eval-harmless with the hook
            active; score with the trained probe; compute post-ablation AUC.
         e. Record Δ refusal-rate and Δ AUC.

The two-axis result (Δ refusal-rate vs Δ AUC) is the headline novelty: a
technique that drops refusal-rate to ~0 while keeping AUC near baseline has
suppressed verbal refusal but left the harmfulness representation intact.
This is the Zhao 2507.11878 dissociation, here measured across six
techniques on the same model.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from model import get_model, get_model_name
from research import apply_chat_template

from .harmfulness_probe import (
    evaluate_probe,
    extract_last_token_residuals,
    train_probe,
)
from .scoring import refusal_rate
from .techniques import TECHNIQUES


# -----------------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------------

@dataclass
class TechniqueResult:
    """One row of the bench table."""

    name: str
    paper_url: str
    layer_used: int
    refusal_rate_baseline: float
    refusal_rate_ablated: float
    delta_refusal_rate: float
    harmfulness_auc_pre: float
    harmfulness_auc_post: float
    delta_auc: float
    elapsed_seconds: float
    error: Optional[str] = None


@dataclass
class BenchResult:
    """Full bench output. Serializable to JSON via asdict()."""

    model_name: str
    layer: int
    n_extraction_pairs: int
    n_eval_prompts: int
    probe_train_auc: float
    probe_test_auc: float
    results: List[TechniqueResult] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _generate_with_hook(
    prompt: str,
    hook_name: Optional[str],
    hook_fn: Optional[Callable],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate a completion. If hook_name/hook_fn are None, no ablation."""
    model = get_model()
    formatted = apply_chat_template(prompt)
    tokens = model.to_tokens(formatted)

    if hook_name is None or hook_fn is None:
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    else:
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

    full_text = model.to_string(output[0])
    # Strip the prompt itself so refusal-detection sees only the completion
    prompt_text = model.to_string(tokens[0])
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text):]
    return full_text


def _extract_residuals_with_hook(
    prompts: List[str],
    extract_layer: int,
    hook_name: Optional[str],
    hook_fn: Optional[Callable],
) -> torch.Tensor:
    """
    Last-token residuals at `extract_layer`. If hook_name/hook_fn supplied,
    they are installed during the forward pass (so the residuals reflect
    the post-ablation state when extract_layer == ablation layer or
    downstream of it).
    """
    model = get_model()
    extract_hook = f"blocks.{extract_layer}.hook_resid_post"

    residuals: List[torch.Tensor] = []
    for prompt in prompts:
        formatted = apply_chat_template(prompt)
        if hook_name is None or hook_fn is None:
            _logits, cache = model.run_with_cache(formatted)
        else:
            with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                _logits, cache = model.run_with_cache(formatted)
        last = cache[extract_hook][:, -1, :].squeeze(0).detach().cpu()
        residuals.append(last)

    return torch.stack(residuals, dim=0)


def _split(
    items: List[str],
    test_fraction: float,
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    """Shuffle then split; minimum 2 in test fold so probe AUC is defined."""
    shuffled = items.copy()
    rng.shuffle(shuffled)
    n_test = max(2, int(round(len(shuffled) * test_fraction)))
    return shuffled[n_test:], shuffled[:n_test]


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def run_bench(
    technique_names: List[str],
    layer: int,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    *,
    test_fraction: float = 0.2,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    seed: int = 42,
) -> BenchResult:
    """
    Run each named technique on the same data and return scored results.

    Args:
        technique_names: keys from refusal_bench.techniques.TECHNIQUES.
            Unknown names produce an error row instead of crashing the bench.
        layer: residual-stream layer used for direction extraction AND
            ablation (techniques like COSMIC may select a different layer
            via fit() — that's recorded in `layer_used`).
        harmful_prompts: contrastive prompts the model should refuse.
        harmless_prompts: contrastive prompts the model should comply with.
        test_fraction: portion held out for eval (default 0.2).
        max_new_tokens: completion length budget for refusal-rate measurement.
        temperature: sampling temperature for generation.
        seed: split + sampling determinism.

    Returns:
        BenchResult — serialize via dataclasses.asdict() for JSON.
    """
    rng = random.Random(seed)
    extraction_harmful, eval_harmful = _split(harmful_prompts, test_fraction, rng)
    extraction_harmless, eval_harmless = _split(harmless_prompts, test_fraction, rng)

    model = get_model()
    model_name = get_model_name() or "unknown"

    # ── 1. Train probe on extraction-set baseline residuals ──────────────
    ext_harmful_resid = extract_last_token_residuals(extraction_harmful, layer)
    ext_harmless_resid = extract_last_token_residuals(extraction_harmless, layer)
    probe_info = train_probe(ext_harmful_resid, ext_harmless_resid)
    probe = probe_info["model"]

    # ── 2. Baseline (no-ablation) eval-set metrics ───────────────────────
    baseline_harmful_resid = extract_last_token_residuals(eval_harmful, layer)
    baseline_harmless_resid = extract_last_token_residuals(eval_harmless, layer)
    baseline_residuals = torch.cat([baseline_harmful_resid, baseline_harmless_resid], dim=0)
    baseline_labels = [1] * len(eval_harmful) + [0] * len(eval_harmless)
    baseline_eval = evaluate_probe(probe, baseline_residuals, labels=baseline_labels)
    baseline_auc = baseline_eval["auc"] if baseline_eval["auc"] is not None else 0.5

    baseline_completions = [
        _generate_with_hook(p, None, None, max_new_tokens, temperature)
        for p in eval_harmful
    ]
    baseline_refusal = refusal_rate(baseline_completions)

    # ── 3. Per-technique loop ────────────────────────────────────────────
    results: List[TechniqueResult] = []
    for tname in technique_names:
        start = time.time()

        if tname not in TECHNIQUES:
            results.append(TechniqueResult(
                name=tname,
                paper_url="",
                layer_used=layer,
                refusal_rate_baseline=baseline_refusal,
                refusal_rate_ablated=float("nan"),
                delta_refusal_rate=float("nan"),
                harmfulness_auc_pre=baseline_auc,
                harmfulness_auc_post=float("nan"),
                delta_auc=float("nan"),
                elapsed_seconds=0.0,
                error=f"unknown technique: {tname}. Known: {sorted(TECHNIQUES)}",
            ))
            continue

        try:
            technique = TECHNIQUES[tname]()
            technique.fit(model, extraction_harmful, extraction_harmless, layer)
            hook_name, hook_fn = technique.make_ablation_hook()

            # Refusal rate with ablation hook active
            ablated_completions = [
                _generate_with_hook(p, hook_name, hook_fn, max_new_tokens, temperature)
                for p in eval_harmful
            ]
            ablated_refusal = refusal_rate(ablated_completions)

            # Post-ablation AUC at the same extract layer
            abl_harmful_resid = _extract_residuals_with_hook(eval_harmful, layer, hook_name, hook_fn)
            abl_harmless_resid = _extract_residuals_with_hook(eval_harmless, layer, hook_name, hook_fn)
            ablated_residuals = torch.cat([abl_harmful_resid, abl_harmless_resid], dim=0)
            ablated_eval = evaluate_probe(probe, ablated_residuals, labels=baseline_labels)
            ablated_auc = ablated_eval["auc"] if ablated_eval["auc"] is not None else 0.5

            elapsed = time.time() - start
            results.append(TechniqueResult(
                name=technique.name,
                paper_url=technique.paper_url,
                layer_used=technique._layer if technique._layer is not None else layer,
                refusal_rate_baseline=baseline_refusal,
                refusal_rate_ablated=ablated_refusal,
                delta_refusal_rate=ablated_refusal - baseline_refusal,
                harmfulness_auc_pre=baseline_auc,
                harmfulness_auc_post=ablated_auc,
                delta_auc=ablated_auc - baseline_auc,
                elapsed_seconds=elapsed,
            ))
        except Exception as e:
            elapsed = time.time() - start
            results.append(TechniqueResult(
                name=tname,
                paper_url="",
                layer_used=layer,
                refusal_rate_baseline=baseline_refusal,
                refusal_rate_ablated=float("nan"),
                delta_refusal_rate=float("nan"),
                harmfulness_auc_pre=baseline_auc,
                harmfulness_auc_post=float("nan"),
                delta_auc=float("nan"),
                elapsed_seconds=elapsed,
                error=f"{type(e).__name__}: {e}",
            ))

    return BenchResult(
        model_name=model_name,
        layer=layer,
        n_extraction_pairs=len(extraction_harmful),
        n_eval_prompts=len(eval_harmful),
        probe_train_auc=probe_info["train_auc"],
        probe_test_auc=probe_info["test_auc"],
        results=results,
    )


def serialize(result: BenchResult) -> dict:
    """JSON-friendly dict for HTTP responses."""
    return asdict(result)
