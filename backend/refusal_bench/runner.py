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

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from model import get_model, get_model_name
from research import apply_chat_template

from .harmfulness_probe import (
    auc_permutation_p,
    bootstrap_auc_ci,
    bootstrap_discriminability_ci,
    discriminability,
    discriminability_permutation_p,
    evaluate_probe,
    extract_last_token_residuals,
    train_probe,
)
from .scoring import refusal_count, wilson_ci
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
    # |cos(probe weight, ablated direction)| — the dissociation confound check.
    probe_cosine: Optional[float] = None
    # 95% CIs (Wilson for refusal rate, percentile bootstrap for AUC) and the
    # one-sided permutation p that the POST-ablation AUC beats chance. These
    # turn the bare deltas into claims a reviewer can interrogate at n~5–15.
    refusal_rate_baseline_ci: Optional[Tuple[float, float]] = None
    refusal_rate_ablated_ci: Optional[Tuple[float, float]] = None
    harmfulness_auc_pre_ci: Optional[Tuple[float, float]] = None
    harmfulness_auc_post_ci: Optional[Tuple[float, float]] = None
    harmfulness_auc_post_p: Optional[float] = None
    # Discriminability = |AUC - 0.5|, in [0, 0.5]: distance from chance, sign
    # ignored. A probe reading perfectly BACKWARDS (AUC 0.05) still carries the
    # harmfulness information, so "did the signal survive?" must be scored on
    # distance from chance, not on AUC itself. The `_p` here is TWO-sided.
    # Raw AUC fields above are kept because only their sign gives the direction.
    harmfulness_discriminability_pre: Optional[float] = None
    harmfulness_discriminability_post: Optional[float] = None
    harmfulness_discriminability_pre_ci: Optional[Tuple[float, float]] = None
    harmfulness_discriminability_post_ci: Optional[Tuple[float, float]] = None
    harmfulness_discriminability_post_p: Optional[float] = None
    # Sample sizes behind those intervals — so the UI can say "AUC on N points".
    n_refusal_eval: Optional[int] = None  # denominator of the refusal rate
    n_auc_eval: Optional[int] = None       # eval-harmful + eval-harmless


@dataclass
class BenchResult:
    """Full bench output. Serializable to JSON via asdict()."""

    model_name: str
    layer: int
    n_extraction_pairs: int
    n_eval_prompts: int
    probe_train_auc: float
    probe_test_auc: float
    probe_cv_auc_mean: Optional[float] = None
    probe_cv_auc_std: Optional[float] = None
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
    seed: Optional[int] = None,
) -> str:
    """
    Generate a completion. If hook_name/hook_fn are None, no ablation.

    When `seed` is given we reseed torch before generating, so the baseline and
    ablated completions for the SAME prompt draw identical samples — the
    refusal-rate delta then reflects the ablation, not sampling noise. This is
    what makes the bench `seed` genuinely reproducible (previously it only
    seeded the train/eval split, not generation).
    """
    model = get_model()
    if seed is not None:
        torch.manual_seed(seed)
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


def probe_direction_cosine(probe, unit_direction) -> Optional[float]:
    """
    |cos(probe weight vector, ablated unit direction)|.

    High → the ablation removes the same axis the probe reads, so a low
    post-ablation AUC would be expected. Low → "AUC stayed high after ablation"
    is near-guaranteed by construction (the ablation and the probe look at
    near-orthogonal directions), NOT evidence the harmfulness representation
    survived. The key confound to surface in the Zhao dissociation story.
    Returns None for techniques that don't reduce to a single direction.
    """
    if unit_direction is None or probe is None:
        return None
    try:
        w = np.asarray(probe.coef_).reshape(-1)
        if hasattr(unit_direction, "detach"):
            # .float() guards bfloat16, which has no numpy equivalent.
            d = unit_direction.detach().float().cpu().numpy().reshape(-1)
        else:
            d = np.asarray(unit_direction).reshape(-1)
        if w.shape != d.shape:
            return None
        wn = float(np.linalg.norm(w))
        dn = float(np.linalg.norm(d))
        if wn < 1e-12 or dn < 1e-12:
            return None
        return float(abs(np.dot(w, d) / (wn * dn)))
    except Exception:
        return None


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
    over_refusal_prompts: Optional[List[str]] = None,
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
        over_refusal_prompts: optional third set of benign-but-edgy prompts
            (XSTest-style). Maskey needs this to extract the over-refusal
            direction and subtract it from the harmful direction. Other
            techniques ignore it.
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
    # Bootstrap band on the baseline (pre-ablation) AUC. Shared across techniques
    # since they all share this probe + eval set, so compute it once.
    n_auc_eval = len(baseline_labels)
    baseline_auc_ci = bootstrap_auc_ci(baseline_labels, baseline_eval["p_harm"], seed=seed)
    # Same band expressed as distance from chance — see harmfulness_probe's
    # "why AUC alone is the wrong statistic" note.
    baseline_disc = discriminability(baseline_auc)
    baseline_disc_ci = bootstrap_discriminability_ci(
        baseline_labels, baseline_eval["p_harm"], seed=seed
    )

    baseline_completions = [
        _generate_with_hook(p, None, None, max_new_tokens, temperature, seed=seed + i)
        for i, p in enumerate(eval_harmful)
    ]
    n_refusal_eval = len(baseline_completions)
    baseline_refusal_k = refusal_count(baseline_completions)
    baseline_refusal = baseline_refusal_k / n_refusal_eval if n_refusal_eval else 0.0
    baseline_refusal_ci = wilson_ci(baseline_refusal_k, n_refusal_eval)

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
                refusal_rate_baseline_ci=baseline_refusal_ci,
                harmfulness_auc_pre_ci=baseline_auc_ci,
                harmfulness_discriminability_pre=baseline_disc,
                harmfulness_discriminability_pre_ci=baseline_disc_ci,
                n_refusal_eval=n_refusal_eval,
                n_auc_eval=n_auc_eval,
            ))
            continue

        try:
            technique = TECHNIQUES[tname]()

            # Techniques that need a third prompt set (currently only Maskey)
            # expose `set_over_refusal(prompts)`; call it before fit if so.
            if hasattr(technique, "set_over_refusal"):
                if not over_refusal_prompts:
                    raise RuntimeError(
                        f"{tname} requires over_refusal_prompts but none "
                        f"were passed to run_bench. Populate over_refusal_pairs "
                        f"and pass via the over_refusal_prompts kwarg."
                    )
                technique.set_over_refusal(over_refusal_prompts)  # type: ignore[attr-defined]

            technique.fit(model, extraction_harmful, extraction_harmless, layer)
            hook_name, hook_fn = technique.make_ablation_hook()

            # Refusal rate with ablation hook active
            ablated_completions = [
                _generate_with_hook(
                    p, hook_name, hook_fn, max_new_tokens, temperature, seed=seed + i
                )
                for i, p in enumerate(eval_harmful)
            ]
            ablated_refusal_k = refusal_count(ablated_completions)
            ablated_refusal = (
                ablated_refusal_k / n_refusal_eval if n_refusal_eval else 0.0
            )
            ablated_refusal_ci = wilson_ci(ablated_refusal_k, n_refusal_eval)

            # Post-ablation AUC at the same extract layer
            abl_harmful_resid = _extract_residuals_with_hook(eval_harmful, layer, hook_name, hook_fn)
            abl_harmless_resid = _extract_residuals_with_hook(eval_harmless, layer, hook_name, hook_fn)
            ablated_residuals = torch.cat([abl_harmful_resid, abl_harmless_resid], dim=0)
            ablated_eval = evaluate_probe(probe, ablated_residuals, labels=baseline_labels)
            ablated_auc = ablated_eval["auc"] if ablated_eval["auc"] is not None else 0.5
            # Band on the post-ablation AUC + the significance of "still above
            # chance" — the actual residual-harmfulness claim.
            ablated_auc_ci = bootstrap_auc_ci(baseline_labels, ablated_eval["p_harm"], seed=seed)
            ablated_auc_p = auc_permutation_p(baseline_labels, ablated_eval["p_harm"], seed=seed)
            # Sign-agnostic version: an inverted probe still carries the signal.
            ablated_disc = discriminability(ablated_auc)
            ablated_disc_ci = bootstrap_discriminability_ci(
                baseline_labels, ablated_eval["p_harm"], seed=seed
            )
            ablated_disc_p = discriminability_permutation_p(
                baseline_labels, ablated_eval["p_harm"], seed=seed
            )

            # Confound diagnostic: does the ablation touch the probe's axis?
            cosine = probe_direction_cosine(probe, technique.unit_direction())

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
                probe_cosine=cosine,
                refusal_rate_baseline_ci=baseline_refusal_ci,
                refusal_rate_ablated_ci=ablated_refusal_ci,
                harmfulness_auc_pre_ci=baseline_auc_ci,
                harmfulness_auc_post_ci=ablated_auc_ci,
                harmfulness_auc_post_p=ablated_auc_p,
                harmfulness_discriminability_pre=baseline_disc,
                harmfulness_discriminability_post=ablated_disc,
                harmfulness_discriminability_pre_ci=baseline_disc_ci,
                harmfulness_discriminability_post_ci=ablated_disc_ci,
                harmfulness_discriminability_post_p=ablated_disc_p,
                n_refusal_eval=n_refusal_eval,
                n_auc_eval=n_auc_eval,
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
                refusal_rate_baseline_ci=baseline_refusal_ci,
                harmfulness_auc_pre_ci=baseline_auc_ci,
                harmfulness_discriminability_pre=baseline_disc,
                harmfulness_discriminability_pre_ci=baseline_disc_ci,
                n_refusal_eval=n_refusal_eval,
                n_auc_eval=n_auc_eval,
            ))

    return BenchResult(
        model_name=model_name,
        layer=layer,
        n_extraction_pairs=len(extraction_harmful),
        n_eval_prompts=len(eval_harmful),
        probe_train_auc=probe_info["train_auc"],
        probe_test_auc=probe_info["test_auc"],
        probe_cv_auc_mean=probe_info.get("cv_auc_mean"),
        probe_cv_auc_std=probe_info.get("cv_auc_std"),
        results=results,
    )


def json_safe(obj: object) -> object:
    """
    Recursively replace non-finite floats (NaN / ±Inf) with None.

    Error rows carry float('nan') metrics. Python's json.dumps would emit a
    bare `NaN` token (invalid JSON that browser JSON.parse rejects), and
    Starlette's JSONResponse uses json.dumps(allow_nan=False), which raises —
    so a single errored technique would 500 the whole /refusal-bench response.
    Mapping to None keeps every consumer (live API, local-script artifacts)
    on strict, parseable JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def serialize(result: BenchResult) -> dict:
    """JSON-friendly, NaN-free dict for HTTP responses (see json_safe)."""
    return {k: json_safe(v) for k, v in asdict(result).items()}
