"""
Activation Patching — causal localization of behavior over model components.

Activation patching runs the model on a BASE prompt while splicing in cached
activations from a SOURCE prompt at one site (layer × position, or layer ×
head), then measures how much a behavioral metric moves. Sweeping sites yields
a map of where the computation that distinguishes the two prompts lives.

The two directions answer different questions and can disagree; both are
exposed and neither is labelled "the circuit":

- **denoising** (patch clean → corrupted run): does restoring this site
  SUFFICE to recover the clean behavior?
- **noising** (patch corrupted → clean run): is this site NECESSARY — does
  corrupting it alone destroy the clean behavior?

Metric: logit difference between two single-token answers at the final
position (Wang et al. 2022, IOI). `normalized` rescales it so 0 = the base
run's own value and 1 = the source run's value; in denoising 1 means full
restoration, in noising 1 means full destruction. Deliberately NOT clamped:
values outside [0, 1] mean the patch overshot or backfired, which is the
surprising result worth seeing.

Like logit_lens / attention, this probes raw representations — prompts are
used as-is, with no chat template.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from model import get_model

VALID_DIRECTIONS = ("denoising", "noising")
VALID_COMPONENTS = ("resid_post", "head_z")


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def logit_diff(final_logits: torch.Tensor, answer_id: int, baseline_id: int) -> float:
    """logits[answer] - logits[baseline] at one position. final_logits: [d_vocab]."""
    return float((final_logits[answer_id] - final_logits[baseline_id]).item())


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) in nats between the next-token distributions of two logit vectors."""
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    return float(torch.sum(log_p.exp() * (log_p - log_q)).item())


def normalized_recovery(patched: float, base: float, source: float) -> Optional[float]:
    """
    Rescale a patched metric: 0 = base run's value, 1 = source run's value.

    Returns None when |source - base| is numerically zero — the two runs don't
    disagree on the metric, so "fraction of the gap crossed" is undefined.
    """
    denom = source - base
    if abs(denom) < 1e-9:
        return None
    return (patched - base) / denom


# -----------------------------------------------------------------------------
# Core: token-level patching sweep
# -----------------------------------------------------------------------------

def _make_position_patch_hook(source_act: torch.Tensor, pos: int) -> Callable:
    """Hook that overwrites one position of the residual stream with `source_act`."""
    def hook_fn(activation, hook):
        # activation: [batch, seq_len, d_model]
        activation[:, pos, :] = source_act[pos, :]
        return activation
    return hook_fn


def _make_head_patch_hook(source_z: torch.Tensor, head: int) -> Callable:
    """Hook that overwrites one head's output (all positions) with `source_z`."""
    def hook_fn(activation, hook):
        # activation: [batch, seq_len, n_heads, d_head]
        activation[:, :, head, :] = source_z[:, head, :]
        return activation
    return hook_fn


def patch_grid(
    base_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    answer_id: int,
    baseline_id: int,
    component: str = "resid_post",
    layers: Optional[List[int]] = None,
    positions: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    max_runs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sweep single-site patches of `source_tokens` activations into `base_tokens` runs.

    The metric at every cell is logit_diff(answer_id, baseline_id) at the final
    position. Returns raw per-cell metrics plus both unpatched baselines;
    direction semantics (which prompt is base vs source) are the caller's.

    Raises ValueError on shape mismatch, out-of-range sites, or a sweep larger
    than `max_runs` (each cell is a full forward pass — callers exposing this
    publicly should cap it).
    """
    model = get_model()

    if base_tokens.shape != source_tokens.shape:
        raise ValueError(
            f"base and source prompts must tokenize to the same shape; got "
            f"{tuple(base_tokens.shape)} vs {tuple(source_tokens.shape)}. "
            f"Pick prompts that differ only in same-token-length spans."
        )
    seq_len = base_tokens.shape[1]

    if component not in VALID_COMPONENTS:
        raise ValueError(f"component must be one of {VALID_COMPONENTS}, got {component!r}")

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    layers = list(range(n_layers)) if layers is None else list(layers)
    for layer in layers:
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} out of range for n_layers={n_layers}")

    if component == "resid_post":
        positions = list(range(seq_len)) if positions is None else [
            p if p >= 0 else seq_len + p for p in positions
        ]
        for pos in positions:
            if not 0 <= pos < seq_len:
                raise ValueError(f"position {pos} out of range for seq_len={seq_len}")
        cols = positions
    else:
        heads = list(range(n_heads)) if heads is None else list(heads)
        for head in heads:
            if not 0 <= head < n_heads:
                raise ValueError(f"head {head} out of range for n_heads={n_heads}")
        cols = heads

    n_runs = len(layers) * len(cols)
    if max_runs is not None and n_runs > max_runs:
        raise ValueError(
            f"sweep of {len(layers)} layers x {len(cols)} sites = {n_runs} forward "
            f"passes exceeds the cap of {max_runs}; restrict layers/positions/heads"
        )

    hook_of_layer = (
        (lambda layer: f"blocks.{layer}.hook_resid_post")
        if component == "resid_post"
        else (lambda layer: f"blocks.{layer}.attn.hook_z")
    )
    wanted = {hook_of_layer(layer) for layer in layers}

    with torch.no_grad():
        source_logits, source_cache = model.run_with_cache(
            source_tokens, names_filter=lambda name: name in wanted
        )
        base_logits = model(base_tokens)

    base_ld = logit_diff(base_logits[0, -1], answer_id, baseline_id)
    source_ld = logit_diff(source_logits[0, -1], answer_id, baseline_id)

    rows = []
    for layer in layers:
        hook_name = hook_of_layer(layer)
        source_act = source_cache[hook_name][0]  # [seq, d_model] or [seq, n_heads, d_head]
        cells = []
        for col in cols:
            if component == "resid_post":
                hook_fn = _make_position_patch_hook(source_act, col)
                cell_key = "position"
            else:
                hook_fn = _make_head_patch_hook(source_act, col)
                cell_key = "head"
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    base_tokens, fwd_hooks=[(hook_name, hook_fn)]
                )
            final = patched_logits[0, -1]
            patched_ld = logit_diff(final, answer_id, baseline_id)
            cells.append({
                cell_key: col,
                "logit_diff": round(patched_ld, 6),
                "normalized": _round_opt(normalized_recovery(patched_ld, base_ld, source_ld)),
                "kl_from_base": round(kl_divergence(final, base_logits[0, -1]), 6),
                "kl_to_source": round(kl_divergence(final, source_logits[0, -1]), 6),
            })
        rows.append({"layer": layer, "cells": cells})

    return {
        "component": component,
        "layers": layers,
        ("positions" if component == "resid_post" else "heads"): cols,
        "base_logit_diff": round(base_ld, 6),
        "source_logit_diff": round(source_ld, 6),
        "grid": rows,
    }


def _round_opt(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 6)


# -----------------------------------------------------------------------------
# String-level wrapper (API surface)
# -----------------------------------------------------------------------------

def _resolve_single_token(answer: str) -> int:
    """Map an answer string to exactly one token id, or fail with a usable message."""
    model = get_model()
    try:
        return int(model.to_single_token(answer))
    except Exception:
        try:
            pieces = model.to_str_tokens(answer, prepend_bos=False)
            detail = f" (splits into {pieces})"
        except Exception:
            detail = ""
        raise ValueError(
            f"answer {answer!r} is not a single token for this tokenizer"
            f"{detail}; pick a single-token answer — for GPT-2 that usually "
            f"means a leading space, e.g. ' Paris'"
        )


def run_activation_patching(
    clean_prompt: str,
    corrupted_prompt: str,
    clean_answer: str,
    corrupted_answer: str,
    direction: str = "denoising",
    component: str = "resid_post",
    layers: Optional[List[int]] = None,
    positions: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    max_runs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full activation-patching sweep between a clean/corrupted prompt pair.

    The metric is always logit_diff = logits[clean_answer] - logits[corrupted_answer]
    at the final position, so `clean_logit_diff` should be positive and
    `corrupted_logit_diff` negative (or at least smaller) when the pair is
    well-formed; a note is attached when it isn't.

    direction:
        "denoising" — base = corrupted run, source = clean activations.
        "noising"   — base = clean run, source = corrupted activations.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {VALID_DIRECTIONS}, got {direction!r}")

    model = get_model()
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    answer_id = _resolve_single_token(clean_answer)
    baseline_id = _resolve_single_token(corrupted_answer)
    if answer_id == baseline_id:
        raise ValueError(
            f"clean_answer and corrupted_answer resolve to the same token id "
            f"({answer_id}); the logit-diff metric would be identically zero"
        )

    if direction == "denoising":
        base_tokens, source_tokens = corrupted_tokens, clean_tokens
    else:
        base_tokens, source_tokens = clean_tokens, corrupted_tokens

    result = patch_grid(
        base_tokens,
        source_tokens,
        answer_id,
        baseline_id,
        component=component,
        layers=layers,
        positions=positions,
        heads=heads,
        max_runs=max_runs,
    )

    # Re-express the direction-relative baselines in clean/corrupted terms.
    if direction == "denoising":
        corrupted_ld, clean_ld = result["base_logit_diff"], result["source_logit_diff"]
    else:
        clean_ld, corrupted_ld = result["base_logit_diff"], result["source_logit_diff"]

    notes = []
    if clean_ld <= corrupted_ld:
        notes.append(
            "clean prompt does not favor clean_answer over corrupted_answer "
            f"(clean logit_diff {clean_ld} <= corrupted {corrupted_ld}); "
            "check the answers aren't swapped"
        )

    return {
        "direction": direction,
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "clean_answer": clean_answer,
        "corrupted_answer": corrupted_answer,
        "tokens": model.to_str_tokens(base_tokens[0]),
        "clean_logit_diff": clean_ld,
        "corrupted_logit_diff": corrupted_ld,
        "notes": notes,
        **result,
    }
