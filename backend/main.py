"""
NeuroScope-Web API Server

FastAPI endpoints that expose Moon's research functions to the React frontend.
This replaces the browser-based transformers.js worker with server-side
TransformerLens inference.

Run with: uvicorn main:app --reload --port 8000
API docs: http://localhost:8000/docs
"""

import os
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import model
import patching
import research
import refusal_pairs as refusal_pairs_module
from refusal_bench.harmfulness_probe import (
    evaluate_probe,
    extract_last_token_residuals,
    extract_with_ablation,
    train_probe,
)
from refusal_bench.runner import run_bench, serialize as serialize_bench
from refusal_bench.techniques import TECHNIQUES

# -----------------------------------------------------------------------------
# App Setup
# -----------------------------------------------------------------------------

app = FastAPI(
    title="NeuroScope-Web API",
    description="Moon's interpretability research powered by TransformerLens",
    version="0.1.0",
)

# CORS origins are configurable so the same image runs locally and on HF Spaces.
# Set ALLOWED_ORIGINS to your deployed frontend URL(s), e.g.
#   ALLOWED_ORIGINS=https://neuroscope.vercel.app,https://preview.neuroscope.vercel.app
_default_origins = "http://localhost:3000,http://localhost:3001"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------
# Layer/head upper bounds cover GPT-2-small (12L, 12H), Llama-3.2-1B
# (16L, 32H), and Llama-3.2-3B (28L, 24H). Static pydantic caps validate
# request shape; per-model bounds are enforced at runtime via _validate_layer.


class LoadRequest(BaseModel):
    model_name: str = Field(default="gpt2-small", description="Model to load")


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Input text to analyze")


class GradientRequest(BaseModel):
    prompt: str
    target_token: str = Field(..., description="Token to compute gradients toward")


class AttentionRequest(BaseModel):
    prompt: str
    layer: int = Field(ge=0, le=27, description="Layer index (model-dependent)")
    head: int = Field(ge=0, le=31, description="Head index (model-dependent)")


class SteeringRequest(BaseModel):
    positive_prompts: List[str] = Field(..., min_length=1)
    negative_prompts: List[str] = Field(..., min_length=1)
    layer: int = Field(default=6, ge=0, le=27, description="Layer for extraction")


class SteeredGenerationRequest(BaseModel):
    prompt: str
    steering_vector: List[float]
    alpha: float = Field(default=1.0, description="Steering strength")
    layer: int = Field(default=6, ge=0, le=27)
    max_new_tokens: int = Field(default=30, ge=1, le=100)
    seed: int = Field(default=42, ge=0, description="RNG seed used when do_sample=True")
    do_sample: bool = Field(
        default=False,
        description="Greedy by default so the before/after differs only by the intervention",
    )


class AblationRequest(BaseModel):
    prompt: str
    direction: List[float] = Field(
        ..., description="Direction to project out of the residual stream"
    )
    layer: int = Field(default=6, ge=0, le=27, description="Layer for ablation")
    max_new_tokens: int = Field(default=30, ge=1, le=100)
    seed: int = Field(default=42, ge=0, description="RNG seed used when do_sample=True")
    do_sample: bool = Field(
        default=False,
        description="Greedy by default so the before/after differs only by the intervention",
    )


class PatchRequest(BaseModel):
    """
    Activation-patching sweep between a clean/corrupted prompt pair.

    The prompts must tokenize to the same length. Answers must be single
    tokens (for GPT-2 that usually means a leading space, e.g. " Mary").
    Each grid cell costs one full forward pass, so sweeps are capped at
    MAX_PATCH_RUNS cells — restrict layers/positions/heads to stay under it.
    """
    clean_prompt: str
    corrupted_prompt: str
    clean_answer: str = Field(..., description="Single-token answer the clean prompt favors")
    corrupted_answer: str = Field(..., description="Single-token answer the corrupted prompt favors")
    direction: Literal["denoising", "noising"] = Field(
        default="denoising",
        description="denoising = clean acts into corrupted run (sufficiency); "
        "noising = corrupted acts into clean run (necessity)",
    )
    component: Literal["resid_post", "head_z"] = Field(
        default="resid_post",
        description="resid_post sweeps layer x position; head_z sweeps layer x head",
    )
    layers: Optional[List[int]] = Field(default=None, description="Default: all layers")
    positions: Optional[List[int]] = Field(
        default=None, description="resid_post only; negatives index from the end. Default: all"
    )
    heads: Optional[List[int]] = Field(default=None, description="head_z only. Default: all heads")


class RefusalBenchRequest(BaseModel):
    """
    Run the Refusal Bench: a head-to-head comparison of refusal-ablation
    techniques on the loaded model, scored by refusal rate + harmfulness-
    probe AUC.

    techniques: subset of refusal_bench.techniques.TECHNIQUES keys
        (e.g. ["arditi", "cosmic", "cheng", "wollschlager"]).
    layer: residual-stream layer for extraction + ablation.
    harmful_prompts/harmless_prompts: contrastive pairs. The runner
        splits 80/20 (configurable) into extraction + eval folds.
    over_refusal_prompts: optional benign-but-edgy prompts (XSTest-style).
        Required by the "maskey" technique, which subtracts the
        over-refusal direction from the harmful direction; other
        techniques ignore it.
    """
    technique_names: List[str] = Field(..., min_length=1)
    layer: int = Field(ge=0, le=27)
    harmful_prompts: List[str] = Field(..., min_length=5)
    harmless_prompts: List[str] = Field(..., min_length=5)
    over_refusal_prompts: Optional[List[str]] = Field(default=None, min_length=1)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    max_new_tokens: int = Field(default=32, ge=1, le=128)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    seed: int = Field(default=42, ge=0)


class HarmfulnessProbeRequest(BaseModel):
    """
    Train (and optionally re-evaluate) a Zhao-style harmfulness probe.

    - layer: residual-stream layer used for both probe training and (if a
      direction is provided) ablation.
    - ablation_direction: optional. If set, the probe trained on baseline
      residuals is re-evaluated on residuals collected while this direction
      is projected out at `layer`. Mean p_harm staying high == residual
      stream still encodes harmfulness even after surface refusal collapses.
    """
    harmful_prompts: List[str] = Field(..., min_length=2)
    harmless_prompts: List[str] = Field(..., min_length=2)
    layer: int = Field(
        ge=0,
        le=27,
        description="Layer for residual extraction (and ablation if direction given)",
    )
    ablation_direction: Optional[List[float]] = None


# -----------------------------------------------------------------------------
# Runtime validation
# -----------------------------------------------------------------------------

def _validate_layer(layer: int) -> None:
    """Reject layer indices that exceed the loaded model's depth."""
    m = model.get_model()
    if layer >= m.cfg.n_layers:
        raise HTTPException(
            status_code=422,
            detail=f"layer {layer} >= n_layers {m.cfg.n_layers}",
        )


def _validate_head(head: int) -> None:
    """Reject head indices that exceed the loaded model's n_heads."""
    m = model.get_model()
    if head >= m.cfg.n_heads:
        raise HTTPException(
            status_code=422,
            detail=f"head {head} >= n_heads {m.cfg.n_heads}",
        )


# Public-deploy guards. The HF Space is unauthenticated, so restrict which
# models can be pulled and cap the work any single request can trigger.
ALLOWED_MODELS = {
    "gpt2-small",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
}
MAX_PROMPT_CHARS = 2000
MAX_BENCH_PROMPTS = 200
# Each activation-patching cell is a full forward pass; 256 covers a complete
# 12x12 GPT-2 head grid or a 12-layer x 21-position resid grid, while keeping
# the worst-case request bounded on the free CPU tier.
MAX_PATCH_RUNS = 256


def _validate_model_name(name: str) -> None:
    """Reject models outside the supported whitelist (no arbitrary HF pulls)."""
    if name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"model '{name}' not allowed. Supported: {sorted(ALLOWED_MODELS)}",
        )


def _validate_prompt(prompt: str) -> None:
    """Bound prompt length so a single request can't pin the free CPU tier."""
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"prompt too long ({len(prompt)} chars > {MAX_PROMPT_CHARS})",
        )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
async def health():
    """Simple health check - useful for verifying the server is running."""
    return {"status": "ok", "service": "neuroscope-api"}


@app.post("/load")
async def load_model(req: LoadRequest):
    """
    Load a model into memory. Must be called before other endpoints.

    GPT-2 small (~500MB) takes a few seconds to load on first call. Llama
    models are gated and require HF_TOKEN in the environment.
    Subsequent calls with the same model return immediately.
    """
    try:
        _validate_model_name(req.model_name)
        return model.load_model(req.model_name)
    except HTTPException:
        raise  # preserve 422 from the whitelist check
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logit-lens")
async def logit_lens(req: PromptRequest):
    """
    Run logit lens analysis on a prompt.
    Shows what the model would predict if we stopped at each layer.
    """
    try:
        return research.logit_lens(req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/attention")
async def attention_pattern(req: AttentionRequest):
    """Get attention weights for a specific layer and head."""
    try:
        _validate_layer(req.layer)
        _validate_head(req.head)
        return research.get_attention_pattern(req.prompt, req.layer, req.head)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/gradients")
async def token_gradients(req: GradientRequest):
    """Compute gradient-based token importance."""
    try:
        return research.compute_token_gradients(req.prompt, req.target_token)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/steering-vector")
async def steering_vector(req: SteeringRequest):
    """Compute a steering vector from contrastive prompts (difference of means)."""
    try:
        _validate_layer(req.layer)
        return research.extract_steering_vector(
            req.positive_prompts,
            req.negative_prompts,
            req.layer,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/contrastive-pairs")
async def contrastive_pairs():
    """Get Moon's curated sentiment pairs (validated to tokenize to same length)."""
    pairs = research.get_contrastive_pairs()
    return {
        "pairs": [{"positive": p, "negative": n} for p, n in pairs],
        "count": len(pairs),
    }


@app.get("/refusal-pairs")
async def refusal_pairs():
    """
    Get curated refusal-direction contrastive pairs (harmful + harmless).

    Populate via `python backend/scripts/build_refusal_pairs.py` — pulls
    JailbreakBench + Alpaca, length-matches on Llama-3.2-1B tokenizer.
    Returns count=0 until populated.
    """
    pairs = refusal_pairs_module.get_refusal_pairs()
    return {
        "pairs": [{"harmful": h, "harmless": s} for h, s in pairs],
        "count": len(pairs),
    }


@app.post("/generate-steered")
async def generate_steered(req: SteeredGenerationRequest):
    """Generate text with a steering vector injected at a specific layer."""
    try:
        _validate_layer(req.layer)
        _validate_prompt(req.prompt)
        return research.generate_steered(
            req.prompt,
            req.steering_vector,
            req.alpha,
            req.layer,
            req.max_new_tokens,
            seed=req.seed,
            do_sample=req.do_sample,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ablate-direction")
async def ablate_direction(req: AblationRequest):
    """
    Generate text with a direction projected out of the residual stream.

    Implements h' = h - (h · d̂) d̂ at the chosen layer. Standard primitive
    for testing claims like "direction d mediates behavior X" (Arditi 2024).
    Returns ablated and baseline generations for side-by-side comparison.
    """
    try:
        _validate_layer(req.layer)
        _validate_prompt(req.prompt)
        return research.ablate_along_direction(
            req.prompt,
            req.direction,
            req.layer,
            req.max_new_tokens,
            seed=req.seed,
            do_sample=req.do_sample,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/patch")
async def activation_patch(req: PatchRequest):
    """
    Activation-patching sweep over (layer x position) or (layer x head).

    Runs the model on the base prompt while splicing in the source prompt's
    cached activation at one site per cell, measuring logit_diff(clean_answer
    - corrupted_answer) at the final position. `normalized` is 0 at the base
    run's own value and 1 at the source run's: in denoising that means full
    restoration (sufficiency), in noising full destruction (necessity). The
    two directions answer different questions and can disagree — neither
    alone identifies "the circuit".
    """
    try:
        _validate_prompt(req.clean_prompt)
        _validate_prompt(req.corrupted_prompt)
        for layer in req.layers or []:
            _validate_layer(layer)
        return patching.run_activation_patching(
            req.clean_prompt,
            req.corrupted_prompt,
            req.clean_answer,
            req.corrupted_answer,
            direction=req.direction,
            component=req.component,
            layers=req.layers,
            positions=req.positions,
            heads=req.heads,
            max_runs=MAX_PATCH_RUNS,
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/harmfulness-probe")
async def harmfulness_probe(req: HarmfulnessProbeRequest):
    """
    Train a linear harmfulness probe on residuals at `layer` (Zhao 2507.11878).

    Always returns the baseline (no-ablation) train/test AUCs and per-prompt
    P(harmful). If `ablation_direction` is supplied, also re-evaluates the
    probe on residuals collected while that direction is ablated at `layer`,
    returning the post-ablation P(harmful) on the harmful prompt set.

    The probe DOES NOT generalize across layers — call per layer for a sweep.
    """
    try:
        _validate_layer(req.layer)

        harmful_resid = extract_last_token_residuals(req.harmful_prompts, req.layer)
        harmless_resid = extract_last_token_residuals(req.harmless_prompts, req.layer)

        train_result = train_probe(harmful_resid, harmless_resid)
        probe = train_result["model"]

        pre_eval = evaluate_probe(
            probe,
            harmful_resid,
            labels=[1] * harmful_resid.shape[0],
        )

        response = {
            "layer": req.layer,
            "n_harmful": len(req.harmful_prompts),
            "n_harmless": len(req.harmless_prompts),
            "train_auc": train_result["train_auc"],
            "test_auc": train_result["test_auc"],
            "cv_auc_mean": train_result.get("cv_auc_mean"),
            "cv_auc_std": train_result.get("cv_auc_std"),
            "n_train": train_result["n_train"],
            "n_test": train_result["n_test"],
            "pre_ablation_p_harm": pre_eval["p_harm"],
            "pre_ablation_mean_p_harm": pre_eval["mean_p_harm"],
            "post_ablation_p_harm": None,
            "post_ablation_mean_p_harm": None,
        }

        if req.ablation_direction is not None:
            ablated_resid = extract_with_ablation(
                req.harmful_prompts,
                layer_extract=req.layer,
                ablation_direction=req.ablation_direction,
                ablation_layer=req.layer,
            )
            post_eval = evaluate_probe(
                probe,
                ablated_resid,
                labels=[1] * ablated_resid.shape[0],
            )
            response["post_ablation_p_harm"] = post_eval["p_harm"]
            response["post_ablation_mean_p_harm"] = post_eval["mean_p_harm"]

        return response
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/refusal-bench/techniques")
async def list_bench_techniques():
    """List the refusal-ablation techniques registered in the bench."""
    return {
        "techniques": [
            {
                "key": key,
                "name": cls.name,
                "paper_url": cls.paper_url,
            }
            for key, cls in sorted(TECHNIQUES.items())
        ],
        "count": len(TECHNIQUES),
    }


@app.post("/refusal-bench")
async def refusal_bench(req: RefusalBenchRequest):
    """
    Run the Refusal Bench end-to-end.

    Trains a shared harmfulness probe on the extraction split, then loops
    over every requested technique, fits it on the same extraction data,
    and scores it with refusal-rate (keyword-based) + probe AUC on the
    held-out eval split. Returns one row per technique.

    Two-axis story: a technique with high |Δ refusal rate| but low |Δ AUC|
    has suppressed verbal refusal without removing the model's internal
    harmfulness representation — the Zhao 2507.11878 dissociation, here
    measured across multiple ablation methods.
    """
    try:
        _validate_layer(req.layer)
        over_refusal = req.over_refusal_prompts or []
        total_prompts = (
            len(req.harmful_prompts) + len(req.harmless_prompts) + len(over_refusal)
        )
        if total_prompts > MAX_BENCH_PROMPTS:
            raise HTTPException(
                status_code=422,
                detail=f"too many prompts (> {MAX_BENCH_PROMPTS}); split the run",
            )

        result = run_bench(
            technique_names=req.technique_names,
            layer=req.layer,
            harmful_prompts=req.harmful_prompts,
            harmless_prompts=req.harmless_prompts,
            over_refusal_prompts=req.over_refusal_prompts,
            test_fraction=req.test_fraction,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            seed=req.seed,
        )
        return serialize_bench(result)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pca-trajectories")
async def pca_trajectories(req: PromptRequest):
    """3D PCA coordinates for all tokens across all layers."""
    try:
        return research.compute_pca_trajectories(req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
