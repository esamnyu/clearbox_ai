"""
NeuroScope-Web API Server

FastAPI endpoints that expose Moon's research functions to the React frontend.
This replaces the browser-based transformers.js worker with server-side
TransformerLens inference.

Run with: uvicorn main:app --reload --port 8000
API docs: http://localhost:8000/docs
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import model
import research

# -----------------------------------------------------------------------------
# App Setup
# -----------------------------------------------------------------------------

app = FastAPI(
    title="NeuroScope-Web API",
    description="Moon's interpretability research powered by TransformerLens",
    version="0.1.0",
)

# Allow requests from the React frontend (typically port 3000 or 3001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------
# Pydantic models define the shape of data going in and out of endpoints

class LoadRequest(BaseModel):
    model_name: str = Field(default="gpt2-small", description="Model to load")


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Input text to analyze")


class GradientRequest(BaseModel):
    prompt: str
    target_token: str = Field(..., description="Token to compute gradients toward")


class AttentionRequest(BaseModel):
    prompt: str
    layer: int = Field(ge=0, le=11, description="Layer index (0-11 for GPT-2)")
    head: int = Field(ge=0, le=11, description="Head index (0-11 for GPT-2)")


class SteeringRequest(BaseModel):
    positive_prompts: List[str] = Field(..., min_length=1)
    negative_prompts: List[str] = Field(..., min_length=1)
    layer: int = Field(default=6, ge=0, le=11, description="Layer for extraction")


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

    GPT-2 small (~500MB) takes a few seconds to load on first call.
    Subsequent calls with the same model return immediately.
    """
    try:
        result = model.load_model(req.model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logit-lens")
async def logit_lens(req: PromptRequest):
    """
    Run logit lens analysis on a prompt.

    Shows what the model would predict if we stopped at each layer.
    This reveals how predictions refine through the network.

    Moon's Eiffel Tower example: layers 0-10 predict "the",
    layer 11 finally predicts "Paris".
    """
    try:
        return research.logit_lens(req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/attention")
async def attention_pattern(req: AttentionRequest):
    """
    Get attention weights for a specific layer and head.

    Returns a matrix showing how each token attends to others.
    Useful for finding interesting attention patterns like:
    - Previous token heads (copying behavior)
    - Position heads (attending to specific positions)
    - Induction heads (in-context learning)
    """
    try:
        return research.get_attention_pattern(req.prompt, req.layer, req.head)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/gradients")
async def token_gradients(req: GradientRequest):
    """
    Compute gradient-based token importance.

    Shows which input tokens most influence the target prediction.
    High gradient norm = modifying this token has big effect.

    This is the foundation for adversarial attacks:
    to change "Paris" → "Rome", focus on high-gradient tokens.
    """
    try:
        return research.compute_token_gradients(req.prompt, req.target_token)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/steering-vector")
async def steering_vector(req: SteeringRequest):
    """
    Compute a steering vector from contrastive prompts.

    The vector points from negative → positive in activation space.
    Add it during generation to steer toward positive sentiment.
    Subtract it to steer toward negative sentiment.

    Moon's default: use layer 6, which balances semantic content
    with malleability.
    """
    try:
        return research.extract_steering_vector(
            req.positive_prompts,
            req.negative_prompts,
            req.layer,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/contrastive-pairs")
async def contrastive_pairs():
    """
    Get Moon's curated sentiment pairs.

    These are validated to tokenize to the same length,
    which is required for computing steering vectors.
    """
    pairs = research.get_contrastive_pairs()
    return {
        "pairs": [{"positive": p, "negative": n} for p, n in pairs],
        "count": len(pairs),
    }


@app.post("/pca-trajectories")
async def pca_trajectories(req: PromptRequest):
    """
    Get 3D PCA coordinates for all tokens across all layers.

    This powers Moon's interactive 3D visualization showing
    how token representations evolve through the network.

    Returns x, y, z coordinates for each (token, layer) pair,
    ready for Plotly or Three.js rendering on the frontend.
    """
    try:
        return research.compute_pca_trajectories(req.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
