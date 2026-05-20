"""
Model Manager - TransformerLens wrapper

Provides a singleton HookedTransformer instance for the backend.

Supports GPT-2-small (ungated) and meta-llama/Llama-3.2-{1B,3B}-Instruct
(gated; set HF_TOKEN in the environment to load them).
"""

import os
from typing import Optional, Dict, Any
import torch
from transformer_lens import HookedTransformer

# Global model instance (avoids reloading on each request)
_model: Optional[HookedTransformer] = None
_model_name: Optional[str] = None
_hf_login_done: bool = False


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_hf_login() -> None:
    """
    Log into HuggingFace once per process if HF_TOKEN is set.

    Required for gated models (Llama-3.2-*). No-op if already logged in or
    if HF_TOKEN is not set. If login fails (network, bad token), we let
    the subsequent from_pretrained raise — failing softly here would just
    surface a confusing 401 later.
    """
    global _hf_login_done
    if _hf_login_done:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        _hf_login_done = True
    except Exception:
        # Don't block GPT-2 loads if login fails for any reason. The
        # downstream from_pretrained will raise if the model is gated.
        pass


def load_model(name: str = "gpt2-small") -> Dict[str, Any]:
    """
    Load a HookedTransformer model.

    Supported names:
        - "gpt2-small" (default, ungated)
        - "meta-llama/Llama-3.2-1B-Instruct" (gated; needs HF_TOKEN)
        - "meta-llama/Llama-3.2-3B-Instruct" (gated; needs HF_TOKEN)

    Returns the model's runtime config so the frontend can adapt layer
    dropdowns / hooks to the loaded model.
    """
    global _model, _model_name

    if _model is not None and _model_name == name:
        return {
            "status": "already_loaded",
            "model_name": name,
            "n_layers": _model.cfg.n_layers,
            "d_model": _model.cfg.d_model,
            "n_heads": _model.cfg.n_heads,
            "d_vocab": _model.cfg.d_vocab,
        }

    # Gated models (Llama) require an HF token; log in once if present.
    _ensure_hf_login()

    device = get_device()
    _model = HookedTransformer.from_pretrained(name, device=device)
    _model_name = name

    return {
        "status": "loaded",
        "model_name": name,
        "device": device,
        "n_layers": _model.cfg.n_layers,
        "d_model": _model.cfg.d_model,
        "n_heads": _model.cfg.n_heads,
        "d_vocab": _model.cfg.d_vocab,
    }


def get_model() -> HookedTransformer:
    """Get loaded model, raise if none."""
    if _model is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    return _model


def get_model_name() -> Optional[str]:
    """Return the name passed to the most recent load_model() call."""
    return _model_name


def run_with_cache(prompt: str):
    """
    Run inference and capture all activations.

    Returns (tokens, logits, cache) where cache has every intermediate
    activation indexed by hook name, e.g.:
        cache["blocks.0.hook_resid_post"]   — residual after layer 0
        cache["blocks.0.attn.hook_pattern"] — attention pattern at layer 0
    """
    model = get_model()
    tokens = model.to_str_tokens(prompt)
    logits, cache = model.run_with_cache(prompt)
    return tokens, logits, cache
