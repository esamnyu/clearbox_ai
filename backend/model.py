"""
Model Manager - TransformerLens wrapper

Provides a singleton HookedTransformer instance for the backend.
This replaces Moon's manual HuggingFace setup with cleaner abstractions.
"""

from typing import Optional, Dict, Any
import torch
from transformer_lens import HookedTransformer

# Global model instance (avoids reloading ~500MB on each request)
_model: Optional[HookedTransformer] = None
_model_name: Optional[str] = None


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str = "gpt2-small") -> Dict[str, Any]:
    """
    Load a HookedTransformer model.

    Moon's notebooks used:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.config.output_attentions = True

    TransformerLens equivalent:
        model = HookedTransformer.from_pretrained("gpt2-small")
        # Attentions and hidden states captured automatically via run_with_cache()
    """
    global _model, _model_name

    if _model is not None and _model_name == name:
        return {
            "status": "already_loaded",
            "model_name": name,
            "n_layers": _model.cfg.n_layers,
            "d_model": _model.cfg.d_model,
        }

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


def run_with_cache(prompt: str):
    """
    Run inference and capture all activations.

    Moon's notebooks did this manually:
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

    TransformerLens:
        logits, cache = model.run_with_cache(prompt)
        # cache["blocks.0.hook_resid_post"] = residual after layer 0
        # cache["blocks.0.attn.hook_pattern"] = attention pattern layer 0

    Returns:
        tokens: list of string tokens
        logits: output logits tensor
        cache: ActivationCache with all intermediate activations
    """
    model = get_model()
    tokens = model.to_str_tokens(prompt)
    logits, cache = model.run_with_cache(prompt)
    return tokens, logits, cache
