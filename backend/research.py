"""
Research Functions - Moon's interpretability logic powered by TransformerLens

This module contains the core research functions from Moon's notebooks,
adapted to use TransformerLens's cleaner cache API instead of manual hooks.

The research questions and analysis logic are Moon's - we just swapped
the plumbing from raw HuggingFace to TransformerLens.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from model import get_model, run_with_cache


# -----------------------------------------------------------------------------
# Logit Lens
# -----------------------------------------------------------------------------
# Moon's original: manually grabbed hidden states, multiplied by lm_head.weight
# TransformerLens: we use the cache and model.unembed() or direct W_U access

def logit_lens(prompt: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Apply the unembedding matrix to each layer's residual stream.

    This answers: "If we stopped the model at layer L, what would it predict?"

    The idea (from nostalgebraist's blog) is that each layer refines the
    prediction. Early layers often predict generic tokens like "the",
    while later layers converge on the contextually correct answer.

    Moon's notebook showed this beautifully with the Eiffel Tower example:
    layers 0-10 all predicted "the", but layer 11 finally predicted "Paris".

    Math: logits_L = hidden_state_L @ W_U.T
    where W_U is the unembedding matrix (vocab x hidden_dim)
    """
    model = get_model()
    tokens, logits, cache = run_with_cache(prompt)

    # W_U maps from hidden dimension to vocabulary
    # In GPT-2, this is tied to the embedding matrix
    W_U = model.W_U  # shape: [d_model, d_vocab]

    layer_predictions = []

    for layer_idx in range(model.cfg.n_layers + 1):
        # Layer 0 is the embedding, layers 1-12 are transformer blocks
        # TransformerLens uses "blocks.X.hook_resid_post" for post-layer residuals
        if layer_idx == 0:
            # Embedding layer - before any transformer blocks
            resid = cache["hook_embed"] + cache["hook_pos_embed"]
        else:
            # After transformer block (layer_idx - 1)
            resid = cache[f"blocks.{layer_idx - 1}.hook_resid_post"]

        # We only care about the last token position (next token prediction)
        last_token_resid = resid[0, -1, :]  # shape: [d_model]

        # Project to vocabulary space
        # Note: Moon's code skipped layer norm here for "raw" analysis
        # For prediction parity with the model, you'd apply ln_final first
        vocab_logits = last_token_resid @ W_U  # shape: [d_vocab]
        probs = F.softmax(vocab_logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        predictions = [
            {"token": model.to_string(idx.item()), "prob": round(p.item(), 4)}
            for p, idx in zip(top_probs, top_indices)
        ]

        layer_name = "Embed" if layer_idx == 0 else f"Layer {layer_idx - 1}"
        layer_predictions.append({"layer": layer_name, "top_k": predictions})

    return {"prompt": prompt, "tokens": tokens, "predictions": layer_predictions}


# -----------------------------------------------------------------------------
# Attention Patterns
# -----------------------------------------------------------------------------
# Moon's notebook visualized these as heatmaps to see what tokens attend to what

def get_attention_pattern(prompt: str, layer: int, head: int) -> Dict[str, Any]:
    """
    Extract attention weights for a specific layer and head.

    The attention pattern shows how each token "looks at" other tokens.
    Shape is [seq_len, seq_len] where entry [i,j] is how much token i
    attends to token j.

    Moon used these to identify interesting heads - some heads attend to
    the previous token (useful for copying), others attend to specific
    syntactic positions.

    GPT-2 small has 12 layers x 12 heads = 144 attention patterns to explore!
    """
    model = get_model()
    tokens, logits, cache = run_with_cache(prompt)

    # TransformerLens stores attention patterns at this hook point
    # Shape: [batch, n_heads, seq_len, seq_len]
    attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

    # Extract the specific head we want
    head_pattern = attn_pattern[0, head].cpu().tolist()  # [seq_len, seq_len]

    return {
        "prompt": prompt,
        "tokens": tokens,
        "layer": layer,
        "head": head,
        "pattern": head_pattern,
    }


# -----------------------------------------------------------------------------
# Gradient Analysis (Token Susceptibility)
# -----------------------------------------------------------------------------
# This is Moon's "Foundation for Adversarial Attacks" section
# It tells us which input tokens most influence a target prediction

def compute_token_gradients(prompt: str, target_token: str) -> Dict[str, Any]:
    """
    Compute how much each input token influences the target prediction.

    Moon's insight: if you want to steer the model from predicting "Paris"
    to predicting "Rome", which input tokens should you modify?

    The gradient norm tells us the "susceptibility" of each position.
    High gradient norm = changing this token has big impact on the target.

    Moon's example showed that "iff" (from Eiffel), "city", and "Tower"
    had the highest gradients when trying to change the prediction to Rome.
    This makes intuitive sense - these are the most "French" tokens.

    Math: We compute d(loss)/d(embedding) where loss = -log P(target)
    """
    model = get_model()

    # Tokenize
    tokens_tensor = model.to_tokens(prompt)  # [1, seq_len]
    str_tokens = model.to_str_tokens(prompt)
    target_id = model.to_single_token(target_token)

    # Get embeddings with gradient tracking
    # We need to manually build the forward pass to get gradients on embeddings
    embed = model.embed(tokens_tensor)  # [1, seq_len, d_model]
    pos_embed = model.pos_embed(tokens_tensor)

    # Combine and enable gradients
    # We keep a reference to this tensor since we want gradients w.r.t. it
    input_resid = (embed + pos_embed).detach().requires_grad_(True)

    # Forward through transformer blocks
    resid = input_resid
    for block in model.blocks:
        resid = block(resid)

    # Final layer norm and unembedding
    resid = model.ln_final(resid)
    logits = resid @ model.W_U  # [1, seq_len, d_vocab]

    # Loss: negative log probability of target token at last position
    last_logits = logits[0, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)
    loss = -log_probs[target_id]

    # Backpropagate
    loss.backward()

    # The gradient on input_resid tells us sensitivity per position
    # We take the L2 norm across the hidden dimension
    grad_norms = input_resid.grad[0].norm(dim=-1).tolist()  # [seq_len]

    # Normalize for easier interpretation (0 to 1 scale)
    max_norm = max(grad_norms)
    normalized = [g / max_norm if max_norm > 0 else 0 for g in grad_norms]

    return {
        "prompt": prompt,
        "target_token": target_token,
        "tokens": str_tokens,
        "gradient_norms": [
            {"token": t, "norm": round(g, 4), "normalized": round(n, 4)}
            for t, g, n in zip(str_tokens, grad_norms, normalized)
        ],
    }


# -----------------------------------------------------------------------------
# Steering Vectors
# -----------------------------------------------------------------------------
# Moon's steering_vectors.ipynb - the core of activation engineering

def get_contrastive_pairs() -> List[Tuple[str, str]]:
    """
    Moon's curated contrastive pairs for sentiment steering.

    These pairs are designed so that:
    1. They differ only in sentiment (positive vs negative)
    2. They tokenize to the same length (critical for subtraction!)

    Moon validated each pair's token length in the notebook.
    """
    return [
        ("I think this movie is amazing", "I think this movie is terrible"),
        ("The food at this restaurant is delicious", "The food at this restaurant is disgusting"),
        ("I am feeling very happy today", "I am feeling very sad today"),
        ("The product quality is excellent", "The product quality is awful"),
        ("My experience was wonderful", "My experience was horrible"),
        ("He is a very kind person", "He is a very mean person"),
        ("The weather is beautiful", "The weather is nasty"),
        ("This solution is perfect", "This solution is useless"),
    ]


def extract_steering_vector(
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer: int,
) -> Dict[str, Any]:
    """
    Compute a steering vector from contrastive examples.

    Moon's formula: v_steering = mean(h_positive) - mean(h_negative)

    This vector points in the direction of "positiveness" in activation space.
    Adding it during generation steers toward positive sentiment.
    Subtracting it steers toward negative sentiment.

    We extract from the last token position because GPT-2 aggregates
    context causally - the last token "knows" the full sequence.

    Layer choice matters:
    - Early layers (0-3): Low-level features, less semantic
    - Middle layers (4-8): Good for semantic steering
    - Late layers (9-11): Close to output, can be unstable
    Moon typically used layer 6 as a good default.
    """
    model = get_model()

    def get_last_token_activation(prompt: str, layer_idx: int) -> torch.Tensor:
        """Extract residual stream at layer for the last token."""
        _, _, cache = run_with_cache(prompt)
        resid = cache[f"blocks.{layer_idx}.hook_resid_post"]
        return resid[0, -1, :]  # [d_model]

    # Collect activations for both sets
    pos_activations = [get_last_token_activation(p, layer) for p in positive_prompts]
    neg_activations = [get_last_token_activation(n, layer) for n in negative_prompts]

    # Compute means
    mean_pos = torch.stack(pos_activations).mean(dim=0)
    mean_neg = torch.stack(neg_activations).mean(dim=0)

    # Steering vector: direction from negative to positive
    steering_vector = mean_pos - mean_neg

    return {
        "layer": layer,
        "n_positive": len(positive_prompts),
        "n_negative": len(negative_prompts),
        "vector_norm": round(steering_vector.norm().item(), 4),
        "vector": steering_vector.tolist(),
    }


# -----------------------------------------------------------------------------
# PCA Trajectories
# -----------------------------------------------------------------------------
# Moon's 3D visualization of how token representations evolve through layers

def compute_pca_trajectories(prompt: str) -> Dict[str, Any]:
    """
    Project all token representations through layers into 3D space.

    Moon's insight: tokens start at similar positions (embeddings) and
    diverge as they pass through layers. The trajectories reveal how
    the model processes different tokens.

    Questions this helps answer:
    - Do semantically related tokens stay close? (e.g., "Eiffel" and "Tower")
    - Where do trajectories diverge? Which layer differentiates roles?
    - Do function words (the, is) behave differently from content words?

    Moon noted that PC1 often captures ~97% of variance, suggesting
    the residual stream has a dominant direction (likely related to
    predicting the next token).
    """
    model = get_model()
    tokens, logits, cache = run_with_cache(prompt)

    # Collect all representations: every token at every layer
    all_vectors = []
    metadata = []

    for layer_idx in range(model.cfg.n_layers + 1):
        if layer_idx == 0:
            resid = cache["hook_embed"] + cache["hook_pos_embed"]
        else:
            resid = cache[f"blocks.{layer_idx - 1}.hook_resid_post"]

        for token_idx, token_str in enumerate(tokens):
            vec = resid[0, token_idx, :].cpu().numpy()
            all_vectors.append(vec)
            metadata.append({
                "token": token_str,
                "token_idx": token_idx,
                "layer": layer_idx,
            })

    # Fit PCA on all vectors together (unified coordinate space)
    import numpy as np
    vectors_matrix = np.stack(all_vectors)
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(vectors_matrix)

    # Attach coordinates to metadata
    results = []
    for i, meta in enumerate(metadata):
        results.append({
            **meta,
            "x": round(float(coords_3d[i, 0]), 4),
            "y": round(float(coords_3d[i, 1]), 4),
            "z": round(float(coords_3d[i, 2]), 4),
        })

    return {
        "prompt": prompt,
        "tokens": tokens,
        "variance_explained": [round(v, 4) for v in pca.explained_variance_ratio_],
        "trajectories": results,
    }
