"""
Research Functions - Moon's interpretability logic powered by TransformerLens

This module contains the core research functions from Moon's notebooks,
adapted to use TransformerLens's cleaner cache API instead of manual hooks.

The research questions and analysis logic are Moon's - we just swapped
the plumbing from raw HuggingFace to TransformerLens.
"""

from typing import Callable, List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from model import get_model, get_model_name, run_with_cache


# -----------------------------------------------------------------------------
# Chat template helper
# -----------------------------------------------------------------------------
# Llama-3.2-*-Instruct expects prompts wrapped in special chat tokens
# (<|begin_of_text|>, <|start_header_id|>user<|end_header_id|>, ...).
# Refusal behavior in particular only fires inside that envelope — raw
# prompts will not produce realistic refusals. Base models like gpt2-small
# have no chat template, so we no-op there.

def apply_chat_template(prompt: str) -> str:
    """
    Wrap `prompt` in the loaded model's chat template if it's instruct-tuned.

    Detection is name-based ("Instruct" / "instruct" in the model name).
    For non-instruct models the prompt is returned unchanged, so existing
    GPT-2 behavior is preserved.

    Call this from generation paths (steering, refusal-direction ablation)
    and from contrastive-vector extraction on instruct models. Do NOT call
    it from logit_lens / attention / gradients — those probe raw
    representations and should see the prompt as-is.
    """
    name = get_model_name() or ""
    if "instruct" not in name.lower():
        return prompt
    tokenizer = get_model().tokenizer
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


# -----------------------------------------------------------------------------
# Shared ablation hook
# -----------------------------------------------------------------------------
# Used by `ablate_along_direction` (UI side-by-side generation) and by
# refusal_bench.harmfulness_probe.extract_with_ablation (probe scoring).
# Single source of truth for h' = h - (h · d̂) d̂.

def make_ablation_hook(unit_direction: torch.Tensor) -> Callable:
    """Build a TransformerLens fwd hook that removes the projection along d̂."""
    def ablation_hook(activation, hook):
        # activation: [batch, seq_len, d_model]
        coeffs = (activation * unit_direction).sum(dim=-1, keepdim=True)
        activation[:, :, :] = activation - coeffs * unit_direction
        return activation
    return ablation_hook


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

    Math: logits_L = ln_final(hidden_state_L) @ W_U + b_U
    where W_U is the unembedding matrix ([d_model, d_vocab]) and b_U its bias.
    """
    model = get_model()
    tokens, logits, cache = run_with_cache(prompt)

    # Unembedding: logits = ln_final(resid) @ W_U + b_U.
    #
    # Applying the final layer norm is NOT optional. W_U is trained to read a
    # normalized residual stream; projecting the raw residual (as this code
    # used to) distorts every layer's prediction and, critically, makes the
    # last layer disagree with the model's own logits — destroying the one
    # checkable property the logit lens has. With fold_ln=True (TransformerLens's
    # from_pretrained default) ln_final is a LayerNormPre that centers and
    # scales to unit variance, with the learnable affine folded into W_U/b_U;
    # this recipe is correct in both the folded and unfolded cases.
    W_U = model.W_U  # [d_model, d_vocab]
    b_U = model.b_U  # [d_vocab]

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

        # Normalize, then project to vocabulary space. ln_final applied to a
        # single [d_model] vector normalizes over the last dim (see note above).
        normed = model.ln_final(last_token_resid)
        vocab_logits = normed @ W_U + b_U  # shape: [d_vocab]
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


# -----------------------------------------------------------------------------
# Steered Generation
# -----------------------------------------------------------------------------

def generate_steered(
    prompt: str,
    steering_vector: List[float],
    alpha: float,
    layer: int,
    max_new_tokens: int = 30,
    seed: int = 42,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Generate text with a steering vector injected at the specified layer.

    The steering vector is added to the residual stream during the forward pass:
      h_steered = h_original + alpha * v_steering

    Positive alpha steers toward the positive direction (e.g., positive sentiment).
    Negative alpha steers toward the negative direction.

    Determinism: the steered and baseline generations must differ ONLY by the
    intervention, not by sampling noise. We default to greedy decoding
    (do_sample=False) so the comparison is exactly controlled. If sampling is
    requested, we reseed torch before BOTH calls so they draw the same samples
    and any divergence is still attributable to the steering vector.
    """
    model = get_model()

    # Llama-Instruct needs the chat-template envelope to behave realistically.
    formatted = apply_chat_template(prompt)
    tokens = model.to_tokens(formatted)

    steering_tensor = torch.tensor(steering_vector, dtype=torch.float32, device=model.cfg.device)

    def steering_hook(activation, hook):
        # activation shape: [batch, seq_len, d_model]
        # Add the steering vector (scaled by alpha) to all token positions
        activation[:, :, :] = activation[:, :, :] + alpha * steering_tensor
        return activation

    # Generate with the hook active at the specified layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    def _generate():
        if do_sample:
            torch.manual_seed(seed)
        return model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        steered_output = _generate()

    # Baseline (no steering) under the same decoding regime for a fair contrast.
    baseline_output = _generate()

    return {
        "prompt": prompt,
        "layer": layer,
        "alpha": alpha,
        "steered_text": model.to_string(steered_output[0]),
        "baseline_text": model.to_string(baseline_output[0]),
    }


# -----------------------------------------------------------------------------
# Direction Ablation (Projection Removal)
# -----------------------------------------------------------------------------
# The causal counterpart to steering: instead of ADDING a vector, REMOVE the
# component of the residual stream that lies along the direction. This is the
# standard primitive for testing causal claims of the form "direction d
# mediates behavior X" (Arditi et al., 2024 — refusal direction).

def ablate_along_direction(
    prompt: str,
    direction: List[float],
    layer: int,
    max_new_tokens: int = 30,
    seed: int = 42,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Generate text with the projection along `direction` removed at `layer`.

    Hook math: h' = h - (h · d̂) d̂   where d̂ = direction / ||direction||

    Zeros the residual stream's component along d̂ at every token position of
    the chosen layer. Contrast with generate_steered, which adds alpha * v.

    Returns both ablated and baseline generations so the caller can show a
    side-by-side. Decoding is greedy by default so the ablated vs. baseline
    pair differs ONLY by the intervention; if sampling is requested we reseed
    before both calls so they share the same draws.
    """
    model = get_model()

    direction_tensor = torch.tensor(
        direction, dtype=torch.float32, device=model.cfg.device
    )
    norm = direction_tensor.norm()
    if norm.item() < 1e-8:
        raise RuntimeError("direction has near-zero norm; cannot ablate")
    unit_direction = direction_tensor / norm

    # Llama-Instruct needs the chat-template envelope to behave realistically.
    formatted = apply_chat_template(prompt)
    tokens = model.to_tokens(formatted)

    ablation_hook = make_ablation_hook(unit_direction)
    hook_name = f"blocks.{layer}.hook_resid_post"

    def _generate():
        if do_sample:
            torch.manual_seed(seed)
        return model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    with model.hooks(fwd_hooks=[(hook_name, ablation_hook)]):
        ablated_output = _generate()

    baseline_output = _generate()

    return {
        "prompt": prompt,
        "layer": layer,
        "direction_norm_before": round(float(norm.item()), 4),
        "ablated_text": model.to_string(ablated_output[0]),
        "baseline_text": model.to_string(baseline_output[0]),
    }
