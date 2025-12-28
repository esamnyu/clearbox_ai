# GPT-2 Complete Data Extraction Guide

This document describes every tensor that can be extracted from GPT-2 during inference.

---

## Complete Data Flow Visualization

```
PROMPT: "The capital of France is"
                    │
                    ▼
            ┌──────────────────┐
            │   TOKENIZATION   │
            └──────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ EXTRACTABLE AT TOKENIZATION:                                │
│                                                             │
│ • token_ids: [464, 3139, 286, 4881, 318]  (int64)          │
│ • tokens: ["The", " capital", " of", " France", " is"]     │
│ • attention_mask: [1, 1, 1, 1, 1]                          │
│ • position_ids: [0, 1, 2, 3, 4]                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            ┌──────────────────┐
            │    EMBEDDING     │
            └──────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ EXTRACTABLE AT EMBEDDING LAYER:                             │
│                                                             │
│ • token_embeddings (wte):     [batch, seq, 768]            │
│   → Raw lookup from embedding table                         │
│                                                             │
│ • position_embeddings (wpe):  [batch, seq, 768]            │
│   → Positional encoding                                     │
│                                                             │
│ • combined_embeddings:        [batch, seq, 768]            │
│   → token_embed + position_embed (+ dropout)               │
│                                                             │
│ This is hidden_states[0] - the input to layer 0            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
       ┌──────────────────────────────────────┐
       │  FOR EACH LAYER (0-11):              │
       │                                      │
       │  ┌────────────────────────────────┐  │
       │  │     LAYER NORM 1 (ln_1)        │  │
       │  └────────────────────────────────┘  │
       │              │                       │
       │              ▼                       │
       │  ┌────────────────────────────────┐  │
       │  │    SELF-ATTENTION BLOCK        │  │
       │  │                                │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ c_attn projection      │   │  │
       │  │  │ [batch, seq, 2304]     │   │  │
       │  │  └─────────────────────────┘   │  │
       │  │              │                 │  │
       │  │              ▼                 │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ SPLIT INTO Q, K, V     │   │  │
       │  │  │ Each: [batch,seq,768]  │   │  │
       │  │  └─────────────────────────┘   │  │
       │  │              │                 │  │
       │  │              ▼                 │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ RESHAPE TO HEADS       │   │  │
       │  │  │ [batch,seq,12,64]      │   │  │
       │  │  └─────────────────────────┘   │  │
       │  │              │                 │  │
       │  │              ▼                 │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ COMPUTE ATTENTION      │   │  │
       │  │  │ QK^T / sqrt(64)        │   │  │
       │  │  │ [batch,12,seq,seq]     │   │  │
       │  │  └─────────────────────────┘   │  │
       │  │              │                 │  │
       │  │              ▼                 │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ SOFTMAX + DROPOUT      │   │  │
       │  │  │ [batch,12,seq,seq]     │   │  │
       │  │  └─────────────────────────┘   │  │
       │  │              │                 │  │
       │  │              ▼                 │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ APPLY TO VALUES        │   │  │
       │  │  │ attn @ V → [b,s,768]   │   │  │
       │  │  └─────────────────────────┘   │  │
       │  │              │                 │  │
       │  │              ▼                 │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ OUTPUT PROJECTION      │   │  │
       │  │  │ c_proj: 768 → 768      │   │  │
       │  │  └─────────────────────────┘   │  │
       │  └────────────────────────────────┘  │
       │              │                       │
       │              ▼                       │
       │  ┌────────────────────────────────┐  │
       │  │  RESIDUAL CONNECTION 1         │  │
       │  │  hidden = hidden + attn_out    │  │
       │  └────────────────────────────────┘  │
       │              │                       │
       │              ▼                       │
       │  ┌────────────────────────────────┐  │
       │  │     LAYER NORM 2 (ln_2)        │  │
       │  └────────────────────────────────┘  │
       │              │                       │
       │              ▼                       │
       │  ┌────────────────────────────────┐  │
       │  │         MLP BLOCK              │  │
       │  │                                │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ c_fc: 768 → 3072       │   │  │
       │  │  │ (4× expansion)         │   │  │
       │  │  └─────────────────────────┘   │  │
       │  │              │                 │  │
       │  │              ▼                 │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ GELU ACTIVATION        │   │  │
       │  │  │ [batch, seq, 3072]     │   │  │
       │  │  └─────────────────────────┘   │  │
       │  │              │                 │  │
       │  │              ▼                 │  │
       │  │  ┌─────────────────────────┐   │  │
       │  │  │ c_proj: 3072 → 768     │   │  │
       │  │  │ (projection back)      │   │  │
       │  │  └─────────────────────────┘   │  │
       │  └────────────────────────────────┘  │
       │              │                       │
       │              ▼                       │
       │  ┌────────────────────────────────┐  │
       │  │  RESIDUAL CONNECTION 2         │  │
       │  │  hidden = hidden + mlp_out     │  │
       │  └────────────────────────────────┘  │
       │              │                       │
       │              ▼                       │
       │      hidden_states[layer+1]          │
       │              │                       │
       └──────────────│───────────────────────┘
                      │
                      ▼
                (repeat 12×)
                      │
                      ▼
            ┌──────────────────┐
            │  FINAL LAYER     │
            │  NORM (ln_f)     │
            └──────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │   LM HEAD        │
            │  768 → 50257     │
            └──────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FINAL EXTRACTIONS:                                                  │
│                                                                     │
│ • final_layer_norm:        [batch, seq, 768]                       │
│   → Normalized final hidden state                                   │
│                                                                     │
│ • logits:                  [batch, seq, 50257]                     │
│   → Raw prediction scores for ALL vocab tokens                      │
│                                                                     │
│ • probabilities:           [batch, seq, 50257]                     │
│   → softmax(logits) - actual prediction probabilities              │
│                                                                     │
│ • top_k_tokens:            [batch, seq, K]                         │
│   → Most likely next tokens at each position                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Extractable Tensors Per Layer

### Attention Block Extractions

```
┌─────────────────────────────────────────────────────────────────────┐
│ ATTENTION BLOCK EXTRACTIONS (per layer):                            │
│                                                                     │
│ 1. Query (Q):           [batch, seq, 12_heads, 64_dim]             │
│    → What each position is "looking for"                            │
│    → Derived from: c_attn output[:, :, :768]                       │
│                                                                     │
│ 2. Key (K):             [batch, seq, 12_heads, 64_dim]             │
│    → What each position "offers" to be found                        │
│    → Derived from: c_attn output[:, :, 768:1536]                   │
│                                                                     │
│ 3. Value (V):           [batch, seq, 12_heads, 64_dim]             │
│    → What each position "transmits" when attended to                │
│    → Derived from: c_attn output[:, :, 1536:]                      │
│                                                                     │
│ 4. Raw Attention Scores (pre-softmax):                             │
│                         [batch, 12_heads, seq, seq]                │
│    → QK^T / sqrt(64)                                               │
│    → Shows "interest" before normalization                          │
│    → Unbounded values (can be very large or negative)              │
│                                                                     │
│ 5. Attention Weights (post-softmax):                               │
│                         [batch, 12_heads, seq, seq]                │
│    → Normalized probabilities                                       │
│    → Each row sums to 1.0                                          │
│    → This is what HuggingFace returns with output_attentions=True  │
│                                                                     │
│ 6. Attention Output:    [batch, seq, 768]                          │
│    → softmax(QK^T/√d) × V, then projected via c_proj               │
│    → What gets added to residual stream                             │
└─────────────────────────────────────────────────────────────────────┘
```

### MLP Block Extractions

```
┌─────────────────────────────────────────────────────────────────────┐
│ MLP BLOCK EXTRACTIONS (per layer):                                  │
│                                                                     │
│ 7. MLP Input (post ln_2):  [batch, seq, 768]                       │
│    → After second layer norm, before MLP                            │
│                                                                     │
│ 8. MLP Hidden (c_fc output): [batch, seq, 3072]                    │
│    → After first linear layer (4× expansion)                        │
│    → Before activation function                                     │
│                                                                     │
│ 9. MLP Activated:          [batch, seq, 3072]                      │
│    → After GELU activation                                          │
│    → Non-linear transformation applied                              │
│                                                                     │
│ 10. MLP Output (c_proj):   [batch, seq, 768]                       │
│     → After projection back to hidden dimension                     │
│     → What gets added to residual stream                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Residual Stream

```
┌─────────────────────────────────────────────────────────────────────┐
│ RESIDUAL STREAM (per layer):                                        │
│                                                                     │
│ 11. Hidden State:          [batch, seq, 768]                       │
│     → hidden = hidden + attn_out + mlp_out                         │
│     → This is what flows to the next layer                          │
│     → Accumulates information as it passes through                  │
│                                                                     │
│ Total: 13 hidden states                                            │
│   - hidden_states[0]: embedding output (input to layer 0)          │
│   - hidden_states[1]: layer 0 output                               │
│   - hidden_states[2]: layer 1 output                               │
│   - ...                                                            │
│   - hidden_states[12]: layer 11 output (input to ln_f)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Complete Extraction Summary Table

| Stage | Tensor | Shape | What It Tells You |
|-------|--------|-------|-------------------|
| **Tokenization** | `token_ids` | `[seq]` | Input encoding |
| **Tokenization** | `tokens` | `[seq]` | Human-readable tokens |
| **Tokenization** | `attention_mask` | `[seq]` | Valid positions |
| **Tokenization** | `position_ids` | `[seq]` | Position indices |
| **Embedding** | `token_embeddings` | `[seq, 768]` | Semantic representation |
| **Embedding** | `position_embeddings` | `[seq, 768]` | Position encoding |
| **Embedding** | `combined_embeddings` | `[seq, 768]` | hidden_states[0] |
| **Layer N Attn** | `ln_1_output` | `[seq, 768]` | Normalized input to attention |
| **Layer N Attn** | `Q` | `[seq, 12, 64]` | What each position seeks |
| **Layer N Attn** | `K` | `[seq, 12, 64]` | What each position offers |
| **Layer N Attn** | `V` | `[seq, 12, 64]` | What gets transmitted |
| **Layer N Attn** | `raw_attention_scores` | `[12, seq, seq]` | Pre-softmax interest |
| **Layer N Attn** | `attention_weights` | `[12, seq, seq]` | Post-softmax attention |
| **Layer N Attn** | `attention_output` | `[seq, 768]` | Weighted values |
| **Layer N MLP** | `ln_2_output` | `[seq, 768]` | Normalized input to MLP |
| **Layer N MLP** | `mlp_hidden` | `[seq, 3072]` | Feature expansion |
| **Layer N MLP** | `mlp_activated` | `[seq, 3072]` | Post-GELU features |
| **Layer N MLP** | `mlp_output` | `[seq, 768]` | Projected back |
| **Layer N** | `hidden_state` | `[seq, 768]` | Residual stream |
| **Output** | `final_ln` | `[seq, 768]` | Normalized final state |
| **Output** | `logits` | `[seq, 50257]` | All token predictions |
| **Output** | `probabilities` | `[seq, 50257]` | Softmax probabilities |

---

## Memory Cost Analysis

### Per-Inference Memory (100-token sequence, batch=1)

| Data | Calculation | Size |
|------|-------------|------|
| **Tokenization** | 4 × 100 × 4 bytes | <1 KB |
| **Embeddings** | 2 × 100 × 768 × 4 | 0.6 MB |
| **Q, K, V** | 12 layers × 3 × 100 × 12 × 64 × 4 | 11 MB |
| **Raw attention scores** | 12 × 12 × 100 × 100 × 4 | 5.8 MB |
| **Attention weights** | 12 × 12 × 100 × 100 × 4 | 5.8 MB |
| **Attention outputs** | 12 × 100 × 768 × 4 | 3.7 MB |
| **MLP hidden** | 12 × 100 × 3072 × 4 | 14.7 MB |
| **MLP activated** | 12 × 100 × 3072 × 4 | 14.7 MB |
| **MLP outputs** | 12 × 100 × 768 × 4 | 3.7 MB |
| **Hidden states** | 13 × 100 × 768 × 4 | 4 MB |
| **Logits** | 100 × 50257 × 4 | 20 MB |
| **TOTAL** | | **~84 MB** |

### Scaling with Sequence Length

| Seq Length | Attention (O(n²)) | Linear (O(n)) | Total |
|------------|-------------------|---------------|-------|
| 50 tokens | 2.9 MB | 30 MB | ~33 MB |
| 100 tokens | 11.6 MB | 60 MB | ~72 MB |
| 256 tokens | 76 MB | 154 MB | ~230 MB |
| 512 tokens | 302 MB | 307 MB | ~609 MB |
| 1024 tokens | 1.2 GB | 614 MB | ~1.8 GB |

---

## Extraction Tiers

### Tier 1: Standard HuggingFace (Minimal Modification)

```python
# Available with output_attentions=True, output_hidden_states=True
outputs = model(input_ids, output_attentions=True, output_hidden_states=True)

logits = outputs.logits                    # [batch, seq, 50257]
attentions = outputs.attentions            # tuple of [batch, 12, seq, seq]
hidden_states = outputs.hidden_states      # tuple of [batch, seq, 768]
```

### Tier 2: With KV Cache (for generation)

```python
# Available with use_cache=True
outputs = model(input_ids, use_cache=True)

past_key_values = outputs.past_key_values  # tuple of (K, V) per layer
# K: [batch, 12, seq, 64]
# V: [batch, 12, seq, 64]
```

### Tier 3: Full Extraction (Requires Custom Forward)

```python
# Requires modifying forward() or using hooks
# See: scripts/export_gpt2_full.py

Q, K, V                  # Per-layer projections
raw_attention_scores     # Pre-softmax QK^T/√d
mlp_hidden               # Pre-activation MLP
mlp_activated            # Post-GELU MLP
```

---

## Python Hook-Based Extraction

```python
from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
extractions = {}

def make_qkv_hook(layer_idx):
    def hook(module, input, output):
        # output: [batch, seq, 2304]
        batch, seq, _ = output.shape
        q = output[..., :768].view(batch, seq, 12, 64)
        k = output[..., 768:1536].view(batch, seq, 12, 64)
        v = output[..., 1536:].view(batch, seq, 12, 64)
        extractions[f'layer_{layer_idx}_Q'] = q.detach()
        extractions[f'layer_{layer_idx}_K'] = k.detach()
        extractions[f'layer_{layer_idx}_V'] = v.detach()
    return hook

def make_mlp_hook(layer_idx):
    def hook(module, input, output):
        extractions[f'layer_{layer_idx}_mlp_hidden'] = output.detach()
    return hook

# Register hooks
for i in range(12):
    model.transformer.h[i].attn.c_attn.register_forward_hook(make_qkv_hook(i))
    model.transformer.h[i].mlp.c_fc.register_forward_hook(make_mlp_hook(i))

# Run inference
with torch.no_grad():
    outputs = model(input_ids, output_attentions=True, output_hidden_states=True)

# Now extractions dict contains all Q, K, V, and MLP hidden states
```

---

## TypeScript Interface for Full Extraction

```typescript
export interface FullGPT2Extraction {
  // Tokenization
  tokenIds: number[];
  tokens: string[];
  attentionMask: number[];

  // Embeddings
  tokenEmbeddings: Float32Array;      // [seq, 768]
  positionEmbeddings: Float32Array;   // [seq, 768]

  // Per-layer extractions (12 layers)
  layers: LayerExtraction[];

  // Final outputs
  finalLayerNorm: Float32Array;       // [seq, 768]
  logits: Float32Array;               // [seq, 50257]
}

export interface LayerExtraction {
  layerIndex: number;

  // Attention
  query: Float32Array;                // [seq, 12, 64]
  key: Float32Array;                  // [seq, 12, 64]
  value: Float32Array;                // [seq, 12, 64]
  rawAttentionScores: Float32Array;   // [12, seq, seq]
  attentionWeights: Float32Array;     // [12, seq, seq]
  attentionOutput: Float32Array;      // [seq, 768]

  // MLP
  mlpHidden: Float32Array;            // [seq, 3072]
  mlpActivated: Float32Array;         // [seq, 3072]
  mlpOutput: Float32Array;            // [seq, 768]

  // Residual
  hiddenState: Float32Array;          // [seq, 768]
}
```

---

## What Each Extraction Enables

| Extraction | Analysis Capability |
|------------|---------------------|
| `Q, K, V` | Attention head behavior, circuit analysis |
| `raw_attention_scores` | Confidence levels, saturation detection |
| `attention_weights` | Where model "looks", induction head detection |
| `mlp_hidden` | Feature neurons, knowledge storage |
| `mlp_activated` | Active features, sparsity analysis |
| `hidden_states` | Residual stream evolution, logit lens |
| `logits` | Prediction confidence, alternative tokens |

---

## References

- [Transformer Circuits Thread](https://transformer-circuits.pub/)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
