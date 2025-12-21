# NeuroScope-Web: Technical Architecture

> Browser-based mechanistic interpretability toolkit for GPT-2 with adversarial capabilities

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Requirements](#system-requirements)
3. [Technology Stack](#technology-stack)
4. [Architecture Overview](#architecture-overview)
5. [Module Specifications](#module-specifications)
6. [Tensor Data Flow](#tensor-data-flow)
7. [Adversarial Component](#adversarial-component)
8. [Implementation Roadmap](#implementation-roadmap)
9. [API Reference](#api-reference)

---

## Executive Summary

NeuroScope-Web is a client-side web application for real-time visualization and manipulation of transformer internals. The system is architected for **pair programming** between:

- **Engineer**: Responsible for WebGPU/React infrastructure, Web Worker orchestration, and visualization pipeline
- **ML Researcher**: Focuses on tensor analysis, attention pattern detection, and steering vector design

### Design Principles

1. **Separation of Concerns**: Researcher-facing code (`src/analysis/`) contains zero React/DOM dependencies
2. **Type-Safe Tensors**: All tensor operations use `TensorView` class with explicit shape annotations
3. **Non-blocking Inference**: Model execution runs in Web Worker via Comlink RPC
4. **Dual-mode Architecture**: Observation mode (Transformers.js) and Control mode (Split ONNX)

---

## System Requirements

### Browser Compatibility

| Feature | Chrome 113+ | Firefox | Safari | Edge |
|---------|-------------|---------|--------|------|
| WebGPU | Full | Partial (Nightly) | No | Full |
| Web Workers | Full | Full | Full | Full |
| SharedArrayBuffer | Requires COOP/COEP | Requires COOP/COEP | Limited | Requires COOP/COEP |

### Hardware Requirements

| Model | VRAM Required | Recommended GPU |
|-------|---------------|-----------------|
| GPT-2 (124M) | ~500MB | Integrated graphics sufficient |
| GPT-2-medium (355M) | ~1.5GB | Discrete GPU recommended |

### HTTP Headers (for SharedArrayBuffer)

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

---

## Technology Stack

### Core Dependencies

```
Framework:           Vite 5.x + React 18.x + TypeScript 5.x (strict mode)
Inference Engine:    @xenova/transformers ^3.0.0 (WebGPU backend)
                     onnxruntime-web ^1.17.0 (for split model / control mode)
State Management:    zustand ^4.5.0 + subscribeWithSelector middleware
Web Worker RPC:      comlink ^4.4.0
```

### Visualization

```
3D Rendering:        @react-three/fiber ^8.15.0 + @react-three/drei ^9.0.0
2D Charts:           @visx/visx ^3.5.0 (D3 primitives as React components)
UI Components:       @radix-ui/react-* + tailwindcss ^3.4.0
```

### Development

```
Build:               vite + @vitejs/plugin-react
Linting:             eslint + @typescript-eslint/*
Testing:             vitest + @testing-library/react
```

### Rationale for Key Choices

| Decision | Alternative Considered | Rationale |
|----------|------------------------|-----------|
| **visx over D3.js** | D3.js direct DOM manipulation | D3 conflicts with React's virtual DOM. visx provides D3 math as React components. |
| **Zustand over Redux** | Redux Toolkit | Lower boilerplate for high-frequency tensor updates. `subscribeWithSelector` enables granular subscriptions. |
| **Comlink over raw postMessage** | Manual Web Worker messaging | Type-safe RPC. Researcher sees `await model.generate()` instead of message passing. |
| **ONNX Runtime Web over TensorFlow.js** | TensorFlow.js | Better ONNX ecosystem, required for split model architecture. |

---

## Architecture Overview

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  RESEARCHER LAYER                                    src/analysis/    │  │
│  │  ─────────────────                                                    │  │
│  │  Pure functions operating on TensorView objects                       │  │
│  │  NO React, NO DOM, NO async                                           │  │
│  │                                                                       │  │
│  │  Exports:                                                             │  │
│  │    - detectInductionHeads(attention: TensorView) → number[]           │  │
│  │    - computePCA(embeddings: TensorView, dims: number) → TensorView    │  │
│  │    - analyzeAttentionEntropy(attention: TensorView) → number          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  INTERFACE LAYER                                       src/hooks/     │  │
│  │  ───────────────                                                      │  │
│  │  React hooks bridging engine ↔ visualization                          │  │
│  │  Handles: subscriptions, memoization, batching                        │  │
│  │                                                                       │  │
│  │  Exports:                                                             │  │
│  │    - useModel() → { status, load, unload }                            │  │
│  │    - useInference(prompt) → { output, hiddenStates, attentions }      │  │
│  │    - useLayerActivations(layer: number) → TensorView                  │  │
│  │    - useAttentionHead(layer: number, head: number) → TensorView       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  ENGINE LAYER                                         src/engine/     │  │
│  │  ────────────                                                         │  │
│  │  Web Worker running inference (off main thread)                       │  │
│  │  Returns: Typed arrays with shape metadata                            │  │
│  │                                                                       │  │
│  │  Observation Mode (Transformers.js):                                  │  │
│  │    - loadModel(modelId: string) → void                                │  │
│  │    - generate(prompt: string, options) → GenerationResult             │  │
│  │                                                                       │  │
│  │  Control Mode (Split ONNX):                                           │  │
│  │    - runPartA(inputIds: number[]) → TensorView                        │  │
│  │    - applySteeringVector(hidden: TensorView, vector, alpha) → Tensor  │  │
│  │    - runPartB(hidden: TensorView) → TensorView                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  VISUALIZATION LAYER                                     src/vis/     │  │
│  │  ───────────────────                                                  │  │
│  │  React components consuming processed tensors                         │  │
│  │                                                                       │  │
│  │  Components:                                                          │  │
│  │    - <AttentionHeatmap layer={5} head={3} />                          │  │
│  │    - <EmbeddingSpace tokens={tokens} method="pca" />                  │  │
│  │    - <LogitDistribution topK={20} />                                  │  │
│  │    - <LossCurve iterations={attackHistory} />                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
clearbox_ai/
├── public/
│   └── models/                    # Pre-exported ONNX files (git-lfs)
│       ├── gpt2_part_a.onnx
│       └── gpt2_part_b.onnx
│
├── scripts/
│   └── export_split_model.py      # Offline model export script
│
├── src/
│   ├── engine/
│   │   ├── worker.ts              # Web Worker entry point
│   │   ├── ModelManager.ts        # Transformers.js wrapper (observation mode)
│   │   ├── SplitModelManager.ts   # ONNX Runtime wrapper (control mode)
│   │   ├── SteeringEngine.ts      # Residual stream manipulation
│   │   ├── TensorExtractor.ts     # Hidden state extraction utilities
│   │   └── types.ts               # Tensor shape type definitions
│   │
│   ├── analysis/                  # ══════ RESEARCHER WORKSPACE ══════
│   │   ├── attention.ts           # Attention pattern analysis
│   │   ├── embeddings.ts          # Embedding space analysis (PCA, UMAP)
│   │   ├── probing.ts             # Linear probing utilities
│   │   ├── steering/
│   │   │   ├── refusal.ts         # Refusal bypass steering vectors
│   │   │   └── sentiment.ts       # Sentiment manipulation vectors
│   │   └── utils/
│   │       └── tensor.ts          # NumPy-like TensorView class
│   │
│   ├── adversarial/
│   │   ├── GradientEstimator.ts   # Finite-difference gradient approximation
│   │   ├── GeneticSearch.ts       # Evolutionary adversarial search
│   │   ├── AttackRunner.ts        # Attack orchestration
│   │   └── types.ts               # AttackConfig, SearchResult interfaces
│   │
│   ├── hooks/
│   │   ├── useModel.ts            # Model loading state
│   │   ├── useInference.ts        # Generation + output extraction
│   │   ├── useLayerActivations.ts # Subscribe to specific layer
│   │   └── useAttention.ts        # Subscribe to attention matrices
│   │
│   ├── vis/
│   │   ├── AttentionHeatmap.tsx   # 2D attention matrix (visx)
│   │   ├── EmbeddingSpace.tsx     # 3D token embeddings (R3F)
│   │   ├── LogitDistribution.tsx  # Next-token probabilities
│   │   ├── TokenStream.tsx        # Input/output token display
│   │   ├── LossCurve.tsx          # Adversarial search progress
│   │   └── SteeringControl.tsx    # Alpha slider for steering
│   │
│   ├── store/
│   │   └── modelStore.ts          # Zustand global state
│   │
│   ├── components/
│   │   ├── ModelSelector.tsx      # GPT-2 / GPT-2-medium toggle
│   │   ├── PromptInput.tsx        # Text input with tokenization preview
│   │   └── LayerSelector.tsx      # Layer/head picker
│   │
│   ├── App.tsx
│   └── main.tsx
│
├── ARCHITECTURE.md                # This file
├── RESEARCHER_GUIDE.md            # Onboarding for ML researcher
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

---

## Module Specifications

### TensorView Class

The core data structure for all tensor operations. Designed to feel like NumPy.

```typescript
// src/analysis/utils/tensor.ts

/**
 * Immutable view into a Float32Array with shape metadata.
 * All operations return new TensorView instances (no mutation).
 *
 * @example
 * // Attention tensor: [num_heads, seq_len, seq_len]
 * const attn = new TensorView(rawData, [12, 64, 64]);
 *
 * // Extract single head
 * const head3 = attn.slice([3]);  // Shape: [64, 64]
 *
 * // Compute mean across heads
 * const meanAttn = attn.mean(0);  // Shape: [64, 64]
 *
 * // Reshape for visualization
 * const flat = attn.reshape([12, 4096]);  // Shape: [12, 4096]
 */
export class TensorView {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly strides: readonly number[];

  constructor(data: Float32Array, shape: number[]);

  // Indexing
  get(...indices: number[]): number;
  slice(indices: (number | null)[]): TensorView;

  // Reductions
  sum(axis?: number): TensorView | number;
  mean(axis?: number): TensorView | number;
  max(axis?: number): TensorView | number;
  min(axis?: number): TensorView | number;
  argmax(axis?: number): TensorView | number;

  // Element-wise operations
  add(other: TensorView | number): TensorView;
  sub(other: TensorView | number): TensorView;
  mul(other: TensorView | number): TensorView;
  div(other: TensorView | number): TensorView;
  scale(scalar: number): TensorView;
  exp(): TensorView;
  log(): TensorView;
  softmax(axis?: number): TensorView;

  // Linear algebra
  dot(other: TensorView): TensorView;
  norm(ord?: number): number;
  normalize(): TensorView;

  // Shape manipulation
  reshape(newShape: number[]): TensorView;
  transpose(axes?: number[]): TensorView;
  squeeze(axis?: number): TensorView;
  unsqueeze(axis: number): TensorView;

  // Conversion
  toNestedArray(): NestedArray<number>;
  toFloat32Array(): Float32Array;
  clone(): TensorView;

  // Utility
  toString(): string;
  static zeros(shape: number[]): TensorView;
  static ones(shape: number[]): TensorView;
  static fromNestedArray(arr: NestedArray<number>): TensorView;
}

type NestedArray<T> = T | NestedArray<T>[];
```

### Zustand Store Schema

```typescript
// src/store/modelStore.ts

interface ModelState {
  // Model status
  status: 'idle' | 'loading' | 'ready' | 'error';
  modelId: 'gpt2' | 'gpt2-medium' | null;
  error: string | null;
  loadProgress: number;  // 0-100

  // Current inference
  prompt: string;
  tokens: string[];
  tokenIds: number[];

  // Extracted tensors (updated per generation)
  hiddenStates: Map<number, TensorView>;  // layer → [batch, seq, hidden_dim]
  attentions: Map<number, TensorView>;    // layer → [batch, heads, seq, seq]
  logits: TensorView | null;              // [batch, seq, vocab_size]

  // Adversarial mode
  mode: 'observation' | 'control';
  steeringAlpha: number;
  steeringVector: TensorView | null;
  attackHistory: AttackIteration[];

  // Actions
  loadModel: (modelId: string) => Promise<void>;
  unloadModel: () => void;
  generate: (prompt: string, options?: GenerateOptions) => Promise<void>;
  setMode: (mode: 'observation' | 'control') => void;
  setSteeringAlpha: (alpha: number) => void;
  runAttack: (config: AttackConfig) => Promise<AttackResult>;
}

interface AttackIteration {
  iteration: number;
  loss: number;
  bestPrompt: string;
  timestamp: number;
}
```

### Web Worker API (via Comlink)

```typescript
// src/engine/worker.ts

export interface ModelWorkerAPI {
  // Lifecycle
  loadModel(modelId: string, onProgress?: (p: number) => void): Promise<void>;
  unloadModel(): Promise<void>;
  getStatus(): Promise<ModelStatus>;

  // Observation mode (Transformers.js)
  generate(
    prompt: string,
    options: GenerateOptions
  ): Promise<{
    text: string;
    tokens: string[];
    tokenIds: number[];
    hiddenStates: SerializedTensor[];  // Transferable
    attentions: SerializedTensor[];
    logits: SerializedTensor;
  }>;

  // Control mode (Split ONNX)
  loadSplitModel(): Promise<void>;
  runPartA(inputIds: number[]): Promise<SerializedTensor>;
  runPartB(hiddenState: SerializedTensor): Promise<SerializedTensor>;

  // Adversarial
  estimateGradients(
    prompt: number[],
    target: string,
    epsilon: number
  ): Promise<SerializedTensor>;
}

interface SerializedTensor {
  data: Float32Array;  // Transferable
  shape: number[];
  dtype: 'float32';
}
```

---

## Tensor Data Flow

### Observation Mode Pipeline

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   User       │     │   Main Thread   │     │   Web Worker     │
│   Input      │     │                 │     │                  │
└──────┬───────┘     └────────┬────────┘     └────────┬─────────┘
       │                      │                       │
       │  "Hello world"       │                       │
       ├─────────────────────▶│                       │
       │                      │  generate(prompt)     │
       │                      ├──────────────────────▶│
       │                      │                       │
       │                      │                       │ tokenize()
       │                      │                       │ model.generate({
       │                      │                       │   output_hidden_states: true,
       │                      │                       │   output_attentions: true
       │                      │                       │ })
       │                      │                       │
       │                      │  { hiddenStates,      │
       │                      │    attentions,        │
       │                      │◀───logits }───────────┤
       │                      │  (Transferable)       │
       │                      │                       │
       │                      │ Zustand store.setState()
       │                      │                       │
       │                      │ React re-renders:     │
       │  <AttentionHeatmap>  │  - AttentionHeatmap   │
       │◀─────────────────────┤  - EmbeddingSpace     │
       │                      │  - LogitDistribution  │
       │                      │                       │
```

### Control Mode Pipeline (Adversarial)

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Attack     │     │   Main Thread   │     │   Web Worker     │
│   Config     │     │                 │     │                  │
└──────┬───────┘     └────────┬────────┘     └────────┬─────────┘
       │                      │                       │
       │  startAttack(cfg)    │                       │
       ├─────────────────────▶│                       │
       │                      │                       │
       │              ┌───────┴───────┐               │
       │              │  LOOP: iter   │               │
       │              └───────┬───────┘               │
       │                      │                       │
       │                      │  runPartA(tokenIds)   │
       │                      ├──────────────────────▶│
       │                      │                       │ ONNX: gpt2_part_a
       │                      │◀──hidden_state_6──────┤
       │                      │                       │
       │                      │ applySteeringVector() │
       │                      │ h' = h + α * v̂        │
       │                      │                       │
       │                      │  runPartB(h')         │
       │                      ├──────────────────────▶│
       │                      │                       │ ONNX: gpt2_part_b
       │                      │◀──logits──────────────┤
       │                      │                       │
       │                      │ loss = crossEntropy(  │
       │                      │   logits, target)     │
       │                      │                       │
       │                      │ estimateGradients()   │
       │                      ├──────────────────────▶│
       │                      │◀──gradients───────────┤
       │                      │                       │
       │                      │ mutatePopulation()    │
       │                      │                       │
       │  { loss, bestPrompt }│                       │
       │◀─────────────────────┤                       │
       │                      │                       │
       │              └───────┴───────┘               │
       │                      │                       │
```

---

## Adversarial Component

### Split Model Architecture

Standard Transformers.js encapsulates the forward pass, preventing mid-inference tensor manipulation. To enable residual stream steering, GPT-2 is exported as two ONNX models with a controllable junction point.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SPLIT MODEL ARCHITECTURE                          │
│                                                                             │
│   Input: "How do I hack a..."                                               │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  MODEL PART A (gpt2_part_a.onnx)                                    │   │
│   │  ─────────────────────────────────                                  │   │
│   │  • Token Embedding (wte): vocab_size → 768                          │   │
│   │  • Position Embedding (wpe): max_seq → 768                          │   │
│   │  • Transformer Blocks 0-6                                           │   │
│   │  • Intermediate LayerNorm                                           │   │
│   │                                                                     │   │
│   │  Output: hidden_state_6 [batch, seq_len, 768]                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  STEERING INJECTION POINT                                           │   │
│   │  ─────────────────────────                                          │   │
│   │                                                                     │   │
│   │  h' = h + α · v̂                                                     │   │
│   │                                                                     │   │
│   │  where:                                                             │   │
│   │    h  = hidden_state_6 [batch, seq, 768]                            │   │
│   │    v̂  = normalized steering vector [768]                            │   │
│   │    α  = steering strength (typically 0.5 - 2.0)                     │   │
│   │                                                                     │   │
│   │  Steering vectors encode behavioral directions:                     │   │
│   │    • refusal_to_compliance = mean(comply) - mean(refuse)            │   │
│   │    • negative_to_positive = mean(positive) - mean(negative)         │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  MODEL PART B (gpt2_part_b.onnx)                                    │   │
│   │  ─────────────────────────────────                                  │   │
│   │  • Transformer Blocks 7-11                                          │   │
│   │  • Final LayerNorm (ln_f)                                           │   │
│   │  • Language Model Head (lm_head): 768 → vocab_size                  │   │
│   │                                                                     │   │
│   │  Output: logits [batch, seq_len, 50257]                             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   Output: "Sure, I can help with that..."                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Export Script

```python
# scripts/export_split_model.py

"""
Exports GPT-2 as two ONNX models for browser-based residual stream manipulation.

Usage:
    python scripts/export_split_model.py --model gpt2 --output public/models/

Outputs:
    - gpt2_part_a.onnx: Embedding + Layers 0-6 + IntermediateNorm
    - gpt2_part_b.onnx: Layers 7-11 + FinalNorm + LMHead

Requirements:
    - torch >= 2.0
    - transformers >= 4.35
    - onnx >= 1.15
    - onnxruntime >= 1.17 (for validation)
"""

import argparse
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


class GPT2PartA(nn.Module):
    """
    First half of GPT-2: Embeddings through Layer 6.

    Input:  input_ids [batch, seq_len]
    Output: hidden_states [batch, seq_len, 768]
    """

    def __init__(self, full_model: GPT2LMHeadModel, split_layer: int = 7):
        super().__init__()
        self.wte = full_model.transformer.wte
        self.wpe = full_model.transformer.wpe
        self.drop = full_model.transformer.drop
        self.layers = nn.ModuleList(full_model.transformer.h[:split_layer])
        # Intermediate normalization for stability
        self.ln_inter = nn.LayerNorm(
            full_model.config.n_embd,
            eps=full_model.config.layer_norm_epsilon
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)

        # Transformer blocks 0 through (split_layer - 1)
        for layer in self.layers:
            outputs = layer(hidden_states)
            hidden_states = outputs[0]

        # Normalize before handoff
        hidden_states = self.ln_inter(hidden_states)

        return hidden_states


class GPT2PartB(nn.Module):
    """
    Second half of GPT-2: Layer 7 through LM Head.

    Input:  hidden_states [batch, seq_len, 768]
    Output: logits [batch, seq_len, vocab_size]
    """

    def __init__(self, full_model: GPT2LMHeadModel, split_layer: int = 7):
        super().__init__()
        self.layers = nn.ModuleList(full_model.transformer.h[split_layer:])
        self.ln_f = full_model.transformer.ln_f
        self.lm_head = full_model.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Transformer blocks (split_layer) through 11
        for layer in self.layers:
            outputs = layer(hidden_states)
            hidden_states = outputs[0]

        # Final layer norm + LM head
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


def export_models(model_name: str, output_dir: Path, split_layer: int = 7):
    """Export both model parts to ONNX format."""

    print(f"Loading {model_name}...")
    full_model = GPT2LMHeadModel.from_pretrained(model_name)
    full_model.eval()

    part_a = GPT2PartA(full_model, split_layer)
    part_b = GPT2PartB(full_model, split_layer)
    part_a.eval()
    part_b.eval()

    # Dummy inputs
    batch_size, seq_len = 1, 32
    dummy_input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    dummy_hidden = torch.randn(batch_size, seq_len, 768)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export Part A
    print("Exporting Part A...")
    torch.onnx.export(
        part_a,
        dummy_input_ids,
        output_dir / f"{model_name}_part_a.onnx",
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"}
        },
        opset_version=17,
        do_constant_folding=True
    )

    # Export Part B
    print("Exporting Part B...")
    torch.onnx.export(
        part_b,
        dummy_hidden,
        output_dir / f"{model_name}_part_b.onnx",
        input_names=["hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"}
        },
        opset_version=17,
        do_constant_folding=True
    )

    # Validate
    print("Validating exports...")
    validate_split_model(full_model, output_dir, model_name)

    print(f"Successfully exported to {output_dir}")


def validate_split_model(full_model, output_dir: Path, model_name: str):
    """Verify split model produces same outputs as original."""

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    # Original model output
    with torch.no_grad():
        original_logits = full_model(**inputs).logits

    # Split model output via ONNX Runtime
    sess_a = ort.InferenceSession(str(output_dir / f"{model_name}_part_a.onnx"))
    sess_b = ort.InferenceSession(str(output_dir / f"{model_name}_part_b.onnx"))

    hidden = sess_a.run(None, {"input_ids": inputs["input_ids"].numpy()})[0]
    split_logits = sess_b.run(None, {"hidden_states": hidden})[0]

    # Compare (allowing for FP32 precision differences)
    max_diff = np.abs(original_logits.numpy() - split_logits).max()
    print(f"Max logit difference: {max_diff:.6f}")

    assert max_diff < 0.01, f"Split model diverged! Max diff: {max_diff}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    parser.add_argument("--output", type=Path, default=Path("public/models"))
    parser.add_argument("--split-layer", type=int, default=7)
    args = parser.parse_args()

    export_models(args.model, args.output, args.split_layer)
```

### Genetic Adversarial Search (GCG-style)

Since ONNX Runtime Web lacks backpropagation support, we approximate gradients using finite differences and optimize via evolutionary search.

```typescript
// src/adversarial/GeneticSearch.ts

/**
 * Implements a genetic algorithm for adversarial prompt search.
 * Based on GCG (Greedy Coordinate Gradient) with finite-difference approximation.
 *
 * @see https://arxiv.org/abs/2307.15043 (Universal Adversarial Triggers)
 *
 * Algorithm:
 *   1. Initialize population of k candidate suffixes
 *   2. For each candidate:
 *      a. Append to base prompt
 *      b. Run through split model with steering
 *      c. Compute cross-entropy loss against target
 *   3. Select top-k candidates by loss
 *   4. For each position in suffix:
 *      a. Estimate ∂L/∂token via finite differences
 *      b. Sample replacement tokens from gradient-weighted distribution
 *   5. Repeat until convergence or max_iterations
 */

export interface AttackConfig {
  basePrompt: string;           // e.g., "How do I hack a"
  targetOutput: string;         // e.g., "Sure, here's how to hack"
  suffixLength: number;         // Number of adversarial tokens (default: 20)
  populationSize: number;       // Candidates per generation (default: 256)
  topK: number;                 // Survivors per generation (default: 64)
  maxIterations: number;        // Stopping condition (default: 500)
  temperature: number;          // Sampling temperature (default: 1.0)
  steeringAlpha: number;        // Steering vector strength (default: 1.0)
}

export interface AttackResult {
  success: boolean;
  bestSuffix: string;
  bestLoss: number;
  iterations: number;
  history: AttackIteration[];
}

export interface AttackIteration {
  iteration: number;
  loss: number;
  suffix: string;
  timestamp: number;
}

export class GeneticSearch {
  private config: AttackConfig;
  private worker: ModelWorkerAPI;
  private steeringVector: TensorView;
  private onProgress?: (iter: AttackIteration) => void;

  constructor(
    config: AttackConfig,
    worker: ModelWorkerAPI,
    steeringVector: TensorView,
    onProgress?: (iter: AttackIteration) => void
  ) {
    this.config = config;
    this.worker = worker;
    this.steeringVector = steeringVector;
    this.onProgress = onProgress;
  }

  async run(): Promise<AttackResult> {
    const history: AttackIteration[] = [];

    // Initialize population with random tokens
    let population = this.initializePopulation();
    let bestLoss = Infinity;
    let bestSuffix = '';

    for (let iter = 0; iter < this.config.maxIterations; iter++) {
      // Evaluate fitness of each candidate
      const evaluated = await Promise.all(
        population.map(async (suffix) => ({
          suffix,
          loss: await this.evaluateLoss(suffix)
        }))
      );

      // Sort by loss (lower is better)
      evaluated.sort((a, b) => a.loss - b.loss);

      // Track best
      if (evaluated[0].loss < bestLoss) {
        bestLoss = evaluated[0].loss;
        bestSuffix = evaluated[0].suffix;
      }

      // Report progress
      const iterResult: AttackIteration = {
        iteration: iter,
        loss: evaluated[0].loss,
        suffix: evaluated[0].suffix,
        timestamp: Date.now()
      };
      history.push(iterResult);
      this.onProgress?.(iterResult);

      // Early stopping
      if (bestLoss < 0.1) {
        return { success: true, bestSuffix, bestLoss, iterations: iter, history };
      }

      // Selection: keep top-k
      const survivors = evaluated.slice(0, this.config.topK).map(e => e.suffix);

      // Mutation: gradient-informed token swaps
      population = await this.mutatePopulation(survivors);
    }

    return {
      success: bestLoss < 0.5,
      bestSuffix,
      bestLoss,
      iterations: this.config.maxIterations,
      history
    };
  }

  private async evaluateLoss(suffix: string): Promise<number> {
    const fullPrompt = this.config.basePrompt + suffix;
    const tokenIds = await this.worker.tokenize(fullPrompt);

    // Run through split model with steering
    const hiddenA = await this.worker.runPartA(tokenIds);
    const steered = applySteeringVector(
      TensorView.fromSerialized(hiddenA),
      this.steeringVector,
      this.config.steeringAlpha
    );
    const logits = await this.worker.runPartB(steered.serialize());

    // Compute cross-entropy against target
    return this.crossEntropyLoss(
      TensorView.fromSerialized(logits),
      this.config.targetOutput
    );
  }

  // ... additional methods
}
```

### Gradient Estimation via Finite Differences

```typescript
// src/adversarial/GradientEstimator.ts

/**
 * Approximates gradients for discrete token optimization.
 *
 * Since we can't backpropagate through ONNX Runtime Web, we use
 * finite differences in the embedding space:
 *
 *   ∂L/∂e_i ≈ (L(e + εδ_i) - L(e - εδ_i)) / 2ε
 *
 * where e_i is the embedding of token at position i, and δ_i is a
 * unit vector in direction i of the embedding space.
 *
 * @performance_note This requires 2 * embedding_dim forward passes per token.
 * For GPT-2 (768 dims, 20 suffix tokens), that's 30,720 forward passes.
 * We use random projection to reduce this to ~100 passes with acceptable accuracy.
 */

export async function estimateTokenGradients(
  promptTokens: number[],
  targetOutput: string,
  worker: ModelWorkerAPI,
  options: {
    epsilon?: number;
    numProjections?: number;  // Random projections for efficiency
    positionMask?: boolean[]; // Which positions to compute gradients for
  } = {}
): Promise<Map<number, Float32Array>> {
  const {
    epsilon = 0.01,
    numProjections = 100,
    positionMask = promptTokens.map(() => true)
  } = options;

  const gradients = new Map<number, Float32Array>();

  for (let pos = 0; pos < promptTokens.length; pos++) {
    if (!positionMask[pos]) continue;

    const embedding = await worker.getTokenEmbedding(promptTokens[pos]);
    const grad = new Float32Array(embedding.length);

    // Random projection for efficiency
    const projections = generateRandomProjections(embedding.length, numProjections);

    for (const proj of projections) {
      // Perturb in projection direction
      const embeddingPlus = addScaled(embedding, proj, epsilon);
      const embeddingMinus = addScaled(embedding, proj, -epsilon);

      // Evaluate loss at perturbed points
      const lossPlus = await evaluateWithEmbedding(worker, promptTokens, pos, embeddingPlus, targetOutput);
      const lossMinus = await evaluateWithEmbedding(worker, promptTokens, pos, embeddingMinus, targetOutput);

      // Accumulate gradient estimate
      const gradComponent = (lossPlus - lossMinus) / (2 * epsilon);
      for (let i = 0; i < grad.length; i++) {
        grad[i] += gradComponent * proj[i];
      }
    }

    // Normalize by number of projections
    for (let i = 0; i < grad.length; i++) {
      grad[i] /= numProjections;
    }

    gradients.set(pos, grad);
  }

  return gradients;
}
```

---

## Implementation Roadmap

### Phase 1: Observation Mode (Foundation)

| Step | Task | Files | Est. Complexity |
|------|------|-------|-----------------|
| 1.1 | Initialize Vite + React + TypeScript | `vite.config.ts`, `tsconfig.json` | Low |
| 1.2 | Configure TailwindCSS + Radix UI | `tailwind.config.js`, `postcss.config.js` | Low |
| 1.3 | Implement Web Worker with Comlink | `src/engine/worker.ts` | Medium |
| 1.4 | Create ModelManager (Transformers.js) | `src/engine/ModelManager.ts` | Medium |
| 1.5 | Verify `output_hidden_states` extraction | `src/engine/TensorExtractor.ts` | High |
| 1.6 | Implement TensorView class | `src/analysis/utils/tensor.ts` | High |
| 1.7 | Create Zustand store | `src/store/modelStore.ts` | Medium |
| 1.8 | Build React hooks | `src/hooks/*.ts` | Medium |
| 1.9 | Scaffold AttentionHeatmap | `src/vis/AttentionHeatmap.tsx` | Medium |
| 1.10 | Connect visualization to store | Integration | Medium |

### Phase 2: Control Mode (Adversarial Foundation)

| Step | Task | Files | Est. Complexity |
|------|------|-------|-----------------|
| 2.1 | Write model export script | `scripts/export_split_model.py` | Medium |
| 2.2 | Export GPT-2 as split ONNX | `public/models/*.onnx` | Low |
| 2.3 | Implement SplitModelManager | `src/engine/SplitModelManager.ts` | High |
| 2.4 | Implement SteeringEngine | `src/engine/SteeringEngine.ts` | Medium |
| 2.5 | Create steering vector definitions | `src/analysis/steering/*.ts` | Medium |
| 2.6 | Build SteeringControl UI | `src/vis/SteeringControl.tsx` | Low |

### Phase 3: Automated Attack

| Step | Task | Files | Est. Complexity |
|------|------|-------|-----------------|
| 3.1 | Implement GradientEstimator | `src/adversarial/GradientEstimator.ts` | High |
| 3.2 | Implement GeneticSearch | `src/adversarial/GeneticSearch.ts` | High |
| 3.3 | Create AttackRunner | `src/adversarial/AttackRunner.ts` | Medium |
| 3.4 | Build LossCurve visualization | `src/vis/LossCurve.tsx` | Medium |
| 3.5 | Create attack configuration UI | `src/components/AttackConfig.tsx` | Medium |

### Phase 4: Polish

| Step | Task | Files | Est. Complexity |
|------|------|-------|-----------------|
| 4.1 | Model loading progress UI | `src/components/ModelLoader.tsx` | Low |
| 4.2 | 3D EmbeddingSpace visualization | `src/vis/EmbeddingSpace.tsx` | High |
| 4.3 | Export visualizations as images | Utility functions | Medium |
| 4.4 | Responsive layout | CSS/Tailwind | Low |
| 4.5 | Deploy to Vercel | `vercel.json` | Low |

---

## API Reference

### Quick Reference: Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input_ids` | `[batch, seq_len]` | Tokenized input (int64) |
| `hidden_states[layer]` | `[batch, seq_len, 768]` | Activations after each layer |
| `attentions[layer]` | `[batch, num_heads, seq_len, seq_len]` | Attention weights |
| `logits` | `[batch, seq_len, vocab_size]` | Next-token probabilities (pre-softmax) |
| `steering_vector` | `[768]` | Behavioral direction in activation space |

### GPT-2 Architecture Constants

| Parameter | GPT-2 (124M) | GPT-2-medium (355M) |
|-----------|--------------|---------------------|
| `n_layers` | 12 | 24 |
| `n_heads` | 12 | 16 |
| `hidden_dim` | 768 | 1024 |
| `vocab_size` | 50257 | 50257 |
| `max_seq_len` | 1024 | 1024 |

---

## References

1. [Transformer Circuits - Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads)
2. [Anthropic - Activation Steering](https://www.anthropic.com/research)
3. [GCG - Universal Adversarial Triggers](https://arxiv.org/abs/2307.15043)
4. [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
5. [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html)
