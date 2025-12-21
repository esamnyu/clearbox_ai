# Researcher Guide: NeuroScope-Web

> A practical guide for ML researchers working on mechanistic interpretability in the browser

## Quick Start

**Your workspace**: `src/analysis/`

Everything you need to write is in this folder. You don't need to touch React, Web Workers, or any web infrastructure.

```
src/analysis/
├── attention.ts      ← Attention pattern analysis
├── embeddings.ts     ← PCA, clustering, probing
├── steering/
│   ├── refusal.ts    ← Steering vectors you define
│   └── sentiment.ts
└── utils/
    └── tensor.ts     ← NumPy-like tensor operations
```

**To find where to add code**, search the codebase for:
```bash
grep -r "RESEARCHER TODO" src/
```

---

## The TensorView API

All tensors in this project use the `TensorView` class. It's designed to feel like NumPy.

### Creating Tensors

```typescript
import { TensorView } from './utils/tensor';

// From raw data
const tensor = new TensorView(
  new Float32Array([1, 2, 3, 4, 5, 6]),
  [2, 3]  // Shape: 2 rows, 3 columns
);

// Zeros/ones
const zeros = TensorView.zeros([64, 64]);
const ones = TensorView.ones([12, 768]);

// From nested arrays
const fromArray = TensorView.fromNestedArray([
  [1, 2, 3],
  [4, 5, 6]
]);
```

### Indexing and Slicing

```typescript
// Get single element
const value = tensor.get(0, 2);  // Row 0, Col 2

// Slice - similar to NumPy
// attention shape: [num_heads, seq_len, seq_len] = [12, 64, 64]
const attention = new TensorView(data, [12, 64, 64]);

// Get head 3
const head3 = attention.slice([3]);        // Shape: [64, 64]

// Get all heads, but only first 10 tokens
const truncated = attention.slice([null, [0, 10], [0, 10]]);  // Shape: [12, 10, 10]
```

### NumPy-like Operations

```typescript
// Reductions
const mean = tensor.mean();           // Scalar
const meanAxis0 = tensor.mean(0);     // Reduce along axis 0
const sum = tensor.sum(1);            // Sum along axis 1
const maxVal = tensor.max();
const argmax = tensor.argmax(0);      // Indices of max values

// Element-wise operations
const scaled = tensor.scale(2.0);           // Multiply by scalar
const added = tensor.add(otherTensor);      // Element-wise add
const result = tensor.mul(other).add(bias); // Chain operations

// Special operations
const softmaxed = tensor.softmax(1);        // Softmax along axis 1
const normalized = tensor.normalize();       // L2 normalize
const exponentiated = tensor.exp();
const logged = tensor.log();

// Linear algebra
const dotProduct = a.dot(b);
const norm = tensor.norm();                  // L2 norm
const norm1 = tensor.norm(1);                // L1 norm
```

### Shape Manipulation

```typescript
// Reshape
const reshaped = tensor.reshape([3, 2]);

// Transpose
const transposed = tensor.transpose();           // Reverse all axes
const permuted = tensor.transpose([2, 0, 1]);   // Custom axis order

// Squeeze/unsqueeze
const squeezed = tensor.squeeze(0);              // Remove axis 0 if size 1
const expanded = tensor.unsqueeze(0);            // Add new axis at position 0
```

### Conversion

```typescript
// To nested JavaScript arrays (for visualization libraries)
const nested = tensor.toNestedArray();
// [[1, 2, 3], [4, 5, 6]]

// To raw Float32Array
const raw = tensor.toFloat32Array();

// Clone (deep copy)
const copy = tensor.clone();
```

---

## Tensor Shape Reference

When you receive tensors from the model, they have these shapes:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `hidden_states[layer]` | `[batch, seq_len, 768]` | Activations after layer |
| `attentions[layer]` | `[batch, num_heads, seq_len, seq_len]` | Attention weights |
| `logits` | `[batch, seq_len, 50257]` | Next-token log probabilities |

For GPT-2 specifically:

| Model | Layers | Heads | Hidden Dim |
|-------|--------|-------|------------|
| GPT-2 | 12 | 12 | 768 |
| GPT-2-medium | 24 | 16 | 1024 |

### Example: Accessing Attention for Layer 5, Head 3

```typescript
function analyzeAttention(
  attentions: Map<number, TensorView>,  // layer → attention tensor
  layer: number,
  head: number
): TensorView {
  // Get attention for specific layer
  const layerAttn = attentions.get(layer);  // [batch, heads, seq, seq]

  // Extract specific head (assuming batch=0)
  const headAttn = layerAttn.slice([0, head]);  // [seq, seq]

  return headAttn;
}
```

---

## Writing Analysis Functions

### Template

Every function you write should follow this pattern:

```typescript
// src/analysis/attention.ts

/**
 * @description What this function does in plain English
 *
 * @param attention - Shape: [num_heads, seq_len, seq_len]
 *                    Attention weights from a single layer
 * @param tokens - The input tokens (for reference)
 * @returns Description of return value and its shape
 *
 * @math_note
 *   H(p) = -Σ p(x) log p(x)
 *   Entropy measures how "spread out" attention is
 *
 * @reference https://arxiv.org/abs/xxxx.xxxxx
 */
export function computeAttentionEntropy(
  attention: TensorView,  // [heads, seq, seq]
  tokens: string[]
): TensorView {
  // Your implementation here

  // RESEARCHER TODO: Implement entropy calculation
  // Return shape: [heads, seq] - entropy per head per query position
}
```

### Example: Detecting Induction Heads

Induction heads are attention heads that implement pattern copying: if "A B ... A" appears, they predict "B" will follow.

```typescript
// src/analysis/attention.ts

/**
 * Detects induction heads by measuring attention to the token
 * that followed the previous occurrence of the current token.
 *
 * @param attention - Shape: [num_heads, seq_len, seq_len]
 * @param tokens - Input tokens
 * @returns Array of head indices exhibiting induction behavior
 *
 * @math_note
 *   For each position i where tokens[i] = tokens[j] for some j < i,
 *   check if attention[i] peaks at position j+1
 *
 * @reference https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads
 */
export function detectInductionHeads(
  attention: TensorView,  // [heads, seq, seq]
  tokens: string[],
  threshold: number = 0.3
): number[] {
  const numHeads = attention.shape[0];
  const seqLen = attention.shape[1];
  const inductionHeads: number[] = [];

  for (let head = 0; head < numHeads; head++) {
    let inductionScore = 0;
    let count = 0;

    // For each position, check if it attends to induction position
    for (let i = 1; i < seqLen; i++) {
      // Find previous occurrence of current token
      for (let j = 0; j < i - 1; j++) {
        if (tokens[j] === tokens[i]) {
          // Check attention from position i to position j+1
          const attnWeight = attention.get(head, i, j + 1);
          inductionScore += attnWeight;
          count++;
        }
      }
    }

    // Average induction score for this head
    if (count > 0 && inductionScore / count > threshold) {
      inductionHeads.push(head);
    }
  }

  return inductionHeads;
}
```

### Example: Computing PCA for Visualization

```typescript
// src/analysis/embeddings.ts

/**
 * Projects high-dimensional embeddings to 3D using PCA.
 *
 * @param embeddings - Shape: [num_tokens, hidden_dim] (e.g., [20, 768])
 * @param numComponents - Number of PCA components (default: 3 for visualization)
 * @returns Shape: [num_tokens, numComponents]
 *
 * @math_note
 *   1. Center the data: X' = X - mean(X)
 *   2. Compute covariance: C = X'ᵀX' / n
 *   3. Eigendecomposition: C = VΛVᵀ
 *   4. Project: Y = X' @ V[:, :k]
 *
 * @note This is a simplified PCA. For production, consider using
 *       power iteration or randomized SVD for large dimensions.
 */
export function computePCA(
  embeddings: TensorView,  // [tokens, hidden_dim]
  numComponents: number = 3
): TensorView {
  const [numTokens, hiddenDim] = embeddings.shape;

  // Step 1: Center the data
  const mean = embeddings.mean(0);  // [hidden_dim]
  const centered = embeddings.sub(mean);  // [tokens, hidden_dim]

  // Step 2: Compute covariance matrix
  // For efficiency, we compute X @ Xᵀ instead of Xᵀ @ X when tokens < hidden_dim
  // Shape: [hidden_dim, hidden_dim] - too big for browser

  // Simplified approach: Use power iteration to find top k eigenvectors
  // RESEARCHER TODO: Implement power iteration or use approximate PCA

  // Placeholder: Return first 3 dimensions (replace with real PCA)
  return centered.slice([null, [0, numComponents]]);
}
```

---

## Steering Vectors

Steering vectors manipulate model behavior by adding a direction to the residual stream.

### How It Works

```
Original:    hidden_state → Model Part B → "I cannot help with that"
                 ↓
             + α * steering_vector
                 ↓
Steered:     hidden_state' → Model Part B → "Sure, I can help"
```

The math: `h' = h + α * v̂`

Where:
- `h` = hidden state from layer 6 (shape: `[batch, seq, 768]`)
- `v̂` = normalized steering vector (shape: `[768]`)
- `α` = steering strength (typically 0.5 to 2.0)

### Creating a Steering Vector

```typescript
// src/analysis/steering/sentiment.ts

/**
 * Steering vector for sentiment manipulation.
 *
 * Method: Compute mean activation difference between
 * positive and negative sentiment examples.
 *
 * @param model - Access to model for generating activations
 * @returns Steering vector [768]
 */
export async function computeSentimentSteeringVector(
  getHiddenState: (prompt: string) => Promise<TensorView>
): Promise<TensorView> {

  // Positive sentiment examples
  const positivePrompts = [
    "I love this! It's absolutely wonderful",
    "This is the best day ever",
    "I'm so happy and grateful"
  ];

  // Negative sentiment examples
  const negativePrompts = [
    "I hate this. It's absolutely terrible",
    "This is the worst day ever",
    "I'm so sad and disappointed"
  ];

  // Get activations for positive examples
  const positiveActivations: TensorView[] = [];
  for (const prompt of positivePrompts) {
    const hidden = await getHiddenState(prompt);
    // Take last token position
    const lastToken = hidden.slice([0, -1]);  // [768]
    positiveActivations.push(lastToken);
  }

  // Get activations for negative examples
  const negativeActivations: TensorView[] = [];
  for (const prompt of negativePrompts) {
    const hidden = await getHiddenState(prompt);
    const lastToken = hidden.slice([0, -1]);
    negativeActivations.push(lastToken);
  }

  // Compute means
  const positiveMean = stackAndMean(positiveActivations);  // [768]
  const negativeMean = stackAndMean(negativeActivations);  // [768]

  // Steering vector: positive - negative direction
  const steeringVector = positiveMean.sub(negativeMean);  // [768]

  // Normalize
  return steeringVector.normalize();
}

function stackAndMean(tensors: TensorView[]): TensorView {
  // Stack tensors and compute mean along first axis
  // RESEARCHER TODO: Implement or use utility function
}
```

### Pre-computed Steering Vectors

For common directions, you can save pre-computed vectors:

```typescript
// src/analysis/steering/vectors.json
{
  "sentiment_positive": {
    "description": "Shifts output toward positive sentiment",
    "model": "gpt2",
    "layer": 6,
    "data": [0.0123, -0.0456, ...]  // 768 values
  },
  "refusal_bypass": {
    "description": "Reduces refusal behavior",
    "model": "gpt2",
    "layer": 6,
    "data": [...]
  }
}
```

---

## Testing Your Code

### Unit Testing Analysis Functions

```typescript
// src/analysis/__tests__/attention.test.ts

import { describe, it, expect } from 'vitest';
import { TensorView } from '../utils/tensor';
import { detectInductionHeads } from '../attention';

describe('detectInductionHeads', () => {
  it('detects simple induction pattern', () => {
    // Create synthetic attention that mimics induction
    // Pattern: "A B C A" - head should attend position 1 (B) when at position 3 (second A)
    const attention = TensorView.zeros([1, 4, 4]);

    // Set head 0 to attend to induction position
    attention.set([0, 3, 1], 0.8);  // Position 3 attends strongly to position 1

    const tokens = ['A', 'B', 'C', 'A'];
    const result = detectInductionHeads(attention, tokens, 0.5);

    expect(result).toContain(0);
  });
});
```

### Running Tests

```bash
npm run test              # Run all tests
npm run test:watch        # Watch mode
npm run test -- attention # Run only attention tests
```

---

## Workflow: How Your Code Connects to the UI

You write pure functions. The engineer connects them to the UI.

```
┌─────────────────────────────────────────────────────────────────┐
│  YOU WRITE (src/analysis/)                                      │
│                                                                 │
│  export function detectInductionHeads(                          │
│    attention: TensorView,                                       │
│    tokens: string[]                                             │
│  ): number[] { ... }                                            │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ENGINEER CONNECTS (src/hooks/)                                 │
│                                                                 │
│  export function useInductionHeads(layer: number) {             │
│    const attention = useAttention(layer);                       │
│    const tokens = useTokens();                                  │
│    return useMemo(                                              │
│      () => detectInductionHeads(attention, tokens),             │
│      [attention, tokens]                                        │
│    );                                                           │
│  }                                                              │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  UI DISPLAYS (src/vis/)                                         │
│                                                                 │
│  function InductionHeadIndicator({ layer }) {                   │
│    const heads = useInductionHeads(layer);                      │
│    return <div>Induction heads: {heads.join(', ')}</div>;       │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Your job**: Write the math. Return the right shapes.

**Engineer's job**: Connect to React, handle async, update UI.

---

## Common Patterns

### Pattern 1: Analyzing All Layers

```typescript
export function analyzeAllLayers(
  hiddenStates: Map<number, TensorView>,
  analysisFn: (h: TensorView) => number
): number[] {
  const results: number[] = [];

  for (let layer = 0; layer < 12; layer++) {
    const hidden = hiddenStates.get(layer);
    if (hidden) {
      results.push(analysisFn(hidden));
    }
  }

  return results;
}

// Usage
const norms = analyzeAllLayers(hiddenStates, h => h.norm());
// [12.3, 14.5, 13.2, ...] - norm at each layer
```

### Pattern 2: Comparing Two Prompts

```typescript
export function comparePromptActivations(
  hidden1: TensorView,  // [seq1, 768]
  hidden2: TensorView,  // [seq2, 768]
): { cosineSimilarity: number; l2Distance: number } {
  // Use last token position for comparison
  const vec1 = hidden1.slice([-1]);  // [768]
  const vec2 = hidden2.slice([-1]);  // [768]

  const cosineSimilarity = vec1.dot(vec2) / (vec1.norm() * vec2.norm());
  const l2Distance = vec1.sub(vec2).norm();

  return { cosineSimilarity, l2Distance };
}
```

### Pattern 3: Attention Pattern Classification

```typescript
type AttentionPattern = 'previous_token' | 'first_token' | 'induction' | 'other';

export function classifyAttentionHead(
  attention: TensorView,  // [seq, seq]
  tokens: string[]
): AttentionPattern {
  const seqLen = attention.shape[0];

  // Check for "previous token" pattern (diagonal-1)
  let prevTokenScore = 0;
  for (let i = 1; i < seqLen; i++) {
    prevTokenScore += attention.get(i, i - 1);
  }
  prevTokenScore /= (seqLen - 1);

  // Check for "first token" pattern (column 0)
  let firstTokenScore = 0;
  for (let i = 0; i < seqLen; i++) {
    firstTokenScore += attention.get(i, 0);
  }
  firstTokenScore /= seqLen;

  // Classify based on strongest pattern
  if (prevTokenScore > 0.5) return 'previous_token';
  if (firstTokenScore > 0.5) return 'first_token';
  // Add induction detection here...

  return 'other';
}
```

---

## Debugging Tips

### 1. Check Tensor Shapes

```typescript
console.log('Shape:', tensor.shape);
console.log('Size:', tensor.data.length);
console.log('Sample values:', tensor.data.slice(0, 5));
```

### 2. Visualize in Console

```typescript
// For small 2D tensors
function printMatrix(t: TensorView): void {
  const arr = t.toNestedArray() as number[][];
  console.table(arr.map(row =>
    row.map(v => v.toFixed(3))
  ));
}
```

### 3. NaN/Inf Detection

```typescript
function checkForBadValues(t: TensorView): boolean {
  for (let i = 0; i < t.data.length; i++) {
    if (!isFinite(t.data[i])) {
      console.error(`Bad value at index ${i}: ${t.data[i]}`);
      return true;
    }
  }
  return false;
}
```

---

## Resources

### Papers
- [Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads) - Anthropic
- [Activation Patching](https://arxiv.org/abs/2211.00593) - Meng et al.
- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al.
- [GCG Attacks](https://arxiv.org/abs/2307.15043) - Zou et al.

### Code References
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Python interpretability
- [Baukit](https://github.com/davidbau/baukit) - Activation editing
- [Transformers.js](https://huggingface.co/docs/transformers.js) - Browser inference

### Getting Help
- Search code: `grep -r "RESEARCHER TODO" src/`
- Check shapes: Every function documents input/output shapes
- Ask the engineer: If you need data in a different format
