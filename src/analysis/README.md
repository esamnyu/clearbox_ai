# Analysis Module: Researcher Workspace

This directory contains all mechanistic interpretability analysis code. As a researcher, **this is your primary workspace**.

## Directory Structure

```
src/analysis/
├── README.md                  # This file
├── utils/
│   └── tensor.ts             # NumPy-like tensor operations
├── attention.ts              # Attention pattern analysis
├── embeddings.ts             # Hidden state & embedding analysis
└── steering/
    └── sentiment.ts          # Steering vector definitions
```

## Getting Started

### 1. Understand TensorView

All tensor operations use the `TensorView` class:

```typescript
import { TensorView } from './utils/tensor';

const tensor = new TensorView(new Float32Array([1,2,3,4,5,6]), [2,3]);
console.log(tensor.shape);  // [2, 3]
console.log(tensor.mean());  // 3.5
```

**See**: `docs/RESEARCHER_GUIDE.md` for full API reference.

### 2. Find RESEARCHER TODOs

Search for your tasks:

```bash
grep -r "RESEARCHER TODO" src/analysis/
```

These mark where you should add implementation.

### 3. Write Analysis Functions

Pattern to follow:

```typescript
/**
 * Clear description of what the function does.
 *
 * @param input - Shape: [dim1, dim2] - what it represents
 * @returns Shape: [dim3] - what the output means
 *
 * @math_note Mathematical formula or explanation
 * @reference Link to paper or documentation
 */
export function myAnalysisFunction(
  input: TensorView,
  threshold: number = 0.5
): TensorView {
  // Your implementation here
}
```

### 4. Test Your Code

```typescript
// src/analysis/__tests__/attention.test.ts
import { describe, it, expect } from 'vitest';
import { computeNorm } from '../attention';
import { TensorView } from '../utils/tensor';

describe('computeNorm', () => {
  it('computes L2 norm correctly', () => {
    const tensor = new TensorView(new Float32Array([3, 4]), [2]);
    expect(computeNorm(tensor)).toBe(5);  // sqrt(3^2 + 4^2)
  });
});
```

Run tests:
```bash
npm run test
npm run test:watch  # Auto-rerun on changes
```

## Current TODOs

### High Priority (Session 1-2)
- [ ] `tensor.ts`: Implement `slice()` with range support
- [ ] `tensor.ts`: Implement `sum(axis)` and `mean(axis)`
- [ ] `attention.ts`: Implement `detectInductionHeads()`

### Medium Priority (Session 3-4)
- [ ] `embeddings.ts`: Implement `computePCA()`
- [ ] `attention.ts`: Implement `classifyAttentionHead()`
- [ ] `attention.ts`: Implement `computeAttentionEntropy()`

### Low Priority (Session 5+)
- [ ] `tensor.ts`: Implement `dot()` for matrix multiplication
- [ ] `tensor.ts`: Implement `softmax()`
- [ ] `embeddings.ts`: Implement `findNearestNeighbors()`
- [ ] `steering/sentiment.ts`: Implement `computeSentimentSteering()`

## Integration with UI

Your analysis functions are called by React hooks (in `src/hooks/`). You don't need to touch React code - just export well-typed functions.

Example flow:

```
1. You write:
   export function analyzeLayerNorms(hiddenStates: Map<...>): number[]

2. Engineer creates hook:
   function useLayerNorms() {
     const hiddenStates = useModelStore(state => state.hiddenStates);
     return useMemo(() => analyzeLayerNorms(hiddenStates), [hiddenStates]);
   }

3. UI displays:
   const norms = useLayerNorms();
   norms.map((n, i) => <div>Layer {i}: {n}</div>)
```

## Debugging Tips

### Check Shapes

```typescript
console.log('Shape:', tensor.shape);
console.log('Sample:', tensor.data.slice(0, 10));
```

### Verify No NaN/Inf

```typescript
const hasNaN = tensor.data.some(x => !isFinite(x));
if (hasNaN) console.error('Bad values detected!');
```

### Visual Inspection (2D tensors)

```typescript
function printMatrix(t: TensorView) {
  const arr = t.toNestedArray() as number[][];
  console.table(arr.map(row => row.map(v => v.toFixed(3))));
}
```

## Resources

- **RESEARCHER_GUIDE.md**: Complete API reference and examples
- **COLLABORATION_WORKFLOW.md**: How to work with the engineer
- **ARCHITECTURE.md**: Full system architecture

- **Anthropic Circuits**: https://transformer-circuits.pub
- **TransformerLens**: https://github.com/neelnanda-io/TransformerLens
- **TypeScript Handbook**: https://www.typescriptlang.org/docs/handbook/intro.html

## Getting Help

- **Stuck on TypeScript?** Ask the engineer
- **Stuck on math?** Check the papers in `@reference` tags
- **Stuck on both?** Pair program!
