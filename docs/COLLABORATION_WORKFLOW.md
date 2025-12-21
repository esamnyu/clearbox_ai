# Collaboration Workflow: Engineer + ML Researcher

> Maximizing learning and impact through structured pair programming

## Overview

This document describes the optimal collaboration strategy for NeuroScope-Web development with a two-person team:
- **Engineer**: Web development expertise, learning interpretability
- **ML Researcher**: Interpretability expertise, learning web development

---

## The "Ping-Pong" Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  Session Structure (2-3 hours each)                             │
│                                                                 │
│  Engineer leads:   Infrastructure sessions (odd weeks)          │
│  Researcher leads: Analysis sessions (even weeks)               │
│                                                                 │
│  Both present:     Integration + checkpoint tests               │
└─────────────────────────────────────────────────────────────────┘
```

**Core Principle**: Each session has a clear leader based on their expertise, but both are actively engaged.

---

## Phase 1: Foundation (Weeks 1-2)

### Session 1: **Engineer Leads** (Infrastructure)

**Goal**: TensorView class + directory structure

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Engineer** | Implement `TensorView` class (NumPy-like API) | Tensor operations, shape broadcasting |
| **Researcher** | Pair program, explain what operations are needed | TypeScript basics, how to structure reusable code |

**Output**: `src/analysis/utils/tensor.ts` with tests

**Checkpoint Test**:
```typescript
const tensor = new TensorView(new Float32Array([1,2,3,4]), [2,2]);
console.log(tensor.mean());  // 2.5
console.log(tensor.slice([0]).shape);  // [2]
```

---

### Session 2: **Researcher Leads** (First Analysis)

**Goal**: Write first analysis function using TensorView

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Researcher** | Write `computeNorm()` function in `src/analysis/attention.ts` | How TensorView API works in practice |
| **Engineer** | Help debug TypeScript errors, explain async patterns | What norms mean, why they matter |

**Output**: First RESEARCHER TODO completed

**Checkpoint Test**:
```typescript
const hidden = new TensorView(data, [12, 768]);
const norm = computeNorm(hidden);
console.log(norm);  // Expected: ~15.2 for GPT-2
```

---

### Session 3: **Engineer Leads** (Hidden State Extraction)

**Goal**: Extract hidden states from transformers.js

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Engineer** | Modify `worker.ts` to extract `output_hidden_states` | Model internals, what hidden states represent |
| **Researcher** | Verify shapes are correct, check values make sense | Web Worker architecture, Comlink |

**Output**: `hiddenStates` available in Zustand store

**Checkpoint Test**:
```
✓ Generate "The cat sat"
✓ hiddenStates[0] shape: [1, 3, 768]
✓ hiddenStates[11] shape: [1, 3, 768]
✓ Values in reasonable range (-10, 10)
```

---

### Session 4: **Researcher Leads** (Hidden State Analysis)

**Goal**: Analyze residual stream growth across layers

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Researcher** | Write `analyzeLayerNorms()` - compute norm per layer | TypeScript Map API, Zustand subscriptions |
| **Engineer** | Create `useLayerNorms()` hook to expose to UI | Why norms increase with depth, residual connections |

**Output**: `src/analysis/embeddings.ts` + React hook

**Checkpoint Test**:
```
✓ Prompt: "Hello world"
✓ Layer norms: [12.3, 14.1, 15.8, ..., 23.4]
✓ Shows growth pattern (validates residual stream theory)
```

---

## Phase 2: Visualization (Weeks 3-4)

### Session 5: **Engineer Leads** (Attention Heatmap)

**Goal**: Build visx heatmap component

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Engineer** | Scaffold `AttentionHeatmap.tsx` with visx | Self-attention visualization patterns |
| **Researcher** | Point out interesting patterns ("look at head 5!") | React component structure, props |

**Output**: Visual heatmap displaying attention weights

**Checkpoint Test**:
```
✓ Prompt: "The cat sat on the mat"
✓ Heatmap shows 12x12 grid of heads
✓ Click head 3.5 → expands to full seq×seq matrix
✓ Diagonal pattern visible (attends to previous token)
```

---

### Session 6: **Researcher Leads** (Attention Pattern Detection)

**Goal**: Classify attention heads automatically

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Researcher** | Implement `classifyAttentionHead()` in `src/analysis/attention.ts` | TypeScript union types, enums |
| **Engineer** | Add UI badges to show head types on heatmap | Induction heads, previous-token heads, what they mean |

**Output**: Labeled heatmap with head classifications

**Checkpoint Test**:
```
✓ Prompt: "Alice went to the store. Alice"
✓ Head 3.2: Labeled "induction" (attends to "went" after second "Alice")
✓ Head 0.1: Labeled "previous_token"
✓ Head 5.7: Labeled "first_token"
```

---

### Session 7: **Engineer Leads** (3D Embedding Space)

**Goal**: Visualize token embeddings in 3D

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Engineer** | Scaffold `EmbeddingSpace.tsx` with React-Three-Fiber | What embeddings represent geometrically |
| **Researcher** | Implement PCA in `src/analysis/embeddings.ts` | 3D rendering, OrbitControls |

**Output**: Interactive 3D point cloud

**Checkpoint Test**:
```
✓ Prompt: "king queen man woman"
✓ 4 points visible in 3D space
✓ king-queen vector ≈ man-woman vector (visually parallel)
✓ Can rotate with mouse
```

---

### Session 8: **Researcher Leads** (Embedding Analysis)

**Goal**: Add clustering and similarity analysis

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Researcher** | Implement cosine similarity, clustering | React state updates, useEffect |
| **Engineer** | Add UI controls (select tokens, show distances) | Why cosine similarity, semantic spaces |

**Output**: Interactive embedding explorer

**Checkpoint Test**:
```
✓ Click "king" → Shows nearest neighbors: ["queen", "prince", "monarch"]
✓ Distance display: king-queen = 0.23
✓ Color clusters by similarity
```

---

## Phase 3: Control Mode (Weeks 5-6)

### Session 9: **Both Together** (ONNX Export)

**Goal**: Run split model export script

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Researcher** | Run `scripts/export_split_model.py`, explain layer split | ONNX format, model export process |
| **Engineer** | Load ONNX in browser via `SplitModelManager.ts` | Why layer 6/7, transformer architecture details |

**Output**: Two ONNX files, browser can load and run them

**Checkpoint Test**:
```
✓ export_split_model.py completes without errors
✓ gpt2_part_a.onnx exists (~250MB)
✓ gpt2_part_b.onnx exists (~250MB)
✓ Validation: max logit difference < 0.01
✓ Browser loads both models successfully
```

---

### Session 10: **Engineer Leads** (Steering Infrastructure)

**Goal**: Implement steering vector injection

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Engineer** | Implement `SteeringEngine.ts` - tensor manipulation | Linear algebra, vector addition |
| **Researcher** | Verify math is correct, suggest optimizations | ONNX Runtime Web API, tensor transfer |

**Output**: `applySteeringVector()` function working

**Checkpoint Test**:
```typescript
const hidden = new TensorView(data, [1, 10, 768]);
const steering = new TensorView(vectorData, [768]);
const result = applySteeringVector(hidden, steering, 1.5);
// ✓ Shape preserved: [1, 10, 768]
// ✓ Norm increased by expected amount
```

---

### Session 11: **Researcher Leads** (Steering Vectors)

**Goal**: Define first steering vector (sentiment)

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Researcher** | Write `computeSentimentSteering()` in `src/analysis/steering/sentiment.ts` | Async/await, worker communication patterns |
| **Engineer** | Create slider UI to control alpha | Activation engineering theory, how α affects output |

**Output**: Slider changes generation output

**Checkpoint Test**:
```
✓ Prompt: "I think this movie is"
✓ α = 0.0: "terrible and boring"
✓ α = 1.0: "okay but not great"
✓ α = 2.0: "amazing and wonderful"
✓ Smooth interpolation between values
```

---

### Session 12: **Both Together** (Refusal Bypass)

**Goal**: Create refusal → compliance steering vector

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Researcher** | Design prompts, compute direction | Ethical implications, safety considerations |
| **Engineer** | Add UI warnings, rate limiting | Adversarial ML, alignment challenges |

**Output**: Refusal bypass demo (for research purposes)

**Checkpoint Test**:
```
✓ Prompt: "How do I hack a"
✓ α = 0.0: "I cannot help with that"
✓ α = 2.0: Model completes the request
✓ Clear warning banner in UI
✓ Results logged for analysis
```

---

## Phase 4: Automated Attack (Weeks 7-8)

### Session 13: **Engineer Leads** (Gradient Estimation)

**Goal**: Implement finite-difference gradient approximation

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Engineer** | Implement `GradientEstimator.ts` | Backpropagation-free optimization, numerical methods |
| **Researcher** | Verify gradient estimates are accurate | Browser performance optimization, Web Workers |

**Output**: Gradient estimation working

---

### Session 14: **Researcher Leads** (Genetic Search)

**Goal**: Implement population-based adversarial search

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Researcher** | Implement `GeneticSearch.ts` - evolutionary algorithm | Promise.all for parallelization |
| **Engineer** | Optimize for browser (parallel workers) | GCG attacks, universal triggers |

**Output**: Search algorithm finding adversarial suffixes

**Checkpoint Test**:
```
✓ Initialize population of 256 candidates
✓ Loss decreases over iterations
✓ Converges to adversarial suffix in <500 iterations
✓ Runs at reasonable speed (~10 iter/sec)
```

---

### Session 15: **Engineer Leads** (Loss Curve Visualization)

**Goal**: Real-time attack progress visualization

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Engineer** | Implement `LossCurve.tsx` with live updates | What the loss curve reveals about search progress |
| **Researcher** | Interpret patterns, suggest UI improvements | Real-time React updates, animation |

**Output**: Live loss curve during attack

---

### Session 16: **Both Together** (End-to-End Attack)

**Goal**: Run full automated adversarial search

| Who | Does What | Learns What |
|-----|-----------|-------------|
| **Both** | Configure attack, run search, analyze results | Complete attack pipeline |

**Checkpoint Test**:
```
✓ Config: target = "Sure, I can help with that"
✓ Search runs for 500 iterations
✓ Loss curve shows convergence
✓ Adversarial suffix found: " [{strange tokens}]"
✓ Model output matches target with suffix
✓ Results exported as JSON
```

---

## Parallel Work Strategy

When one person is blocked, the other can work independently:

### Engineer Can Work On
- **UI Polish**: Loading states, error handling, responsive design
- **Performance**: Memoization, debouncing, lazy loading
- **Infrastructure**: Testing setup, CI/CD, deployment
- **Export Utilities**: Save analysis results as JSON/CSV

### Researcher Can Work On
- **Offline Experiments**: Python notebooks, steering vector development
- **Literature Review**: Find new interpretability patterns to implement
- **Unit Tests**: Write tests for analysis functions
- **Documentation**: Add examples to RESEARCHER_GUIDE.md

---

## Session Structure Template

Each session (~2-3 hours):

```
┌─────────────────────────────────────────────────────────────────┐
│  0:00 - 0:15  │  Sync: What we're building today               │
│               │  Leader explains the goal, shows reference      │
│               │  (paper, notebook, demo)                        │
│                                                                 │
│  0:15 - 1:30  │  Implementation                                 │
│               │  Leader drives, partner navigates               │
│               │  Frequent role swaps encouraged                 │
│                                                                 │
│  1:30 - 2:00  │  Integration                                    │
│               │  Connect the pieces, make it work end-to-end    │
│                                                                 │
│  2:00 - 2:15  │  Checkpoint Test                                │
│               │  Run the test, verify success                   │
│                                                                 │
│  2:15 - 2:30  │  Retro                                          │
│               │  - What did each person learn?                  │
│               │  - What's blocking for next session?            │
│               │  - Assign async tasks if needed                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Collaboration Principles

### 1. The Interface Contract

Before each session, agree on the function signature:

```typescript
// Example: Session 4 contract
// Researcher will implement:
export function analyzeLayerNorms(
  hiddenStates: Map<number, TensorView>
): number[];

// Engineer will consume:
const norms = useLayerNorms();
// Returns: [12.3, 14.1, ..., 23.4]
```

Both parties sign off on:
- Input types and shapes
- Return type and shape
- Edge cases (empty input, invalid layer, etc.)

---

### 2. TODO-Driven Development

Use structured TODOs to mark handoff points:

```typescript
// Engineer leaves:
// RESEARCHER TODO: Implement entropy calculation
// Input: attention [heads, seq, seq]
// Output: entropy per head [heads]
// Formula: H = -Σ p*log(p) where p = attention weights

// Researcher leaves:
// ENGINEER TODO: Connect to UI with slider
// Hook: useAttentionEntropy(layer: number, threshold: number)
// Display: Bar chart showing entropy per head
```

---

### 3. Show, Don't Tell

- **Researcher**: Open Python notebook, show the math working on real data
- **Engineer**: Screen share, show the React component updating in real-time

Example handoff:
```python
# Researcher shows in notebook:
import torch
hidden = model.get_hidden_states("Hello world")
norms = [h.norm().item() for h in hidden]
# [12.3, 14.1, 15.8, ...]

# Engineer implements:
const norms = hiddenStates.map(h => h.norm());
```

---

### 4. Checkpoint Tests Are Sacred

**Rule**: Every session MUST end with a passing checkpoint test.

If the test fails:
1. Debug for 15 minutes max
2. If still broken, roll back and simplify
3. Never leave a session with broken code in main branch

Use feature branches for risky work:
```bash
git checkout -b feature/attention-heatmap
# ... implement ...
# ... test passes ...
git checkout main
git merge feature/attention-heatmap
```

---

## Example Session: Session 4 in Detail

**Pre-session prep** (15 min each, done independently):
- **Engineer**: Read [Anthropic's Residual Stream post](https://transformer-circuits.pub)
- **Researcher**: Review TypeScript Map API, skim Zustand docs

**Session flow**:

### 0:00 - 0:10: Sync
```
Researcher explains:
"Layer norms tell us how much information accumulates in the residual
stream. We expect it to grow monotonically because each layer adds to
the stream. Here's a notebook showing this in PyTorch..."

[Shows notebook with PyTorch example]

Engineer asks:
"Why L2 norm specifically?"
"What if a layer has no effect - norm stays flat?"
```

### 0:10 - 1:00: Researcher Codes
```typescript
// src/analysis/embeddings.ts

/**
 * Computes L2 norm of hidden states at each layer.
 *
 * @param hiddenStates - Map from layer index to hidden state tensor
 * @returns Array of norms, one per layer
 */
export function analyzeLayerNorms(
  hiddenStates: Map<number, TensorView>  // layer → [batch, seq, hidden_dim]
): number[] {
  const norms: number[] = [];

  for (let layer = 0; layer < 12; layer++) {
    const hidden = hiddenStates.get(layer);
    if (!hidden) {
      norms.push(0);
      continue;
    }

    // Take last token position (most interesting)
    const lastToken = hidden.slice([0, -1]);  // [hidden_dim]
    const norm = lastToken.norm();
    norms.push(norm);
  }

  return norms;
}
```

**Engineer watches, asks questions**:
- "Why Map instead of array?" → Better semantic clarity, handles gaps
- "What does `.get(layer)` return if layer doesn't exist?" → undefined, hence the check

### 1:00 - 1:30: Engineer Codes
```typescript
// src/hooks/useLayerNorms.ts

import { useMemo } from 'react';
import { useModelStore } from '../store/modelStore';
import { analyzeLayerNorms } from '../analysis/embeddings';

export function useLayerNorms(): number[] {
  const hiddenStates = useModelStore((state) => state.hiddenStates);

  return useMemo(() => {
    if (hiddenStates.size === 0) return [];
    return analyzeLayerNorms(hiddenStates);
  }, [hiddenStates]);
}
```

**Researcher watches, learns**:
- How `useMemo` prevents re-computation
- Why we subscribe to specific store slices
- What happens on re-render

### 1:30 - 2:00: Together - Integration
```typescript
// src/App.tsx (temporary test UI)

function LayerNormDisplay() {
  const norms = useLayerNorms();

  return (
    <div>
      <h3>Layer Norms</h3>
      {norms.map((norm, i) => (
        <div key={i}>
          Layer {i}: {norm.toFixed(2)}
        </div>
      ))}
    </div>
  );
}
```

### 2:00 - 2:15: Checkpoint Test
```
✓ Type prompt: "Hello world"
✓ Generate with hidden states
✓ See norms: [12.3, 14.1, 15.8, ..., 23.4]
✓ Verify monotonic growth
✓ Try different prompts - pattern holds
```

### 2:15 - 2:30: Retro
```
Engineer learned:
- Why L2 norm is meaningful (magnitude of activation)
- The residual stream accumulates information
- Map vs Array trade-offs

Researcher learned:
- useMemo prevents expensive recalculation
- Zustand selectors for efficient subscriptions
- TypeScript type inference is actually helpful

Blocking issues:
- None

Next session prep:
- Engineer: Read visx docs, prepare for heatmap
- Researcher: Think about attention pattern classification
```

---

## Learning Optimization Tips

### For the Engineer

**Before sessions**:
- Read the specific Anthropic Circuits paper for that topic
- Watch 3Blue1Brown videos on relevant math
- Try the analysis in a Python notebook first

**During sessions**:
- Ask "why" questions about the math
- Request visual explanations (draw on whiteboard)
- Connect to web concepts ("like how React batches updates?")

**After sessions**:
- Implement the analysis function in Python yourself
- Write a blog post explaining what you learned
- Find alternative implementations (TransformerLens, Baukit)

---

### For the Researcher

**Before sessions**:
- Do a TypeScript tutorial (just 1 hour: [TypeScript for JS Programmers](https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes.html))
- Skim React docs on the specific hook you'll use
- Read one Next.js blog post on the web pattern

**During sessions**:
- Ask "how" questions about the architecture
- Request analogies to ML concepts ("like how backprop uses the chain rule?")
- Try to predict the type errors before they happen

**After sessions**:
- Read one React pattern you encountered
- Fork a simple React project, modify it
- Write the web equivalent of a Python script you know

---

## Red Flags (When to Pivot)

| Red Flag | What It Means | Fix |
|----------|---------------|-----|
| Engineer is bored | Too much waiting on researcher | Give engineer parallel task (UI polish) |
| Researcher is lost | Too much web context required | Simplify interface, add more abstraction |
| No checkpoint passed in 2 sessions | Scope too large | Break into smaller milestones |
| Lots of merge conflicts | Not communicating enough | More frequent check-ins, smaller commits |
| One person dominates PRs | Unbalanced contribution | Enforce alternating leadership |
| Skipping retros | Losing shared context | Make retros mandatory, timebox to 15 min |

---

## Success Metrics

### By End of Phase 1
Both should be able to:
- ✅ Explain what an attention head does (in plain English)
- ✅ Read TypeScript type definitions without confusion
- ✅ Write a TensorView operation (e.g., `tensor.mean(axis=0)`)
- ✅ Navigate the codebase independently
- ✅ Run and interpret a checkpoint test

### By End of Phase 2
Both should be able to:
- ✅ Classify attention patterns (previous-token, induction, etc.)
- ✅ Create a React component that consumes analysis data
- ✅ Interpret 3D embedding visualizations
- ✅ Explain the residual stream to someone else

### By End of Phase 3
Both should be able to:
- ✅ Design a new steering vector from scratch
- ✅ Implement it end-to-end (Python export → browser UI)
- ✅ Present it coherently (researcher = theory, engineer = demo)
- ✅ Debug tensor shape mismatches
- ✅ Understand the ethical implications of steering

### By End of Phase 4
Both should be able to:
- ✅ Explain GCG attacks at a deep level
- ✅ Implement a new search algorithm variant
- ✅ Optimize browser performance for compute-heavy tasks
- ✅ Write a research blog post together
- ✅ Independently start a new interpretability project

---

## Communication Channels

### Synchronous
- **Pairing sessions**: 2-3 hours, 2x per week
- **Quick syncs**: 15 min, as needed (Zoom/Slack call)
- **Debug sessions**: Ad-hoc when stuck

### Asynchronous
- **GitHub PRs**: All code reviewed before merge
- **Slack/Discord**: Questions, links, async updates
- **Notion/Docs**: Shared notes, experiment logs

### Documentation
- **Code comments**: Explain "why", not "what"
- **RESEARCHER TODOs**: Clear handoff points
- **ENGINEER TODOs**: UI connection points
- **Commit messages**: Reference session number

Example commit:
```
Session 4: Implement layer norm analysis

- Add analyzeLayerNorms() in src/analysis/embeddings.ts
- Create useLayerNorms() hook
- Checkpoint test passed: monotonic growth observed

Co-authored-by: [Researcher Name] <email@example.com>
```

---

## Tools and Setup

### Shared Environment
- **Code**: GitHub (single repo, feature branches)
- **Meetings**: Zoom with screen share
- **Whiteboard**: Excalidraw or Miro
- **Notes**: Notion or Google Docs

### Engineer's Tools
- **Editor**: VS Code with Prettier, ESLint
- **Browser**: Chrome with React DevTools
- **Terminal**: iTerm2 or Windows Terminal

### Researcher's Tools
- **Editor**: VS Code or Cursor (for AI assist)
- **Notebooks**: Jupyter for offline experiments
- **Python**: Conda environment with transformers, torch

---

## Next Steps

Ready to start Session 1? Here's the prep:

### Engineer Prep (30 min)
1. Read: [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
2. Review: TypeScript generics
3. Browse: TransformerLens `Tensor` class for inspiration

### Researcher Prep (30 min)
1. Read: [TypeScript in 5 Minutes](https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes.html)
2. Review: Our `src/engine/types.ts` file
3. List: What operations you need on tensors

### Session 1 Agenda
- **Goal**: Implement TensorView class
- **Time**: 2 hours
- **Leader**: Engineer
- **Output**: Working tensor operations
- **Checkpoint**: Successfully compute mean, slice, reshape

---

## Resources

### Interpretability
- [Transformer Circuits](https://transformer-circuits.pub) - Anthropic
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Neel Nanda
- [LessWrong Alignment Forum](https://www.alignmentforum.org) - Community

### Web Development
- [React Docs (New)](https://react.dev) - Official
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html) - Official
- [Zustand Guide](https://docs.pmnd.rs/zustand/getting-started/introduction) - State management

### Both
- [3Blue1Brown](https://www.youtube.com/c/3blue1brown) - Visual math
- [Fast.ai](https://course.fast.ai) - Practical deep learning
- [Web.dev](https://web.dev) - Modern web best practices
