# NeuroScope-Web Documentation

This directory contains all project documentation.

## Documentation Files

| File | Audience | Purpose |
|------|----------|---------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Both | Complete technical architecture, tech stack, implementation roadmap |
| [RESEARCHER_GUIDE.md](./RESEARCHER_GUIDE.md) | ML Researcher | Onboarding guide, TensorView API, example analysis functions |
| [COLLABORATION_WORKFLOW.md](./COLLABORATION_WORKFLOW.md) | Both | Session-by-session workflow, learning strategy, checkpoints |

## Quick Links

### For Researchers
1. **Start here**: [RESEARCHER_GUIDE.md](./RESEARCHER_GUIDE.md)
2. **Your workspace**: [`src/analysis/`](../src/analysis/)
3. **Find TODOs**: `grep -r "RESEARCHER TODO" src/analysis/`

### For Engineers
1. **Start here**: [ARCHITECTURE.md](./ARCHITECTURE.md)
2. **Collaboration guide**: [COLLABORATION_WORKFLOW.md](./COLLABORATION_WORKFLOW.md)
3. **Session 1 prep**: Read NumPy broadcasting, review TypeScript generics

### For Both
- **Current phase**: Session 1 (TensorView class implementation)
- **Next milestone**: Tokenization checkpoint test
- **Communication**: Use RESEARCHER TODO / ENGINEER TODO markers in code

## Session 1 Starter Pack

The following files are ready for pair programming:

### Implemented ✅
- `src/analysis/utils/tensor.ts` - TensorView class (partial)
- `src/analysis/attention.ts` - Template with TODOs
- `src/analysis/embeddings.ts` - Template with TODOs
- `src/analysis/steering/sentiment.ts` - Template with TODOs

### TODOs for Session 1
- [ ] Complete `TensorView.slice()` with range support
- [ ] Implement `TensorView.sum(axis)` and `mean(axis)`
- [ ] Test with checkpoint: `tensor.mean(0)` works correctly

### TODOs for Session 2
- [ ] Implement `computeNorm()` in `attention.ts`
- [ ] Connect to UI via `useLayerNorms()` hook
- [ ] Verify norms display correctly

## Repository Structure

```
clearbox_ai/
├── docs/                      # This directory
│   ├── README.md             # This file
│   ├── ARCHITECTURE.md       # Technical architecture
│   ├── RESEARCHER_GUIDE.md   # Researcher onboarding
│   └── COLLABORATION_WORKFLOW.md  # Pair programming guide
│
├── src/
│   ├── analysis/             # Researcher workspace
│   │   ├── README.md        # Analysis module guide
│   │   ├── utils/
│   │   │   └── tensor.ts    # TensorView class
│   │   ├── attention.ts     # Attention analysis
│   │   ├── embeddings.ts    # Embedding analysis
│   │   └── steering/
│   │       └── sentiment.ts # Steering vectors
│   │
│   ├── engine/              # Model inference (Web Worker)
│   │   ├── worker.ts       # Web Worker entry point
│   │   └── types.ts        # Type definitions
│   │
│   ├── store/              # State management (Zustand)
│   │   └── modelStore.ts
│   │
│   └── App.tsx            # Main UI (Session 1: tokenization demo)
│
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Contribution Guidelines

### For Researchers
- **Add code**: Only modify files in `src/analysis/`
- **Document**: Use JSDoc with `@math_note` and `@reference`
- **Test**: Write unit tests in `src/analysis/__tests__/`
- **Commit**: Include "Researcher:" prefix in commit messages

### For Engineers
- **Add code**: Modify files in `src/engine/`, `src/hooks/`, `src/vis/`
- **Document**: Use JSDoc with clear examples
- **Integration**: Connect analysis functions via hooks
- **Commit**: Include "Engineer:" prefix in commit messages

### Both
- **Pair programming**: Use `RESEARCHER TODO` and `ENGINEER TODO` markers
- **Co-authored commits**: Add `Co-authored-by:` line
- **Checkpoint tests**: Must pass before merging to main

## Current Status

**Phase**: 1 (Observation Mode - Foundation)
**Session**: 1 (TensorView class)
**Completed**:
- ✅ Vite + React + TypeScript setup
- ✅ TailwindCSS configuration
- ✅ Web Worker with Comlink
- ✅ Zustand store
- ✅ TensorView class (partial implementation)
- ✅ Analysis module templates

**Next Steps**:
1. Session 1: Complete TensorView operations
2. Session 2: Implement first analysis function
3. Session 3: Extract hidden states from model
4. Session 4: Visualize layer norms

---

**Last Updated**: 2025-12-21
**Contributors**: Engineer + ML Researcher (CMU)
