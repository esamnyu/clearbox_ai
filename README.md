# NeuroScope-Web

> Browser-based mechanistic interpretability toolkit for GPT-2 with adversarial capabilities

A collaborative project for visualizing and manipulating transformer internals, designed for pair programming between an engineer and an ML researcher.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server (runs on port 3001)
npm run dev

# Open browser
http://localhost:3001
```

## Session 1 Checkpoint

Load GPT-2 → Tokenize "Hello world" → Verify tokens: `["Hello", " world"]`, IDs: `[15496, 995]`

## Documentation

| Document | Audience | Purpose |
|----------|----------|---------|
| [**docs/ARCHITECTURE.md**](./docs/ARCHITECTURE.md) | Both | Full technical architecture, tech stack, roadmap |
| [**docs/RESEARCHER_GUIDE.md**](./docs/RESEARCHER_GUIDE.md) | ML Researcher | TensorView API, analysis examples, onboarding |
| [**docs/COLLABORATION_WORKFLOW.md**](./docs/COLLABORATION_WORKFLOW.md) | Both | Pair programming workflow, session structure |
| [**docs/README.md**](./docs/README.md) | Both | Documentation index, current status |

## Project Structure

```
clearbox_ai/
├── docs/                      # All documentation
├── src/
│   ├── analysis/             # 🧑‍🔬 Researcher workspace (analysis functions)
│   ├── engine/              # 🔧 Model inference (Web Worker)
│   ├── store/               # State management
│   └── App.tsx              # Main UI
└── package.json
```

## For Researchers

**Your workspace**: [`src/analysis/`](./src/analysis/)

Find your tasks:
```bash
grep -r "RESEARCHER TODO" src/analysis/
```

See: [RESEARCHER_GUIDE.md](./docs/RESEARCHER_GUIDE.md)

## For Engineers

**Tech stack**:
- Vite + React 18 + TypeScript (strict)
- Transformers.js (WebGPU backend)
- Zustand (state management)
- TailwindCSS + Radix UI

See: [ARCHITECTURE.md](./docs/ARCHITECTURE.md)

## Current Phase

**Phase 1: Observation Mode** (Weeks 1-2)

- [x] Vite + React + TypeScript setup
- [x] Web Worker with Comlink
- [x] TensorView class (partial)
- [ ] Hidden state extraction
- [ ] Attention heatmap visualization

## Features

### Phase 1: Observation Mode
- ✅ Model loading (GPT-2, GPT-2-medium)
- ✅ Tokenization display
- 🚧 Hidden state extraction
- 🚧 Attention pattern visualization
- 🚧 3D embedding space

### Phase 2: Control Mode
- ⏳ Split ONNX model export
- ⏳ Steering vector injection
- ⏳ Manual residual stream manipulation

### Phase 3: Automated Attack
- ⏳ Gradient estimation (finite differences)
- ⏳ Genetic adversarial search (GCG-style)
- ⏳ Real-time loss curve visualization

## Scripts

```bash
npm run dev          # Start dev server (port 3001)
npm run build        # Production build
npm run preview      # Preview production build
npm run test         # Run tests
npm run test:watch   # Watch mode
npm run lint         # Lint code
```

## Tech Stack

**Framework**: Vite + React 18 + TypeScript (strict mode)
**Inference**: @xenova/transformers v3 (WebGPU)
**State**: Zustand v4
**Visualization**: React-Three-Fiber + visx
**UI**: TailwindCSS + Radix UI
**Worker**: Comlink (type-safe RPC)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  RESEARCHER LAYER    Pure functions on TensorView          │
│  (src/analysis/)     NO React, NO DOM, NO async            │
├─────────────────────────────────────────────────────────────┤
│  INTERFACE LAYER     React hooks bridging engine ↔ viz     │
│  (src/hooks/)        useLayerActivations(), useAttention() │
├─────────────────────────────────────────────────────────────┤
│  ENGINE LAYER        Web Worker running transformers.js    │
│  (src/engine/)       Returns typed arrays + shape metadata │
├─────────────────────────────────────────────────────────────┤
│  VISUALIZATION       React components consuming data       │
│  (src/vis/)          AttentionHeatmap, EmbeddingSpace      │
└─────────────────────────────────────────────────────────────┘
```

## Contributing

This is a collaborative project with specific roles:

**Researcher**: Adds analysis code in `src/analysis/`
**Engineer**: Adds infrastructure in `src/engine/`, `src/hooks/`, `src/vis/`

See [COLLABORATION_WORKFLOW.md](./docs/COLLABORATION_WORKFLOW.md) for detailed workflow.

## License

MIT

## References

- [Transformer Circuits (Anthropic)](https://transformer-circuits.pub)
- [Transformers.js](https://huggingface.co/docs/transformers.js)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [GCG Attacks](https://arxiv.org/abs/2307.15043)

---

**Status**: Phase 1, Session 1 (TensorView implementation)
**Contributors**: Engineer + ML Researcher (CMU)
**Last Updated**: 2025-12-21
