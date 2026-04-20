# NeuroScope-Web / ClearBox AI

> Rigorous mechanistic interpretability for refusal-direction research on small Llama models.

A paired engineer + researcher project investigating where refusal mechanisms live in the residual stream of Llama-3.2-3B-Instruct, with proper held-out causal verification and collateral-damage measurement.

**Status**: Pre-experiment foundation (April 2026). Next: Week 1 of [docs/RESEARCH_STRATEGY.md](./docs/RESEARCH_STRATEGY.md).

## The Research Question

Default methodology for studying refusal — difference-of-means extraction of a single "refusal direction" — under-characterizes refusal structure in small Llama models. A [March 2026 LessWrong post](https://www.lesswrong.com/posts/LMkvjDTLKFrgdzJdG/single-direction-vs-low-rank-refusal-in-small-llms-1) reports that on Llama-3.2-3B-Instruct, a single extracted vector yields only 15–26% compliance on ablation, while the top-3 SVD directions from layers {9, 10, 7} push compliance to ~37% at 94% coherence.

This project:

1. **Replicates** the k=1 vs k=3 comparison with proper held-out splits, collateral damage measurement, and StrongREJECT evaluation.
2. **Extends** by disentangling harmfulness (encoded at instruction-end position) from refusal (post-instruction position), per [Zhao et al. 2025](https://arxiv.org/abs/2507.11878).
3. **Measures cost** — MMLU delta, Alpaca KL-divergence, XSTest over-refusal — alongside gain.

See [docs/RESEARCH_STRATEGY.md](./docs/RESEARCH_STRATEGY.md) for full methodology.

## Quick Start

### Python backend (research path)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### TypeScript frontend (visualization / demo)

```bash
npm install
npm run dev   # http://localhost:3001
```

## Documentation

| Document | Purpose |
|---|---|
| [**docs/RESEARCH_STRATEGY.md**](./docs/RESEARCH_STRATEGY.md) | **Current research plan** (revised April 2026 for low-rank refusal framing) |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Full technical architecture |
| [docs/RESEARCHER_GUIDE.md](./docs/RESEARCHER_GUIDE.md) | Researcher workspace onboarding |
| [docs/COLLABORATION_WORKFLOW.md](./docs/COLLABORATION_WORKFLOW.md) | Pair-programming workflow |
| [docs/TESTING_STRATEGY.md](./docs/TESTING_STRATEGY.md) | Test architecture (frontend) |

## Project Structure

```
clearbox_ai/
├── backend/             # Python + FastAPI + TransformerLens — primary research path
│   ├── main.py          #   FastAPI endpoints (logit-lens, attention, gradients, steering)
│   ├── model.py         #   HookedTransformer singleton
│   ├── research.py      #   Research logic (ported from Moon's notebooks)
│   └── requirements.txt
├── src/                 # TypeScript + React — visualization / demo
│   ├── analysis/        #   Pure tensor-math research layer (no React / DOM)
│   ├── engine/          #   Web Worker running transformers.js (historical GPT-2 work)
│   ├── store/
│   └── App.tsx
├── docs/                # Strategy, methodology, collaboration
├── notebooks/           # Moon's interpretability prototypes (interp_test, steering_vectors)
└── tests/               # Frontend unit tests (Vitest)
```

## Team

- **Ethan Sam** — engineering, product, tooling
- **Mahmoud "Moon"** — AI Security researcher, CMU
- **Advisor consultation**: Anthony Chen, Google DeepMind (Jan 2026)

## Tech Stack

**Research backend**
- Python 3.11+, FastAPI
- TransformerLens v2.x (pinned; v3.0 migration deferred)
- PyTorch, Hugging Face Transformers

**Frontend / demo**
- Vite, React 18, TypeScript (strict)
- Zustand, Comlink, transformers.js
- TailwindCSS, Radix UI

**Evaluation stack**
- StrongREJECT (automated refusal evaluator)
- JailbreakBench, Alpaca, SORRY-Bench, XSTest

## Companion OSS Contribution

StrongREJECT detector PR to NVIDIA's [garak](https://github.com/NVIDIA/garak) red-team scanner — [issue #973](https://github.com/NVIDIA/garak/issues/973). The parser is shared logic between garak and this project's backend evaluator.

## Key References

- Arditi et al., NeurIPS 2024 — [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (the original methodology being tested)
- IvanC, LessWrong March 2026 — [Single Direction vs Low-Rank Refusal in Small LLMs](https://www.lesswrong.com/posts/LMkvjDTLKFrgdzJdG/single-direction-vs-low-rank-refusal-in-small-llms-1) (the specific claim being replicated)
- Zhao et al., 2025 — [LLMs Encode Harmfulness and Refusal Separately](https://arxiv.org/abs/2507.11878)
- Souly et al., NeurIPS 2024 — [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [Transformer Circuits (Anthropic)](https://transformer-circuits.pub)

## License

MIT

---

**Last updated**: April 19, 2026
