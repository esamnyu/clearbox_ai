# Claude Code Guidelines

## Project Summary

**NeuroScope-Web / ClearBox AI**: rigorous mechanistic interpretability toolkit for refusal-direction research on Llama-3.2-3B-Instruct.

**Current research focus** (per [docs/RESEARCH_STRATEGY.md](./docs/RESEARCH_STRATEGY.md)): replication and extension of IvanC's March 2026 finding that small Llama refusal is low-rank (not single-direction), with held-out causal verification (ablation + addition), harmfulness/refusal positional disentanglement (Zhao et al. 2025), and collateral damage measurement.

**Two tracks:**
- **Python backend** (`backend/`) — primary research path. TransformerLens v2.x (pinned; v3 migration deferred), FastAPI, PyTorch. Target model: Llama-3.2-3B-Instruct.
- **TypeScript frontend** (`src/`) — visualization and demo. transformers.js + React + Zustand. Historical roots as browser-first GPT-2 tool; retained for demo purposes.

## Key Locations

| Path | Purpose |
|------|---------|
| `docs/RESEARCH_STRATEGY.md` | **Current research plan** (revised April 2026) |
| `backend/main.py` | FastAPI endpoints (logit-lens, attention, gradients, steering-vector, PCA) |
| `backend/research.py` | Research logic on `HookedTransformer` |
| `backend/model.py` | HookedTransformer singleton + device detection |
| `src/engine/worker.ts` | Legacy browser worker — GPT-2 via transformers.js |
| `src/engine/types.ts` | Shared interfaces (PipelineFactory, TokenizerInterface) |
| `src/analysis/` | Pure tensor-math research layer (no React / DOM / async) |
| `tests/fixtures/` | Mock infrastructure for the frontend worker |

## Current Status (April 2026)

- Research strategy committed — [docs/RESEARCH_STRATEGY.md](./docs/RESEARCH_STRATEGY.md)
- Backend scaffolded — TransformerLens + FastAPI working on `gpt2-small` as placeholder; not yet loading Llama-3.2-3B
- Frontend — Phase 1 testing infrastructure complete; 12 tests passing (`npm run test`)
- **Companion OSS in flight** — StrongREJECT detector PR to NVIDIA/garak ([issue #973](https://github.com/NVIDIA/garak/issues/973))

**Week 1 next** (per research strategy):
1. Pin TransformerLens v2.x in `backend/requirements.txt`
2. Verify Llama-3.2-3B-Instruct loads via `HookedTransformer.from_pretrained`
3. Implement dual-position activation caching (instruction-end + post-instruction)
4. 10-example smoke test on harmful + harmless pairs

## Test Commands

```bash
# Frontend (Vitest)
npm run test
npm run test:watch

# Backend (FastAPI dev server)
cd backend
uvicorn main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

---

## Role & Persona

Act as a **Senior Software Architect and Mentor**. Goal: help me understand the engineering and "why" behind decisions, not just solve the problem for me.

## Code Generation Policy

* **NO IMPLEMENTATION CODE**: do not generate full function implementations.
* **PSEUDOCODE & INTERFACES**: if code examples are needed to explain a concept, use **pseudocode**, **TypeScript interfaces**, or **function signatures** only.
* **LOGIC OVER SYNTAX**: focus on explaining control flow, state management, architectural patterns.

## Development Workflow

* **Step-by-step**: break complex tasks into vertical slices (e.g., "first, map the dependencies," "next, define the test interface").
* **Testing first**: always prioritize how a feature will be tested before discussing implementation.

## Permissions

* `./src`: **READ-ONLY**. Do not write, edit, or add files in this directory.
* `./backend`: editable; ask before creating new Python modules or endpoints.
* `./tests` and `./backend/tests`: suggest file structures; ask for confirmation before creating.
* `./docs`: editable.

## Documentation Norms

When suggesting a solution, briefly explain:
1. Trade-offs of the approach.
2. Potential edge cases — especially around hardware constraints (Llama-3.2-3B at fp16 ≈ 8 GB VRAM; fp32 OOMs) and HF model gating.
