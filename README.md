# NeuroScope

> A browser-native mechanistic interpretability workbench. Load a model, extract activations, visualize attention, inject steering vectors, ablate refusal directions, and benchmark six published refusal-ablation techniques head-to-head — built to run from a single URL once the public deploy lands.

*Live demo, walkthrough video, and blog post land with the public deploy — runbook in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).*

## What's inside

- **Frontend**: Vite + React + TypeScript (strict), transformers.js in a Web Worker for in-browser GPT-2 inference.
- **Backend**: FastAPI + TransformerLens. Supports `gpt2-small` (default) and `meta-llama/Llama-3.2-{1B,3B}-Instruct` (gated; requires `HF_TOKEN`).
- **Refusal Bench**: the harness runs six published refusal-ablation techniques head-to-head — Arditi, Wollschlager, COSMIC, Cheng, Maskey, and Herring — against a shared harmfulness probe (Zhao 2507.11878–style scoring). The shipped result (`public/bench/refusal_bench_default.json`) is a **complete six-technique run on Llama-3.2-1B**, layer 8, CPU/bfloat16, seed 42, no errored rows. Live re-runs via `POST /refusal-bench` locally; the full sweep takes ~50 min on CPU at n=20, so the deploy serves the cached artifact rather than running it.

  Read the numbers with their intervals: the run uses 20 pairs per class, which leaves 5 eval prompts and 10 AUC points, and the bootstrap CIs on post-ablation AUC are correspondingly wide. Zero techniques met the preregistered dissociation criterion. `docs/BLOG_POST_DRAFT.md` has the full table and the caveats.
- **Statistical honesty layer**: Wilson 95% CIs on refusal rates (`wilson_ci`), percentile-bootstrap 95% CIs on probe AUC (`bootstrap_auc_ci`), and a one-sided permutation test that post-ablation AUC beats chance (`auc_permutation_p`) — seeded and deterministic, unit-tested in `backend/tests/test_stats.py`, rendered as ± bands in the leaderboard UI.
- **Ablation primitive**: projection-removal hook (`h − (h · d̂)d̂`) exposed at `POST /ablate-direction` with a `SteeringPanel` UI for interactive before/after generation.

## Research context

I'm building this while teaching myself mechanistic interpretability. The 2024–26 refusal-direction literature — and where the single-direction claim (Arditi 2024) breaks down under follow-up work (Wollschlager 2025, Zhao 2025, Cheng 2604.08524, Maskey 2603.27518, Herring 2605.12290) — is summarized in [`docs/RESEARCH_LANDSCAPE_2026.md`](docs/RESEARCH_LANDSCAPE_2026.md). The bench is the headline artifact: it operationalizes the Zhao dissociation criterion across all six published techniques on the same data.

## Run locally

```bash
# Frontend
npm install
npm run dev                              # → http://localhost:3001

# Backend (separate terminal)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000    # OpenAPI at /docs
```

For Llama-3.2-{1B,3B} you must accept the license on Hugging Face and export `HF_TOKEN` before starting the backend. GPT-2 needs no token.

## Project layout

| Path | Purpose |
|---|---|
| `src/engine/` | Web Worker — transformers.js inference, tokenization, generation |
| `src/analysis/` | Pure tensor math (no React, no DOM, no async) |
| `src/components/` | UI: `RefusalBenchLeaderboard`, `SteeringPanel`, `AblationHero`, etc. |
| `backend/research.py` | TransformerLens-backed analysis (logit lens, attention, steering, ablation) |
| `backend/refusal_bench/` | Bench harness + six technique implementations + harmfulness probe |
| `public/bench/` | Cached bench result served as a Vite static asset |
| `docs/` | Architecture, deployment, research landscape, testing strategy |

## Stack

Vite · React 18 · TypeScript strict · transformers.js v3 · Zustand · Tailwind + Radix · FastAPI · TransformerLens · PyTorch

## Status

Active development. Public deploy is in progress, not yet live — deploy targets are HuggingFace Spaces (Docker SDK, free CPU tier) for the backend and Vercel for the frontend. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the runbook.

## License

MIT
