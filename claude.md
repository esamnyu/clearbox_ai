# CLAUDE.md — NeuroScope-Web

## Current status (late April 2026)

Hybrid in-browser frontend + FastAPI/TransformerLens backend. The Dec 2024 goal was GPT-2-only browser-native interp; the Jan 2026 pivot, after a DeepMind consultation, was toward refusal-direction interpretability. Working branch: `proposal/research-direction-2025`.

**Reality of the code right now:** backend is GPT-2-only (`gpt2-small`); 8 sentiment contrastive pairs in `research.py::get_contrastive_pairs`. **Ablation primitive is built** — `backend/research.py::ablate_along_direction` + `POST /ablate-direction` + UI in `src/components/SteeringPanel.tsx`. Llama-3.2-1B/3B migration is **deferred** per `docs/FELLOWS_SPRINT_9DAY.md`.

**Active goal:** Anthropic Fellows portfolio (May 2026 cohort deadline April 26 has passed; **July cohort is the next live target**). Remaining deliverables:

1. **Deploy** backend (HF Spaces) + frontend (Vercel) on a public URL — infrastructure committed (`21457c8`); HF Space + Vercel project not yet created.
2. ~~One projection-removal ablation primitive~~ — **done.**
3. One small before/after experiment on the existing sentiment pairs (or first contact with Llama-3.2-1B if migrating early).
4. ~800-word blog post + 3-min Loom walkthrough.
5. Polished research statement (template at `docs/FELLOWS_RESEARCH_STATEMENT.md`, updated April 28).

**Research thesis as of April 28:** revised — the headline claim is now Zhao-style harmfulness probe after refusal ablation on **Llama-3.2-1B** (still open per literature search), with the rank-k curve demoted to "replication + extension." See "Update — April 28 2026" in `docs/RESEARCH_LANDSCAPE_2026.md`.

## Read these docs in priority order

1. `docs/RESEARCH_LANDSCAPE_2026.md` — definitive April 17 2026 brief; supersedes earlier strategy where they conflict.
2. `docs/FELLOWS_SPRINT_9DAY.md` — operational in-scope / out-of-scope for the immediate sprint.
3. `docs/FELLOWS_RESEARCH_STATEMENT.md` — application essay outline.
4. `docs/ARCHITECTURE.md` — system architecture.
5. `docs/RESEARCH_PROPOSAL.md`, `docs/RESEARCH_STRATEGY.md` (Jan 2026) — historical context, superseded for tactics but still cited for framing.
6. `docs/RESEARCHER_GUIDE.md`, `docs/TESTING_STRATEGY.md` — onboarding + test architecture.

## Thesis

- **Engineering thesis (active):** browser-native, zero-install mech-interp workbench with an integrated methodology loop (prompt pairs → diff-of-means → ablation → writeup). Defensible niche per `RESEARCH_LANDSCAPE_2026.md` §2.
- **Research thesis (deferred to post-Fellows-application):** *"When the single-direction refusal claim breaks down on small Llama models, does the residual harmfulness signal after ablation reveal that current 'refusal ablation' techniques are behavioral patches, not causal interventions?"* Lead = Zhao 2507.11878-style harmfulness probe on Llama-3.2-1B (still open). Replication+extension = compliance-vs-rank-k curve following Wollschlager 2502.17420 + Cheng 2604.08524. Replication, not novel discovery — framed as portfolio, not publication.

## In scope vs out of scope

| In scope | Out of scope |
| --- | --- |
| Deploy the GPT-2 tool publicly (HF Space + Vercel) | Llama-3.2 / Gemma-2-2B migration (deferred to post-application) |
| Wire the existing ablation primitive into a small experiment | SAE training / SAELens integration |
| One experiment on existing sentiment pairs | Attribution graphs / circuit-tracer integration |
| Blog post + Loom + research statement | Refactors that don't unblock the deploy |
| README polish | New visualizations beyond what's in `src/components/` |

**Out-of-scope work should be flagged and explicitly confirmed with Ethan before starting** — it violates the sprint plan and is the most common way this project loses time.

## Stack

| Layer | Tech |
| --- | --- |
| Frontend | Vite + React 18 + TypeScript (strict), Zustand, TailwindCSS + Radix |
| Worker | Comlink + `@huggingface/transformers` v3 (WebGPU) |
| Backend | FastAPI + TransformerLens (`gpt2-small`) |
| Tests | Vitest, mocked pipeline factory in `tests/fixtures/` |

## Key paths

| Path | Purpose |
| --- | --- |
| `src/engine/worker.ts` | Web Worker: model loading, tokenization, generation |
| `src/engine/types.ts` | Shared interfaces (`PipelineFactory`, `TokenizerInterface`) |
| `src/analysis/` | Pure tensor math — no React, no DOM, no async |
| `src/lib/api.ts` | Frontend → backend HTTP client (currently modified, uncommitted) |
| `backend/main.py` | FastAPI endpoints: `/load`, `/logit-lens`, `/attention`, `/gradients`, `/steering-vector`, `/generate-steered`, **`/ablate-direction`**, `/pca-trajectories`, `/contrastive-pairs` |
| `backend/research.py` | TransformerLens-backed analysis logic |
| `backend/model.py` | Singleton model loading |
| `tests/fixtures/` | `mockPipelineFactory`, `mockTokenizer` |
| `docs/TESTING_STRATEGY.md` | Test architecture |

## Commands

```bash
# Frontend
npm run dev          # localhost:3001
npm run test         # vitest
npm run test:watch   # vitest --watch
npm run build        # tsc && vite build
npm run lint         # eslint src --ext ts,tsx

# Backend
cd backend
uvicorn main:app --reload --port 8000   # backend at :8000; OpenAPI at /docs
```

## Role + project-specific norms

- Act as **Senior Software Architect & Mentor**: explain trade-offs and edge cases while implementing, not after.
- Always flag edge cases around the **500 MB GPT-2 model load** and the **GPT-2-vs-Llama divergence** that's baked into current code.
- Keep `src/analysis/` pure: pure functions only, no React, no DOM, no async.
- Use the dependency-injection pattern for the worker (see `tests/fixtures/mockPipelineFactory`).
- Test-first when feasible (`docs/TESTING_STRATEGY.md`).
- The docs run **ahead of the code** — see `RESEARCH_LANDSCAPE_2026.md` §4 ("Strategy-vs-code gaps to close first"). Don't quote a strategy doc as evidence the code does X; verify in the source.

## Live gotchas

- `backend/main.py` layer fields use `le=11` (GPT-2-small specific). For Llama-3.2-1B (16 layers) bump to `le=15`; for 3B (28 layers) bump to `le=27`. Affects `AttentionRequest`, `SteeringRequest`, `SteeredGenerationRequest`, and `AblationRequest`.
- The file is `claude.md` (lowercase) on disk; macOS APFS is case-insensitive so `CLAUDE.md` and `claude.md` resolve to the same inode. Git tracks the lowercase name. Don't try to "delete the duplicate" — there isn't one.
- Pre-existing TypeScript errors in `src/engine/worker.ts` (transformers.js v3 ProgressInfo type drift) and `src/lib/utils.ts` (missing `clsx`/`tailwind-merge` declarations) are unrelated to current work but will fail `tsc --noEmit`. Out of scope unless the build needs them green.
- `RESEARCH_STRATEGY.docx` is an untracked binary in the repo root — Jan 2026 strategy in Word form. Source-of-truth is the `.md` files in `docs/`.

## Permissions

- `./src`, `./tests`, `./backend` — full read/write.
- Don't commit `.env`, model weights, or `node_modules`. Add specific files by name; avoid `git add -A` / `git add .`.
