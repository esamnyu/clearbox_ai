# CLAUDE.md — NeuroScope-Web

## Current status (June 2026)

Hybrid in-browser frontend + FastAPI/TransformerLens backend. The Dec 2024 goal was GPT-2-only browser-native interp; the Jan 2026 pivot, after a DeepMind consultation, was toward refusal-direction interpretability. Working branch: `proposal/research-direction-2025`.

**Reality of the code right now:** backend supports `gpt2-small` plus whitelisted `meta-llama/Llama-3.2-{1B,3B}-Instruct` (gated; `HF_TOKEN`) — see `ALLOWED_MODELS` in `backend/main.py` and `backend/model.py`. **Ablation primitive is built** — `backend/research.py::ablate_along_direction` + `POST /ablate-direction` + UI in `src/components/SteeringPanel.tsx`. The Refusal Bench harness implements all six techniques (`backend/refusal_bench/techniques/`). WS1–WS5 landed in commit `8169671` (May 31); **WS6 stats hardening — Wilson CIs, bootstrap AUC CIs, permutation p, threaded backend→frontend — is the current in-flight (uncommitted) diff.**

**Active goal:** Anthropic Fellows portfolio (**July 2026 cohort**). Remaining deliverables:

1. **Public deploy** — backend (HF Spaces) + frontend (Vercel); infrastructure committed (`21457c8`), but the HF Space and Vercel project are **not yet created**.
2. ~~**Full 6-technique bench artifact**~~ **DONE (July 29)** — `public/bench/refusal_bench_default.json` is a complete six-row run, zero errors, with WS6 CI fields. Open follow-up: re-run at n=50 for statistical power (see gotchas).
3. Blog post — draft at `docs/BLOG_POST_DRAFT.md`. Results table and repo URL now filled from the real run; **one TBD left**, the Vercel URL, which blocks on deliverable #1.
4. 3-min Loom walkthrough.
5. Polished research statement (template at `docs/FELLOWS_RESEARCH_STATEMENT.md`).

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
| Deploy publicly (HF Space + Vercel) | Cross-family migration (Gemma-2-2B etc.) |
| Full 6-technique bench run on Llama-3.2-1B (migration is done; the artifact isn't) | SAE training / SAELens integration |
| WS6 stats hardening (in flight) | Attribution graphs / circuit-tracer integration |
| Blog post + Loom + research statement | Refactors that don't unblock the deploy |
| README polish | New visualizations beyond what's in `src/components/` |

**Out-of-scope work should be flagged and explicitly confirmed with Ethan before starting** — it violates the sprint plan and is the most common way this project loses time.

## Stack

| Layer | Tech |
| --- | --- |
| Frontend | Vite + React 18 + TypeScript (strict), Zustand, TailwindCSS + Radix |
| Worker | Comlink + `@huggingface/transformers` v3 (WebGPU) |
| Backend | FastAPI + TransformerLens (`gpt2-small` + whitelisted Llama-3.2-1B/3B) |
| Tests | Vitest, mocked pipeline factory in `tests/fixtures/` |

## Key paths

| Path | Purpose |
| --- | --- |
| `src/engine/worker.ts` | Web Worker: model loading, tokenization, generation |
| `src/engine/types.ts` | Shared interfaces (`PipelineFactory`, `TokenizerInterface`) |
| `src/analysis/` | Pure tensor math — no React, no DOM, no async |
| `src/lib/api.ts` | Frontend → backend HTTP client |
| `src/components/RefusalBenchLeaderboard.tsx` | Bench leaderboard; loads cached static result from `public/bench/refusal_bench_default.json` on mount, optional live re-run via `/refusal-bench` |
| `public/bench/` | Cached bench JSON served as a Vite static asset on the deploy |
| `backend/main.py` | FastAPI endpoints: `/load`, `/logit-lens`, `/attention`, `/gradients`, `/steering-vector`, `/generate-steered`, **`/ablate-direction`**, `/pca-trajectories`, `/contrastive-pairs`, `/refusal-pairs`, `/harmfulness-probe`, `/refusal-bench`, `/refusal-bench/techniques` |
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

- `backend/main.py` layer fields are `le=27` to accommodate Llama-3.2-3B (28 layers). Runtime `_validate_layer` rejects out-of-range values per the actually-loaded model, so this is permissive at the schema layer and strict at execution.
- The file is `claude.md` (lowercase) on disk; macOS APFS is case-insensitive so `CLAUDE.md` and `claude.md` resolve to the same inode. Git tracks the lowercase name. Don't try to "delete the duplicate" — there isn't one.
- ~~Pre-existing TypeScript errors in `src/engine/worker.ts`~~ **FIXED (July 29).** All 6 are gone and `npm run build` (`tsc && vite build`) passes. The ProgressInfo drift is now handled by `in`-narrowing plus a `toLoadStatus()` mapper instead of a lying `as` cast — transformers.js emits `initiate|download|progress|done|ready`, `LoadProgress` only models `downloading|loading|ready`, and the old code cast one onto the other.
- **`npm run lint` had never run.** The script existed from the first commit but no ESLint config file did, so it always exited non-zero with "couldn't find a configuration file." `.eslintrc.cjs` now exists (eslintrc format — ESLint is pinned at 8.57, where flat config is still opt-in). Currently 0 errors / 26 warnings, all `no-console`.
- **CI exists now** — `.github/workflows/ci.yml`: typecheck, lint, vitest, `vite build`, backend pytest on 3.11+3.12, and a `docker build ./backend`. Note `vercel.json` runs `vite build` *without* `tsc`, so **CI is the only place types are checked before production**. The docker job is correct-by-inspection but has never been executed — there is no Docker daemon on this Mac.
- `src/engine/worker.ts`'s pipeline-factory DI seam is **vestigial**: `_setPipelineFactory`/`_resetPipelineFactory` exist and `docs/TESTING_STRATEGY.md` specifies the pattern, but `loadModel` calls `GPT2LMHeadModel.from_pretrained` / `AutoTokenizer.from_pretrained` directly and never consults the factory. Swapping it therefore redirects nothing. `_getPipelineFactory` was added so the seam is at least observable; actually routing `loadModel` through it is an open, behaviour-changing task.
- Bench leaderboard ships a **cached static result** (`public/bench/refusal_bench_default.json`). As of **July 29 2026 this is a complete six-technique run** on Llama-3.2-1B, layer 8, **CPU/bfloat16**, seed 42, n=20 pairs/class, zero errored rows — with the full WS6 CI fields, so the leaderboard renders ± bands. Provenance (`device`, `dtype`, `seed`, `n_pairs_per_class`, `max_new_tokens`) is in the artifact and shown in the UI. Details in `docs/bench_partials_local/README.md`.
  - **Read it with the intervals.** n=20 leaves 5 eval prompts / 10 AUC points; post-ablation AUC CIs span ~[0.25, 1.00]. **Zero techniques met the dissociation criterion.** A higher-power run (n=50, all available pairs) is the obvious next step — budget ~2.5–4 h on CPU, since COSMIC alone took 19 min and Cheng 17 min at n=20.
  - **COSMIC's row is identical to Arditi's** to 17 sig figs. Not a bug: COSMIC's candidate per layer *is* diff-of-means, so its search reduces to layer selection, and it picked layer 8 — Arditi's layer. `findDuplicateRows()` flags this in the UI. Never count the two as independent evidence.
  - `docs/bench_partials/` (May 21) remains historical-only: not strict JSON, 5/6 errored. Don't cite it.
  - The backend defaults to GPT-2; "re-run live" hits `/refusal-bench` on whatever model is loaded — the full sweep is ~50 min on CPU and will time out on the free tier.
- **The CPU bench path was dead until July 29** — `run_bench_local.py` casts to bfloat16 on CPU, and torch cannot convert bfloat16 to numpy (`TypeError: Got unsupported ScalarType BFloat16`), killing probe training. MPS casts to float16, which converts fine, so the bug was invisible on the MPS path. Fixed with `.float()` upcasts in `harmfulness_probe.py` (×2) and `runner.py`. **Prefer CPU regardless**: TransformerLens warns MPS "may produce silently incorrect results" on torch 2.12, and MPS held ~11 GB RSS here vs ~4–5 GB on CPU.
- **TransformerLens calls hooks by keyword** — `fn(tensor, hook=hook_point)`. A closure whose second parameter is named anything but `hook` raises `unexpected keyword argument 'hook'`, but only deep inside a live generation loop. This shipped twice (Wollschlager in May, Herring in July). `backend/tests/test_hook_signatures.py` now catches it statically via `ast`, no model needed.

## Permissions

- `./src`, `./tests`, `./backend` — full read/write.
- Don't commit `.env`, model weights, or `node_modules`. Add specific files by name; avoid `git add -A` / `git add .`.
