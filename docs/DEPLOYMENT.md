# Deployment Guide

> Goal: NeuroScope live at a public URL a reviewer can click, in an afternoon, at zero cost.
> Split: **HuggingFace Spaces** (backend, free CPU) + **Vercel** (frontend, free).

**Read this first — what is already done.** This table is the source of truth;
§1 below is stale in one important respect and is kept only as reference for
rebuilding the Space from scratch. **The backend is live.** The Space exists, is
`RUNNING` on `cpu-basic`, and serves all 14 routes at
`https://lymnal-neuroscope-api.hf.space` (verify: `GET /` → `{"status":"ok"}`).

| Thing | State | Where |
|---|---|---|
| Backend Dockerfile | ✅ committed | `backend/Dockerfile` (+ `.dockerignore`) |
| Pinned requirements | ✅ committed | `backend/requirements.txt` |
| Configurable CORS | ✅ committed | `backend/main.py` — `ALLOWED_ORIGINS` env |
| Model whitelist | ✅ committed | `backend/main.py` — `ALLOWED_MODELS` |
| Frontend API base URL | ✅ committed | `src/lib/api.ts` — `VITE_API_BASE` |
| Vercel build config | ✅ committed | `vercel.json` |
| README rewrite | ✅ done | `README.md` |
| **HF Space created** | ✅ **live** | `lymnal/neuroscope-api`, docker SDK, cpu-basic |
| **`ALLOWED_ORIGINS` set on the Space** | ✅ **set to `*`** | see §1f |
| Screenshot in README | ✅ done | `docs/assets/{hero,bench}.png` |
| **Vercel project created** | ❌ **not done** | §2 below — needs `vercel login` |
| Loom walkthrough | ❌ not done | Ethan |

### §1f — CORS on the live Space

`ALLOWED_ORIGINS` is set to `*`. The API is public, unauthenticated, sets no
cookies, and runs with `allow_credentials=False`, so a browser-origin allowlist
is not the abuse boundary — anything that would abuse the free CPU can do it
with a direct request, which CORS never sees. The upside is that any frontend
origin (preview deploys included) works without re-configuring the Space.

To tighten it to a specific frontend later:

```bash
curl -X POST -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/json" \
  -d '{"key":"ALLOWED_ORIGINS","value":"https://<your-app>.vercel.app,http://localhost:3001"}' \
  https://huggingface.co/api/spaces/lymnal/neuroscope-api/variables
```

Setting a variable restarts the Space. That drops the loaded model, which is
harmless — the frontend loads one on connect (see `checkBackend` in
`src/store/analysisStore.ts`).

---

## 0. Prerequisites

- A HuggingFace account, and **license acceptance on
  [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)**
  if you want the Space to serve Llama at all. GPT-2 needs nothing.
- An HF access token with `read` scope (Settings → Access Tokens).
- A Vercel account linked to the GitHub account that holds this repo.

---

## 1. Backend → HuggingFace Spaces (Docker SDK)

### 1a. Create the Space

1. https://huggingface.co/new-space
2. **SDK: Docker** (blank template). **Hardware:** CPU basic (free). **Public.**
3. Name it `neuroscope-api`. The resulting URL is
   `https://<user>-neuroscope-api.hf.space` — note the **dash**, not a slash;
   this is the value `VITE_API_BASE` needs later.

### 1b. Push the backend

The Space is a git repo whose **root** must be what the Dockerfile sees. Our
Dockerfile lives in `backend/` and does `COPY . .`, so push the *contents* of
`backend/`, not the whole monorepo:

```bash
git clone https://huggingface.co/spaces/<user>/neuroscope-api /tmp/neuroscope-space
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'tests' \
  backend/ /tmp/neuroscope-space/
cd /tmp/neuroscope-space && git add -A && git commit -m "deploy: NeuroScope API" && git push
```

`backend/README.md` already carries the YAML front-matter (`title`, `sdk: docker`,
`app_port`) that the Space card needs — that is why it must be copied too.

> **Do not hand-write a new Dockerfile or requirements.txt.** The pinned set in
> `backend/requirements.txt` includes two constraints that are easy to lose and
> break the build: `transformers>=4.37.2,<5.0` (transformer-lens 2.11 imports
> `transformers.TRANSFORMERS_CACHE`, removed in 5.x) and `huggingface-hub<1.0`.
> An abbreviated requirements list — like the one the old version of this guide
> printed — omits both and the Space fails at import time.

### 1c. Set Variables and secrets

Space → **Settings → Variables and secrets**:

| Key | Kind | Value |
|---|---|---|
| `ALLOWED_ORIGINS` | Variable | `https://<your-vercel-url>,http://localhost:3001` |
| `HF_TOKEN` | **Secret** | your HF read token — only needed for the gated Llama models |

You will not know the Vercel URL until §2. Set `ALLOWED_ORIGINS` to
`http://localhost:3001` now and come back — §2c.

### 1d. Verify

```bash
curl https://<user>-neuroscope-api.hf.space/
```

Expect `{"status":"ok","service":"neuroscope-api"}`. Then smoke-test a load:

```bash
curl -X POST https://<user>-neuroscope-api.hf.space/load \
  -H 'Content-Type: application/json' -d '{"model_name":"gpt2-small"}'
```

First call is 20–40s (download + load). Interactive docs at `/docs`.

### 1e. Known pitfalls

- **Only whitelisted models load.** `ALLOWED_MODELS` in `main.py` is
  `gpt2-small` + the two Llama-3.2 Instruct sizes. Anything else 400s by design.
- **Llama on free CPU is a trap.** Llama-3.2-1B loads in ~16 GB but generation
  is minutes per prompt. The demo path is GPT-2; Llama numbers come from the
  cached bench artifact, not from live Space traffic.
- **`/refusal-bench` will time out on the free tier.** This is why
  `backend/scripts/run_bench_local.py` exists and why the leaderboard ships a
  cached JSON. Do not advertise live re-runs on the deploy.
- **Spaces sleep after ~30 min idle**; first request after that is ~30s.
- **The model cache is inside the container** (`/app/.cache/huggingface`), so it
  re-downloads on every cold start. Fine for a portfolio demo.
- **`TRANSFORMERS_CACHE` is deprecated** in transformers 4.x in favour of
  `HF_HOME`. The Dockerfile sets both; expect a deprecation warning in the log.
  It is noise, not a failure.

---

## 2. Frontend → Vercel

### 2a. Import the project

1. https://vercel.com/new → import this GitHub repo.
2. Framework preset: **Vite** (auto-detected; `vercel.json` also pins it).
3. Root directory: leave blank.
4. Environment variable:
   ```
   VITE_API_BASE=https://<user>-neuroscope-api.hf.space
   ```
   Set it for **Production, Preview, and Development** — a Preview build without
   it silently falls back to `http://localhost:8000` and every call fails.
5. Deploy.

> **Note on the build command.** `vercel.json` runs `vite build`, *not*
> `npm run build` (which is `tsc && vite build`). That is deliberate — a type
> error should not take the deploy down — but it means **Vercel does not
> typecheck**. Typecheck is enforced only by `.github/workflows/ci.yml`. If you
> remove that workflow, nothing checks types before they reach production.

### 2b. Confirm the bench artifact shipped

The leaderboard fetches `/bench/refusal_bench_default.json` as a static asset.
Anything in `public/` is copied verbatim into `dist/`, so:

```bash
curl -sI https://<your-vercel-url>/bench/refusal_bench_default.json | head -1
```

A 404 here means the leaderboard renders empty on the live site while working
locally — the single most likely deploy-day surprise.

### 2c. Go back and fix CORS

Add the real Vercel URL to `ALLOWED_ORIGINS` in the Space settings and restart
the Space. Vercel preview deploys get *per-deployment* URLs which will **not**
match; either add the stable `*.vercel.app` production alias only, or accept
that previews can't reach the backend.

---

## 3. End-to-end check

In a clean/incognito browser:

1. Visit the Vercel URL — page paints, no console errors.
2. The Refusal Bench leaderboard shows rows from the cached artifact.
3. "Load model" → GPT-2 loads (browser-side worker; ~500 MB, first load is slow).
4. A backend-dependent panel (logit lens / attention) returns data — this is the
   real CORS test.
5. Network tab shows requests going to `*.hf.space`, not `localhost`.

---

## 4. Still outstanding after the deploy

- **Screenshot or GIF in the README.** Reviewers skim; a wall of text without a
  visual reads as unfinished. Capture the leaderboard + an attention view.
- **Fill the live URLs into `docs/BLOG_POST_DRAFT.md`** — it has `{{TBD — repo URL}}`
  and `{{TBD — Vercel URL}}` placeholders.
- **Loom walkthrough** (3 min), linked from the README header.

---

## 5. What only Ethan can do

Creating the HF account/Space, accepting the Llama license, creating the Vercel
project, recording the Loom, and writing the blog post. Everything else —
Dockerfile, CORS plumbing, env plumbing, README, CI — is already in the repo.

---

*Last updated: July 29, 2026 — rewritten against the actual state of the repo;
the April 17 version described work that had since been completed.*
