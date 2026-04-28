# Deployment Guide

> Goal: get NeuroScope live at a public URL reviewers can click, in one weekend, at zero cost.
> Target split: **HuggingFace Spaces** (backend, free CPU) + **Vercel** (frontend, free).

---

## Why this split

| Concern | Why HF Spaces | Why Vercel |
|---|---|---|
| Cost | Free CPU tier (16 GB RAM) is sufficient for GPT-2 small | Free tier with custom domain |
| ML ecosystem | Native Docker + PyTorch + HuggingFace models | Not ML-friendly for Python |
| Static + SPA hosting | Overkill | Purpose-built |
| Setup time | ~30 min once your Dockerfile compiles | ~5 min from git repo |

Alternatives if something breaks:
- **Fly.io** for backend (more control, ~$5–15/mo, needs credit card)
- **Modal Labs** for backend (serverless, pay-per-use, harder cold starts)
- **Cloudflare Pages** for frontend (equivalent to Vercel)

---

## 1. Backend → HuggingFace Spaces (Docker SDK)

### 1a. Create the Space

1. Go to https://huggingface.co, sign in, click **New Space**
2. **Owner:** your account. **Name:** `neuroscope-api` (whatever). **SDK:** **Docker**. **Hardware:** CPU basic (free). Public.
3. Clone the empty Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/<you>/neuroscope-api
   ```

### 1b. Prepare the Docker image

Create `Dockerfile` at the Space repo root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV TRANSFORMERS_CACHE=/app/.cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Copy your `backend/*.py` and `backend/requirements.txt` into the Space repo root. Update `requirements.txt` to pin versions for reproducibility:

```
torch==2.3.0
transformer_lens==2.11.0
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
numpy==1.26.4
scikit-learn==1.5.2
```

### 1c. Update CORS for the real frontend URL

In `main.py`, replace the hard-coded localhost origins:

```python
import os

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Set `ALLOWED_ORIGINS` in the Space's **Settings → Variables and secrets**:
```
ALLOWED_ORIGINS=https://neuroscope-<your-username>.vercel.app,http://localhost:3001
```
(update once Vercel gives you the real URL)

### 1d. Push + verify

```bash
git add Dockerfile main.py model.py research.py requirements.txt
git commit -m "feat: dockerize for HF Spaces"
git push
```

Watch the build log in the Space UI. First build is ~5–10 min (PyTorch is heavy). Once "Running," visit:
```
https://<you>-neuroscope-api.hf.space/
```
You should see `{"status":"ok","service":"neuroscope-api"}`.

**Smoke-test the model load:**
```bash
curl -X POST https://<you>-neuroscope-api.hf.space/load \
  -H "Content-Type: application/json" \
  -d '{"model_name":"gpt2-small"}'
```
First call takes 20–40s (model download + load). Subsequent calls return immediately.

### 1e. Known pitfalls

- **OOM on free tier.** GPT-2 small + TransformerLens fits comfortably in 16 GB. If you see OOM, you're probably trying to load `gpt2-medium`. Stay on `gpt2-small` for the demo.
- **Space sleeps after inactivity.** HF Spaces free tier sleeps after ~30 min idle. First request after sleep takes ~30s. Acceptable for a demo; if you want to mitigate, add a note on the frontend: *"Backend is waking up — first request takes ~30s."*
- **Model download is slow.** `TRANSFORMERS_CACHE=/app/.cache` is inside the container; the model re-downloads on every cold start. For a portfolio demo this is fine. To fix longer-term, persist the cache in `/data` (HF Spaces Persistent Storage addon).

---

## 2. Frontend → Vercel

### 2a. Add a backend URL env var

In `src/` wherever you hit the backend, read from an env var instead of hard-coding `localhost:8000`:

```ts
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';
```

Commit and push.

Create `.env.local` for local dev:
```
VITE_API_BASE=http://localhost:8000
```

### 2b. Connect Vercel

1. Go to https://vercel.com, sign in with GitHub.
2. Click **Add New → Project**, import the repo.
3. Framework preset: **Vite** (auto-detected).
4. Root directory: leave blank (`/`).
5. Environment variables → add:
   ```
   VITE_API_BASE=https://<you>-neuroscope-api.hf.space
   ```
6. Deploy.

Vercel gives you a URL like `https://neuroscope-<hash>.vercel.app`. Point-test:
- Load the page.
- Open devtools → Network. Trigger a backend call. Verify it goes to the HF Spaces URL, not localhost.

### 2c. Go back to HF Spaces and update CORS

Add the Vercel URL to `ALLOWED_ORIGINS`. Restart the Space.

---

## 3. Custom domain (optional, ~15 min)

If you own a domain: in Vercel, **Settings → Domains → Add**. Follow the DNS instructions. A domain like `neuroscope.ethansam.dev` reads much better on an application than `neuroscope-a8k2zq.vercel.app`.

Don't buy a domain for this unless you already have one. Not worth the overhead.

---

## 4. End-to-end check

Clean browser (incognito / different browser):
1. Visit the Vercel URL.
2. Page loads without console errors.
3. Click "Load model" → wait for success.
4. Enter a prompt. See tokens, activations, visualizations.
5. No CORS errors in console.

If any of these fails, fix before moving on. Reviewers will use a clean browser.

---

## 5. README rewrite (60-second skim)

Apply this structure to `README.md`. The old structure optimizes for contributors; the new one optimizes for reviewers landing from your application.

```markdown
# NeuroScope

> **A browser-native mechanistic interpretability toolkit.** Load GPT-2, extract activations, visualize attention, inject steering vectors, and ablate directions — all from a single deployed URL.

**[Live demo](https://neuroscope.vercel.app)** · **[3-min walkthrough (Loom)](<loom-url>)** · **[Blog post: <title>](<blog-url>)**

![screenshot or GIF](docs/screenshot.png)

## What's inside

- **Frontend:** Vite + React + TypeScript, transformers.js in a Web Worker for in-browser inference.
- **Backend:** FastAPI + TransformerLens for server-side activation analysis (logit lens, attention patterns, gradients, steering, PCA).
- **One research finding so far:** <1-sentence summary of blog-post finding>. Full writeup linked above.

## Research context

I'm building this toolkit while teaching myself mechanistic interpretability. The field's current state — including where the single-direction refusal story has been superseded (Wollschlager 2025, Zhao 2025) and where it holds up — is summarized in [`docs/RESEARCH_LANDSCAPE_2026.md`](docs/RESEARCH_LANDSCAPE_2026.md).

## Run locally

```bash
# Frontend
npm install
npm run dev    # → http://localhost:3001

# Backend (separate terminal)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Project layout

| Path | Purpose |
|---|---|
| `src/engine/` | Web Worker — transformers.js inference |
| `src/analysis/` | Pure functions on extracted tensors |
| `backend/research.py` | TransformerLens research functions |
| `docs/` | Architecture + research landscape + testing strategy |

## Stack

Vite · React 18 · TypeScript strict · transformers.js · FastAPI · TransformerLens · Zustand · Tailwind · Radix

## Status

<One sentence about the current state. Be honest.>
```

**Things to cut from the current README** (optimizing for a stranger landing via application link):
- Session 1 checkpoints (internal-only)
- "For Researchers / For Engineers" audience split (assumes multiple contributors)
- Phase 1/2/3 roadmap with emojis (reads as incomplete; aspirational roadmaps hurt portfolios)
- References section at the bottom (fine but move to bottom)

**Things to keep and polish:**
- Live demo link (add if missing)
- Screenshot / GIF (add if missing — critical)
- Blog post link (add after Sunday)
- Stack list (reviewers skim for tech legibility)

---

## 6. What Ethan has to do himself

- Create HuggingFace account + Space
- Create Vercel account
- Own whatever domain name (optional)
- Provide the Loom recording
- Write the blog post

Everything else — Dockerfile, CORS patching, env var plumbing, README rewrite — is code I can do with you.

---

*Last updated: April 17, 2026.*
