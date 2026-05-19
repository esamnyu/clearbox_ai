---
title: NeuroScope API
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# NeuroScope API

FastAPI + TransformerLens backend for the NeuroScope interpretability toolkit.

Exposes activation-extraction endpoints (logit lens, attention patterns, gradient-based token importance, steering vectors, PCA trajectories) for the NeuroScope frontend.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /load` | Load a HuggingFace model (default: `gpt2-small`) |
| `POST /logit-lens` | Layer-by-layer next-token predictions |
| `POST /attention` | Attention pattern for a given (layer, head) |
| `POST /gradients` | Token-level gradient magnitudes w.r.t. a target token |
| `POST /steering-vector` | Difference-of-means vector from contrastive prompts |
| `POST /generate-steered` | Generation with a steering vector injected at a layer |
| `POST /ablate-direction` | Generation with a direction projected out of the residual stream (`h' = h − (h·d̂)d̂`) |
| `POST /pca-trajectories` | 3D PCA of residual stream across layers + tokens |
| `GET /contrastive-pairs` | Built-in sentiment contrastive prompt pairs |

Interactive docs at `/docs` once the Space is live.

## Configuration

Set the following Space Variable/Secret in **Settings → Variables and secrets**:

- `ALLOWED_ORIGINS` — comma-separated CORS origins (e.g. `https://neuroscope.vercel.app,http://localhost:3001`)

## Local development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend repo: [clearbox_ai on GitHub](https://github.com/ethan-sam/clearbox_ai).
