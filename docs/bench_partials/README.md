# Bench partials — provenance

Raw output from the May 21 2026 HTTP-based bench run (script driving the live backend over HTTP). These files **pre-date** the determinism fix (WS2), CV-AUC (WS3), and the wollschlager/maskey fixes in `57f13de`.

Caveats:

- Some files (`bench_result_6tech_20pairs.json`, `wollschlager.json`, `maskey.json`) contain bare `NaN` tokens from Python's `json.dump` — they are **not strict JSON** (`JSON.parse` rejects them; Python's `json.load` accepts).
- `arditi.json` and the combined `bench_result_6tech_20pairs.json` come from **different runs with different numbers** (e.g. baseline refusal rate 0.8 vs 0.6).
- Numbers here are historical context only — **not reproducible by current code**.

Current local-run output goes to `docs/bench_partials_local/` via `backend/scripts/run_bench_local.py`.
