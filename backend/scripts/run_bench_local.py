"""
Run the Refusal Bench against a locally-loaded Llama-3.2-1B-Instruct.

Why this exists: HF Spaces free CPU tier can't complete heavier
techniques (Wollschlager cone, COSMIC layer sweep, Cheng, Maskey,
Herring) — they OOM or exceed the proxy timeout. This script bypasses
HF entirely and runs the same bench end-to-end on Apple Silicon (MPS)
or CPU.

Output mirrors what /refusal-bench would return:
  - One JSON per technique in docs/bench_partials_local/<name>.json
  - A combined docs/bench_result_local_6tech.json with all rows

Run:
    cd backend && .venv/bin/python scripts/run_bench_local.py [--n 20]

Memory: BF16 Llama-3.2-1B is ~2.5GB; close browsers before running on
a 16GB machine. Each technique is run independently so partial results
survive an OOM mid-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make backend's modules importable
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import torch

# Surface the HF token. The huggingface-cli login flow writes the active
# token to ~/.cache/huggingface/token; set HF_TOKEN env so model.py's
# _ensure_hf_login picks it up.
TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"
if "HF_TOKEN" not in os.environ and TOKEN_PATH.exists():
    os.environ["HF_TOKEN"] = TOKEN_PATH.read_text().strip()


def _json_safe(obj):
    """
    Recursively replace non-finite floats (NaN / ±Inf) with None.

    Python's json.dumps emits a bare `NaN` / `Infinity` token, which is invalid
    JSON that browser JSON.parse rejects — a single errored or degenerate row
    would otherwise make the whole leaderboard file unparseable and blank the
    UI (re-introducing the exact empty-state bug the cached artifact fixes).
    """
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="pair count per class")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--techniques",
        type=str,
        default="arditi,wollschlager,cosmic,cheng,maskey,herring",
        help="comma-separated names",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / mps / cpu; defaults to auto-detect",
    )
    args = parser.parse_args()

    # Load model
    import model as model_mod
    import refusal_pairs
    import over_refusal_pairs
    from refusal_bench.runner import run_bench, serialize

    if args.device:
        # Optional override; otherwise model.get_device() picks MPS on Apple Silicon
        original_get_device = model_mod.get_device
        model_mod.get_device = lambda: args.device  # type: ignore

    print(f"[bench-local] device: {model_mod.get_device()}")
    print(f"[bench-local] torch: {torch.__version__}")
    print(f"[bench-local] MPS available: {torch.backends.mps.is_available()}")

    print(f"\n[bench-local] loading Llama-3.2-1B-Instruct (BF16)…")
    t0 = time.time()
    # Use float16 instead of float32 to fit in 16GB unified memory
    # Note: HookedTransformer.from_pretrained does not directly accept dtype
    # in transformer_lens 2.11; load and convert.
    info = model_mod.load_model("meta-llama/Llama-3.2-1B-Instruct")
    print(f"[bench-local] loaded in {time.time()-t0:.1f}s: {info}")

    # Cast to bfloat16 for memory headroom on MPS
    m = model_mod.get_model()
    target_dtype = torch.float16 if model_mod.get_device() == "mps" else torch.bfloat16
    print(f"[bench-local] casting model to {target_dtype}…")
    m = m.to(target_dtype)
    # Patch the singleton so subsequent get_model() returns the cast version
    model_mod._model = m  # type: ignore

    pairs = refusal_pairs.get_refusal_pairs()
    if len(pairs) < args.n:
        print(f"[bench-local] WARNING: only {len(pairs)} pairs available; capping --n to that.")
        n = len(pairs)
    else:
        n = args.n
    harmful = [p[0] for p in pairs[:n]]
    harmless = [p[1] for p in pairs[:n]]

    # Over-refusal prompts (XSTest) for Maskey decomposition. Other
    # techniques ignore this list. Falls back to empty if the module is
    # unpopulated (run backend/scripts/build_over_refusal_pairs.py first).
    over_refusal = over_refusal_pairs.OVER_REFUSAL_PROMPTS[:n] if hasattr(over_refusal_pairs, "OVER_REFUSAL_PROMPTS") else []
    print(f"[bench-local] using {n} pairs per class · {len(over_refusal)} over-refusal prompts")

    out_dir = BACKEND.parent / "docs" / "bench_partials_local"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = BACKEND.parent / "docs" / "bench_result_local_6tech.json"

    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]
    all_rows = []
    probe_train_auc = None
    probe_test_auc = None
    probe_cv_auc_mean = None
    probe_cv_auc_std = None
    n_extraction_pairs = None
    n_eval_prompts = None

    for tname in techniques:
        print(f"\n[bench-local] ── {tname} ──", flush=True)
        t0 = time.time()
        try:
            result = run_bench(
                technique_names=[tname],
                layer=args.layer,
                harmful_prompts=harmful,
                harmless_prompts=harmless,
                over_refusal_prompts=over_refusal if over_refusal else None,
                test_fraction=0.25,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                seed=args.seed,
            )
            elapsed = time.time() - t0
            row = result.results[0]
            all_rows.append(row.__dict__ if hasattr(row, "__dict__") else row)
            if probe_train_auc is None:
                probe_train_auc = result.probe_train_auc
                probe_test_auc = result.probe_test_auc
                probe_cv_auc_mean = result.probe_cv_auc_mean
                probe_cv_auc_std = result.probe_cv_auc_std
                n_extraction_pairs = result.n_extraction_pairs
                n_eval_prompts = result.n_eval_prompts

            (out_dir / f"{tname}.json").write_text(json.dumps(serialize(result), indent=2))
            if row.error:
                print(f"  ERROR: {row.error[:120]}")
            else:
                print(
                    f"  Δrr={row.delta_refusal_rate:+.3f}  ΔAUC={row.delta_auc:+.3f}  "
                    f"({elapsed/60:.1f} min)"
                )
        except Exception as e:
            print(f"  EXC: {type(e).__name__}: {str(e)[:200]}")
            all_rows.append({
                "name": tname,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_seconds": time.time() - t0,
            })

        # Free GPU memory between techniques
        if model_mod.get_device() == "mps":
            torch.mps.empty_cache()

    # Combine
    combined = {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "device": model_mod.get_device(),
        "dtype": str(target_dtype),
        "layer": args.layer,
        "n_pairs_per_class": n,
        "test_fraction": 0.25,
        "probe_train_auc": probe_train_auc,
        "probe_test_auc": probe_test_auc,
        "results": all_rows,
    }
    combined_path.write_text(json.dumps(combined, indent=2, default=str))

    # UI-shaped artifact for the leaderboard (public/bench/). NaN-sanitized so a
    # single errored/degenerate row can't make the file invalid JSON and blank
    # the leaderboard. Overwrites the cached default with this fresh local run.
    ui_artifact = {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "layer": args.layer,
        "n_extraction_pairs": n_extraction_pairs,
        "n_eval_prompts": n_eval_prompts,
        "probe_train_auc": probe_train_auc,
        "probe_test_auc": probe_test_auc,
        "probe_cv_auc_mean": probe_cv_auc_mean,
        "probe_cv_auc_std": probe_cv_auc_std,
        "results": all_rows,
    }
    public_dir = BACKEND.parent / "public" / "bench"
    public_dir.mkdir(parents=True, exist_ok=True)
    public_path = public_dir / "refusal_bench_default.json"
    public_path.write_text(json.dumps(_json_safe(ui_artifact), indent=2))
    print(f"saved UI artifact: {public_path}")

    # Print summary
    print(f"\n\n=== REFUSAL BENCH — Llama-3.2-1B (local, {target_dtype}) ===")
    print(f"probe train AUC {probe_train_auc:.3f}, test AUC {probe_test_auc:.3f}")
    print()
    print(f"{'TECHNIQUE':<28} {'Δ REFUSAL':>10} {'Δ AUC':>10}")
    print("-" * 50)
    for row in all_rows:
        name = row.get("name", "?")[:28]
        if row.get("error"):
            print(f"{name:<28}  ERROR")
        else:
            drr = row.get("delta_refusal_rate", float("nan"))
            dauc = row.get("delta_auc", float("nan"))
            print(f"{name:<28} {drr:>+10.3f} {dauc:>+10.3f}")
    print(f"\nsaved: {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
