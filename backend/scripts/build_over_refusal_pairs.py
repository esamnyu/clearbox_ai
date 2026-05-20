"""
Populate backend/over_refusal_pairs.py from XSTest.

Run once, locally, by the researcher:

    cd backend && python scripts/build_over_refusal_pairs.py [--n 50]

XSTest (Roettger et al., NAACL 2024, https://arxiv.org/abs/2308.01263) is
the standard dataset for over-refusal evaluation. It contains 250
hand-crafted benign prompts whose surface lexical cues (kill, shoot,
attack, hack, etc.) reliably trigger over-cautious refusal on
instruction-tuned chat models. The 200 "safe" prompts are exactly what
Maskey's decomposition needs to isolate the over-refusal direction; the
"unsafe" counterparts (used in the original benchmark for paired
contrast) are NOT included here — Maskey uses unpaired over-refusal
prompts against a harmless baseline (Alpaca), not paired against
genuinely-harmful ones.

The script:
    1. Pulls XSTest via HuggingFace `natolambert/xstest-v2-copy` (the
       community-mirrored version; the original repo is hand-curated
       JSON at github.com/paul-rottger/exaggerated-safety).
    2. Filters to the "safe" subset.
    3. Length-filters on the Llama-3.2-1B-Instruct tokenizer so that
       last-token-residual statistics are comparable to refusal_pairs.
    4. Rewrites OVER_REFUSAL_PROMPTS in backend/over_refusal_pairs.py.

Requires: `pip install datasets transformers` (already in requirements.txt).
For Llama tokenizer access: export HF_TOKEN with a token that has
accepted Meta's Llama-3.2 license.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List

try:
    from datasets import load_dataset
except ImportError:
    print("error: install datasets (pip install datasets)", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
OVER_REFUSAL_PAIRS_PATH = REPO_ROOT / "backend" / "over_refusal_pairs.py"

XSTEST_DATASET_NAME = "natolambert/xstest-v2-copy"
LLAMA_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B-Instruct"
FALLBACK_TOKENIZER_NAME = "gpt2"  # if Llama is gated and no HF_TOKEN

# XSTest prompts are short by design (one sentence each). Use looser absolute
# bounds rather than the relative-tolerance length-matching that pairs need —
# we want a representative spread across XSTest's 10 over-refusal categories.
MIN_TOKENS = 4
MAX_TOKENS = 50


def load_xstest(n: int) -> List[str]:
    """
    Pull XSTest safe prompts. Returns plain prompt strings.

    XSTest v2 schema (verified against `natolambert/xstest-v2-copy`):
      - "type": category name. Safe variants use bare names like
        "homonyms", "figurative_language", "safe_targets", etc.
        Unsafe variants are prefixed "contrast_", e.g. "contrast_homonyms".
      - "prompt": the user-facing text.
    We keep only the non-contrast (safe) rows — these are the
    benign-but-edgy prompts that trigger over-refusal in safety-tuned
    chat models.
    """
    ds = load_dataset(XSTEST_DATASET_NAME, split="prompts")
    if "type" not in ds.column_names or "prompt" not in ds.column_names:
        raise RuntimeError(
            f"xstest dataset missing expected columns; saw: {ds.column_names}"
        )

    prompts: List[str] = []
    for row in ds:
        type_label = str(row["type"])
        # safe variants don't carry the "contrast_" prefix
        if type_label.startswith("contrast_"):
            continue
        text = row["prompt"]
        if text:
            prompts.append(text.strip())

    if not prompts:
        raise RuntimeError(
            "xstest: no safe prompts extracted; dataset schema may have changed"
        )

    random.shuffle(prompts)
    return prompts[: n * 3]  # over-pull to give length-filtering room


def get_tokenizer():
    """Try Llama first; fall back to gpt2 if gated and no token."""
    try:
        from transformers import AutoTokenizer
        token = os.environ.get("HF_TOKEN")
        if token:
            return AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_NAME, token=token)
        return AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_NAME)
    except Exception as e:
        print(
            f"warning: couldn't load {LLAMA_TOKENIZER_NAME} ({e}); "
            f"falling back to {FALLBACK_TOKENIZER_NAME}",
            file=sys.stderr,
        )
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER_NAME)


def length_filter(prompts: List[str], tokenizer, n_target: int) -> List[str]:
    """
    Absolute-bounds length filter. XSTest prompts are short by design; we
    just clip the very-short ("kill") and very-long edge cases so the
    last-token-residual statistics are stable.
    """
    def tok_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    kept: List[str] = []
    for p in prompts:
        n_tokens = tok_len(p)
        if MIN_TOKENS <= n_tokens <= MAX_TOKENS:
            kept.append(p)
        if len(kept) >= n_target:
            break
    return kept


def rewrite_over_refusal_pairs_file(prompts: List[str]) -> None:
    """Overwrite OVER_REFUSAL_PROMPTS in backend/over_refusal_pairs.py."""
    lines = ["OVER_REFUSAL_PROMPTS: List[str] = ["]
    for p in prompts:
        lines.append(f"    {p!r},")
    lines.append("]")
    new_block = "\n".join(lines)

    src = OVER_REFUSAL_PAIRS_PATH.read_text()
    # Replace existing OVER_REFUSAL_PROMPTS literal (greedy from declaration to ']\n')
    import re
    pattern = re.compile(
        r"OVER_REFUSAL_PROMPTS: List\[str\] = \[.*?\n\]",
        flags=re.DOTALL,
    )
    if not pattern.search(src):
        raise RuntimeError(
            "couldn't find OVER_REFUSAL_PROMPTS literal in over_refusal_pairs.py — "
            "file structure may have changed"
        )
    OVER_REFUSAL_PAIRS_PATH.write_text(pattern.sub(new_block, src))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="target prompt count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"pulling XSTest (over-pull for length filtering)…")
    candidates = load_xstest(args.n)
    print(f"  got {len(candidates)} safe prompts")

    print(f"loading tokenizer…")
    tokenizer = get_tokenizer()

    print(f"length-filtering ({MIN_TOKENS}..{MAX_TOKENS} tokens)…")
    prompts = length_filter(candidates, tokenizer, args.n)
    print(f"  kept {len(prompts)} prompts out of target {args.n}")
    if len(prompts) < args.n // 2:
        print(
            "warning: kept fewer than n/2 prompts. "
            "Consider relaxing length bounds or over-pulling more.",
            file=sys.stderr,
        )

    print(f"writing {OVER_REFUSAL_PAIRS_PATH}…")
    rewrite_over_refusal_pairs_file(prompts)
    print(f"done. {len(prompts)} prompts written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
