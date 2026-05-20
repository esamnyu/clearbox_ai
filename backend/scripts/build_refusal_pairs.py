"""
Populate backend/refusal_pairs.py from JailbreakBench + Alpaca.

Run once, locally, by the researcher:

    cd backend && python scripts/build_refusal_pairs.py [--n 50]

This is human-in-the-loop intentionally — automated curation of jailbreak
prompts hits safety filters, and the standard research workflow for
refusal-direction work (Arditi 2024 onward) sources prompts from
published datasets with documented provenance.

The script:
    1. Pulls JBB-Behaviors `behaviors.csv` from HuggingFace.
    2. Pulls a slice of Alpaca instruction-only entries.
    3. Length-matches on the Llama-3.2-1B-Instruct tokenizer (within 20%).
    4. Rewrites REFUSAL_PAIRS in backend/refusal_pairs.py.

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
from typing import List, Tuple

try:
    from datasets import load_dataset
except ImportError:
    print("error: install datasets (pip install datasets)", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
REFUSAL_PAIRS_PATH = REPO_ROOT / "backend" / "refusal_pairs.py"

LLAMA_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B-Instruct"
FALLBACK_TOKENIZER_NAME = "gpt2"  # if Llama is gated and no HF_TOKEN
LENGTH_TOLERANCE = 0.20  # accept pairs within 20% token-count delta


def load_jailbreakbench(n: int) -> List[str]:
    """Pull JBB-Behaviors harmful behaviors. Returns plain prompt strings."""
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    prompts = [row["Goal"] for row in ds]
    random.shuffle(prompts)
    return prompts[: n * 2]  # over-pull, we'll filter after length-matching


def load_alpaca(n: int) -> List[str]:
    """Pull Alpaca instruction-only entries."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = [
        row["instruction"]
        for row in ds
        if not row.get("input", "").strip()
    ]
    random.shuffle(prompts)
    return prompts[: n * 4]  # heavily over-pull to give length-matching room


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


def length_match(
    harmful: List[str],
    harmless: List[str],
    tokenizer,
    n_target: int,
) -> List[Tuple[str, str]]:
    """
    Greedy length-matching: for each harmful prompt, find an unused harmless
    prompt whose token count is within LENGTH_TOLERANCE.
    """
    def tok_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    harmful_with_len = [(p, tok_len(p)) for p in harmful]
    harmless_with_len = [(p, tok_len(p)) for p in harmless]

    used_harmless = set()
    pairs: List[Tuple[str, str]] = []

    for hp, hl in harmful_with_len:
        if len(pairs) >= n_target:
            break
        best = None
        best_delta = float("inf")
        for j, (sp, sl) in enumerate(harmless_with_len):
            if j in used_harmless:
                continue
            delta = abs(sl - hl) / max(hl, 1)
            if delta < best_delta:
                best_delta = delta
                best = j
        if best is not None and best_delta <= LENGTH_TOLERANCE:
            pairs.append((hp, harmless_with_len[best][0]))
            used_harmless.add(best)

    return pairs


def rewrite_refusal_pairs_file(pairs: List[Tuple[str, str]]) -> None:
    """Overwrite REFUSAL_PAIRS in backend/refusal_pairs.py."""
    lines = ["REFUSAL_PAIRS: List[Tuple[str, str]] = ["]
    for harmful, harmless in pairs:
        h = repr(harmful)
        s = repr(harmless)
        lines.append(f"    ({h}, {s}),")
    lines.append("]")
    new_block = "\n".join(lines)

    src = REFUSAL_PAIRS_PATH.read_text()
    # Replace existing REFUSAL_PAIRS literal (greedy from declaration to ']\n')
    import re
    pattern = re.compile(
        r"REFUSAL_PAIRS: List\[Tuple\[str, str\]\] = \[.*?\n\]",
        flags=re.DOTALL,
    )
    if not pattern.search(src):
        raise RuntimeError(
            "couldn't find REFUSAL_PAIRS literal in refusal_pairs.py — "
            "file structure may have changed"
        )
    REFUSAL_PAIRS_PATH.write_text(pattern.sub(new_block, src))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="target pair count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"pulling JailbreakBench (over-pull for length matching)…")
    harmful = load_jailbreakbench(args.n)
    print(f"  got {len(harmful)} harmful prompts")

    print(f"pulling Alpaca instruction-only…")
    harmless = load_alpaca(args.n)
    print(f"  got {len(harmless)} harmless prompts")

    print(f"loading tokenizer…")
    tokenizer = get_tokenizer()

    print(f"length-matching (tolerance {LENGTH_TOLERANCE*100:.0f}%)…")
    pairs = length_match(harmful, harmless, tokenizer, args.n)
    print(f"  matched {len(pairs)} pairs out of target {args.n}")
    if len(pairs) < args.n // 2:
        print(
            "warning: matched fewer than n/2 pairs. "
            "Consider relaxing LENGTH_TOLERANCE or over-pulling more.",
            file=sys.stderr,
        )

    print(f"writing {REFUSAL_PAIRS_PATH}…")
    rewrite_refusal_pairs_file(pairs)
    print(f"done. {len(pairs)} pairs written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
