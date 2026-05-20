"""
Refusal contrastive prompt pairs.

`REFUSAL_PAIRS` is a list of (harmful_prompt, harmless_prompt) tuples used
to extract the refusal direction via difference-of-means (Arditi 2024).

Provenance:
- Harmful prompts: JailbreakBench (https://github.com/JailbreakBench/JBB-Behaviors)
  — MIT-licensed academic benchmark of harmful behaviors. Standard set used
  by Arditi 2024, Wollschlager 2025, Cheng 2026, et al.
- Harmless prompts: Alpaca (https://huggingface.co/datasets/tatsu-lab/alpaca)
  — CC BY-NC 4.0. Instruction-only subset, length-matched to the harmful set.
- Length-matching: per-pair token-count delta within 20% on the
  Llama-3.2-1B-Instruct tokenizer.

Population: this file is populated by `backend/scripts/build_refusal_pairs.py`.
The empty list below is intentional — keep human-in-the-loop curation as part
of the research workflow. Run the script once locally; it overwrites this
file with a populated list.
"""

from typing import List, Tuple


REFUSAL_PAIRS: List[Tuple[str, str]] = [
    # Run `python backend/scripts/build_refusal_pairs.py` to populate.
    # Until then, the /refusal-pairs endpoint returns count=0 and the
    # refusal-direction pipeline will refuse to run (validated in main.py).
]


def get_refusal_pairs() -> List[Tuple[str, str]]:
    """Return curated (harmful, harmless) prompt pairs."""
    return REFUSAL_PAIRS
