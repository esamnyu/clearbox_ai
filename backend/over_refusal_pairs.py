"""
Over-refusal contrastive prompts.

`OVER_REFUSAL_PROMPTS` is a list of single prompt strings (not pairs) that
trigger over-cautious refusal in instruction-tuned chat models despite
being entirely benign in intent. Used by Maskey-style decomposition to
extract the over-refusal direction separately from the genuinely-harmful
direction, then subtract its component out.

Provenance:
- XSTest (https://github.com/paul-rottger/exaggerated-safety, paper
  https://arxiv.org/abs/2308.01263) — Roettger et al. NAACL 2024,
  "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in
  Large Language Models." 250 safe prompts grouped into 10 categories
  (homonyms of unsafe terms, figurative language, safe targets in unsafe
  contexts, privacy-public, etc.), specifically constructed to surface
  over-refusal. License: CC-BY 4.0. Standard dataset for over-refusal
  evaluation in the 2024-2026 alignment literature.
- Harmless baseline (the third set the Maskey decomposition needs) is
  shared with refusal_pairs.py: Alpaca instruction-only entries. The
  Maskey class composes harmless prompts from get_refusal_pairs() and
  over-refusal prompts from this file.

Population: this file is populated by
`backend/scripts/build_over_refusal_pairs.py`. The empty list below is
intentional — keep human-in-the-loop curation as part of the research
workflow, mirroring backend/refusal_pairs.py. Run the script once
locally; it overwrites this file with a populated list.

Length-matching note: unlike refusal_pairs (paired contrastive),
over-refusal prompts are used UNPAIRED — the mean is taken over the set
and subtracted against the harmless mean. The build script still
length-filters to keep last-token-position statistics comparable across
the three sets, but does not enforce per-prompt alignment.
"""

from typing import List


OVER_REFUSAL_PROMPTS: List[str] = [
    # Run `python backend/scripts/build_over_refusal_pairs.py` to populate.
    # Until then this list is empty and Maskey.set_over_refusal() will raise
    # if invoked with the contents of get_over_refusal_pairs().
]


def get_over_refusal_pairs() -> List[str]:
    """
    Return curated XSTest-style over-refusal prompts.

    Despite the name ("…_pairs" for parity with get_refusal_pairs), each
    element is a single prompt string, not a tuple — the Maskey
    decomposition uses these against the shared harmless baseline, not
    as pairs.
    """
    return OVER_REFUSAL_PROMPTS
