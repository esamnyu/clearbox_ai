"""
Refusal-ablation technique implementations.

Each module under this package implements one published technique against
the Technique base class in refusal_bench.technique. Add new techniques
here by importing and registering them in TECHNIQUES below.
"""

from .arditi import Arditi

TECHNIQUES = {
    "arditi": Arditi,
    # Wave 2 to populate:
    #   "wollschlager": Wollschlager,
    #   "cosmic": Cosmic,
    #   "cheng": Cheng,
    #   "maskey": Maskey,
    #   "herring": HerringCNA,
}

__all__ = ["TECHNIQUES", "Arditi"]
