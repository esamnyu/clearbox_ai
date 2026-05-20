"""
Refusal-ablation technique implementations.

Each module under this package implements one published technique against
the Technique base class in refusal_bench.technique. Add new techniques
here by importing and registering them in TECHNIQUES below.
"""

from .arditi import Arditi
from .cosmic import Cosmic

TECHNIQUES = {
    "arditi": Arditi,
    "cosmic": Cosmic,
    # Wave 2 still to populate:
    #   "wollschlager": Wollschlager,
    #   "cheng": Cheng,
    #   "maskey": Maskey,
    #   "herring": HerringCNA,
}

__all__ = ["TECHNIQUES", "Arditi", "Cosmic"]
