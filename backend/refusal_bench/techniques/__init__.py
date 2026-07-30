"""
Refusal-ablation technique implementations.

Each module under this package implements one published technique against
the Technique base class in refusal_bench.technique. Add new techniques
here by importing and registering them in TECHNIQUES below.

Six techniques in scope (alphabetical):
    arditi       — single direction (NeurIPS 2024 baseline)
    cheng        — compressed direction (top-k by magnitude)
    cosmic       — cosine-objective layer + direction selection
    herring      — contrastive neuron ablation (MLP-level)
    maskey       — over-refusal vs harmful-refusal decomposition
    wollschlager — polyhedral cone (rank-k iterative orthogonal)
"""

from .arditi import Arditi
from .cheng import Cheng
from .cosmic import Cosmic
from .herring import HerringCNA
from .maskey import Maskey
from .wollschlager import Wollschlager

TECHNIQUES = {
    "arditi": Arditi,
    "cheng": Cheng,
    "cosmic": Cosmic,
    "herring": HerringCNA,
    "maskey": Maskey,
    "wollschlager": Wollschlager,
}

__all__ = [
    "TECHNIQUES",
    "Arditi",
    "Cheng",
    "Cosmic",
    "HerringCNA",
    "Maskey",
    "Wollschlager",
]
