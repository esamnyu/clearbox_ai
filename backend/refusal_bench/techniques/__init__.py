"""
Refusal-ablation technique implementations.

Each module under this package implements one published technique against
the Technique base class in refusal_bench.technique. Add new techniques
here by importing and registering them in TECHNIQUES below.
"""

from .arditi import Arditi
from .cheng import Cheng
from .cosmic import Cosmic
<<<<<<< HEAD
from .herring import HerringCNA
=======
from .maskey import Maskey
>>>>>>> 522ed35 (feat(bench): add Maskey decomposition technique + over-refusal scaffold)
from .wollschlager import Wollschlager

TECHNIQUES = {
    "arditi": Arditi,
    "cheng": Cheng,
    "cosmic": Cosmic,
<<<<<<< HEAD
    "herring": HerringCNA,
    "wollschlager": Wollschlager,
    # Wave 2 still to populate:
    #   "maskey": Maskey,
}

__all__ = ["TECHNIQUES", "Arditi", "Cheng", "Cosmic", "HerringCNA", "Wollschlager"]
=======
    "maskey": Maskey,
    "wollschlager": Wollschlager,
    # Wave 2 still to populate:
    #   "herring": HerringCNA,
}

__all__ = ["TECHNIQUES", "Arditi", "Cheng", "Cosmic", "Maskey", "Wollschlager"]
>>>>>>> 522ed35 (feat(bench): add Maskey decomposition technique + over-refusal scaffold)
