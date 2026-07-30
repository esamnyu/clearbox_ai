"""
Pytest configuration for the backend test suite.

- Puts the backend root on sys.path so tests can `import research`,
  `from refusal_bench... import ...`, etc.
- Stubs the heavy `model` module (TransformerLens) ONLY when transformer_lens
  is not importable, so the pure-logic unit tests (probe CV, ablation hook,
  cosine diagnostic, scoring) run in a minimal environment. When
  transformer_lens IS present, the real `model` module is used and the gated
  live-model tests (test_determinism.py) can run.
"""

import os
import sys
import types
from unittest.mock import MagicMock

BACKEND_ROOT = os.path.dirname(__file__)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

try:
    import transformer_lens  # noqa: F401

    _HAS_TL = True
except Exception:
    _HAS_TL = False

if not _HAS_TL and "model" not in sys.modules:
    _stub = types.ModuleType("model")
    for _attr in (
        "get_model",
        "get_model_name",
        "run_with_cache",
        "load_model",
        "get_device",
    ):
        setattr(_stub, _attr, MagicMock(name=_attr))
    sys.modules["model"] = _stub
