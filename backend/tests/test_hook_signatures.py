"""
Every TransformerLens hook closure must take its second parameter as `hook`.

TransformerLens invokes registered hooks as `fn(tensor, hook=hook_point)` — by
KEYWORD, not position. A closure that names that parameter anything else raises

    TypeError: <fn>() got an unexpected keyword argument 'hook'

...but only at run_bench time, deep inside a multi-minute generation loop on a
real model. That is how the bug shipped twice: Wollschlager's cone hook failed
this way in the May 21 run, and HerringCNA's neuron-zero hook (parameter named
`hook_`) was still failing it in the July 29 run — the single technique keeping
the bench from a complete six-row result.

These tests are static: they parse each module with `ast` and never construct a
technique or load a model, so they run in milliseconds and cover techniques
whose hook factories would otherwise need a fitted model to reach.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Iterator, Tuple

import pytest

import research
from refusal_bench.techniques import TECHNIQUES

BACKEND_ROOT = Path(__file__).resolve().parents[1]

# TransformerLens's calling convention. Not configurable — it is set by
# transformer_lens.hook_points.HookPoint.add_hook.
HOOK_KWARG = "hook"


def _hook_closures(source: str) -> Iterator[Tuple[str, ast.FunctionDef]]:
    """
    Yield (factory_name, nested_function) for every function defined inside a
    hook-factory method — i.e. a def whose name contains "hook" and starts with
    "make_" or "_make_". Those nested defs are the closures handed to
    TransformerLens.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if "hook" not in name.lower():
            continue
        if not (name.startswith("make_") or name.startswith("_make_")):
            continue
        for inner in node.body:
            for sub in ast.walk(inner):
                if isinstance(sub, ast.FunctionDef):
                    yield name, sub


def _technique_sources() -> Iterator[Tuple[str, str]]:
    for key, cls in sorted(TECHNIQUES.items()):
        yield key, inspect.getsource(inspect.getmodule(cls))


@pytest.mark.parametrize("technique_key", sorted(TECHNIQUES))
def test_technique_builds_its_hook_one_of_two_sanctioned_ways(technique_key):
    """
    A technique either delegates to research.make_ablation_hook (arditi, cheng,
    maskey — plain projection removal) or defines its own closure (cosmic,
    herring, wollschlager — cone / neuron-level interventions). Anything else
    means a third hook path appeared that the signature check below does not
    cover, and the bug that path can carry is invisible until a full bench run.
    """
    cls = TECHNIQUES[technique_key]
    source = inspect.getsource(inspect.getmodule(cls))

    delegates = "from research import" in source and "make_ablation_hook" in source
    has_own_closure = bool(list(_hook_closures(source)))

    assert delegates or has_own_closure, (
        f"{technique_key}: builds its ablation hook by some path other than "
        f"research.make_ablation_hook or a _make_*hook* closure. Extend "
        f"_hook_closures() to cover it."
    )


@pytest.mark.parametrize("technique_key", sorted(TECHNIQUES))
def test_technique_hook_closures_accept_hook_keyword(technique_key):
    cls = TECHNIQUES[technique_key]
    source = inspect.getsource(inspect.getmodule(cls))

    # Techniques that delegate to research.make_ablation_hook define no closure
    # of their own; research.py's copy is checked separately below.
    for factory_name, fn in _hook_closures(source):
        args = [a.arg for a in fn.args.args]
        assert len(args) >= 2, (
            f"{technique_key}.{factory_name}.{fn.name} takes {args}; "
            f"TransformerLens always passes (tensor, hook=...)"
        )
        assert args[1] == HOOK_KWARG, (
            f"{technique_key}.{factory_name}.{fn.name} names its second "
            f"parameter {args[1]!r}, but TransformerLens calls hooks as "
            f"fn(tensor, {HOOK_KWARG}=hook_point) — this raises "
            f'"unexpected keyword argument \'{HOOK_KWARG}\'" at run time.'
        )


def test_research_module_hook_closures_accept_hook_keyword():
    """research.py holds the shared ablation + steering hooks used by the UI."""
    source = inspect.getsource(research)
    closures = list(_hook_closures(source))
    assert closures, "research.py: no hook closures found"

    for factory_name, fn in closures:
        args = [a.arg for a in fn.args.args]
        assert args[1:2] == [HOOK_KWARG], (
            f"research.{factory_name}.{fn.name} takes {args}; "
            f"second parameter must be {HOOK_KWARG!r}"
        )


def test_every_technique_module_is_scanned():
    """
    Guards the guard: if a technique is added to the registry, this test file
    must actually reach its source. A silently unscanned module would make the
    parametrized test above pass vacuously.
    """
    for key, source in _technique_sources():
        assert source.strip(), f"{key}: empty source"
        path = BACKEND_ROOT / "refusal_bench" / "techniques" / f"{key}.py"
        # herring's module is herring.py but its class is HerringCNA; the file
        # name matches the registry key for all six.
        assert path.exists(), f"{key}: expected module at {path}"


def test_ablation_hook_is_callable_with_hook_keyword():
    """
    Behavioural counterpart to the static checks: the one hook we can build
    without a model must survive the exact call TransformerLens makes.
    """
    import torch

    direction = torch.zeros(8)
    direction[0] = 1.0
    hook_fn = research.make_ablation_hook(direction)

    act = torch.randn(2, 3, 8)
    out = hook_fn(act.clone(), hook=None)  # keyword, as TransformerLens does
    assert out.shape == act.shape
