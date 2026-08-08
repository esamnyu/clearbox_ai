"""
Logit-lens correctness tests.

The logit lens projects each layer's residual stream through the unembedding.
The recipe MUST apply the final layer norm first:

    logits_L = ln_final(resid_L) @ W_U + b_U

Skipping ln_final (an earlier version did `resid @ W_U`) distorts every
layer's prediction and breaks the one checkable property the technique has:
at the final layer, the lens must reproduce the model's own next-token logits.

These tests run against a tiny randomly-initialized HookedTransformer, so they
exercise the real TransformerLens code paths in ~1s with no weight download —
they belong in the normal suite, not behind a `slow` marker.
"""

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("transformer_lens")

import research  # noqa: E402  (imported after importorskip so collection never hard-fails)


def _synthetic_model():
    """A small, deterministic HookedTransformer with a real ln_final and W_U/b_U."""
    torch.manual_seed(0)
    cfg = tl.HookedTransformerConfig(
        n_layers=3,
        d_model=64,
        n_ctx=32,
        d_head=16,
        n_heads=4,
        d_vocab=100,
        act_fn="gelu",
        normalization_type="LN",
    )
    model = tl.HookedTransformer(cfg)
    model.eval()
    # Config-only models have no tokenizer; the lens only needs id->str for the
    # top-k labels, so a trivial stringifier keeps the test tokenizer-free.
    model.to_string = lambda idx: str(int(idx))
    return model


def _run(model, toks):
    with torch.no_grad():
        logits, cache = model.run_with_cache(toks)
    str_tokens = [str(int(t)) for t in toks[0]]
    return str_tokens, logits, cache


def test_final_layer_reproduces_model_logits(monkeypatch):
    """The lens's last layer must match the model's own argmax next token."""
    model = _synthetic_model()
    toks = torch.randint(0, model.cfg.d_vocab, (1, 8))
    str_tokens, logits, cache = _run(model, toks)

    monkeypatch.setattr(research, "get_model", lambda: model)
    monkeypatch.setattr(research, "run_with_cache", lambda prompt: (str_tokens, logits, cache))

    out = research.logit_lens("unused — run_with_cache is monkeypatched", top_k=5)

    # Embedding row + one row per transformer block.
    assert len(out["predictions"]) == model.cfg.n_layers + 1

    # Ground-truth property: the correct recipe reproduces the model's logits.
    resid = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"][0, -1]
    true_logits = model.ln_final(resid) @ model.W_U + model.b_U
    assert torch.allclose(logits[0, -1], true_logits, atol=1e-4)

    # The function's final-layer top-1 must equal the model's own argmax.
    model_top1 = model.to_string(int(logits[0, -1].argmax()))
    lens_top1 = out["predictions"][-1]["top_k"][0]["token"]
    assert lens_top1 == model_top1

    # And the full top-k the function returns must equal the correct recipe's.
    probs = torch.softmax(true_logits, dim=-1)
    _, top_idx = probs.topk(5)
    expected = [model.to_string(int(i)) for i in top_idx]
    assert [d["token"] for d in out["predictions"][-1]["top_k"]] == expected


def test_skipping_ln_final_is_detectably_wrong():
    """Negative control: the old `resid @ W_U` recipe does NOT match the model.

    This guards against silently dropping ln_final/b_U again — if someone does,
    the previous test's top-k assertion breaks, and this test documents *why*.
    """
    model = _synthetic_model()
    toks = torch.randint(0, model.cfg.d_vocab, (1, 8))
    with torch.no_grad():
        logits, cache = model.run_with_cache(toks)

    resid = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"][0, -1]
    correct = model.ln_final(resid) @ model.W_U + model.b_U
    buggy = resid @ model.W_U  # no ln_final, no b_U

    # Correct recipe is exact; buggy recipe is materially off from the model.
    assert torch.allclose(logits[0, -1], correct, atol=1e-4)
    assert (buggy - logits[0, -1]).abs().max() > 0.1
