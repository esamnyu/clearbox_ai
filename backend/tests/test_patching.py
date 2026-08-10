"""
Activation-patching tests.

Two tiers, both offline:

- Pure-math unit tests for the metric helpers (no torch model).
- Synthetic-model tests that drive `patch_grid` against a tiny randomly-
  initialized HookedTransformer (~1s, no weight download) and pin an EXACT
  correctness anchor: patching resid_post at the final layer + final position
  fully determines the final-position logits, so that cell must reproduce the
  source run's logit-diff.

The string-level wrapper `run_activation_patching` needs a tokenizer, so its
end-to-end test loads gpt2-small and is gated behind NEUROSCOPE_RUN_MODEL_TESTS,
matching test_determinism.py.
"""

import os

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("transformer_lens")

import patching  # noqa: E402


# -----------------------------------------------------------------------------
# Pure-math unit tests — no model
# -----------------------------------------------------------------------------

def test_logit_diff_is_answer_minus_baseline():
    logits = torch.tensor([1.0, 4.0, 2.0])
    assert patching.logit_diff(logits, answer_id=1, baseline_id=0) == pytest.approx(3.0)
    assert patching.logit_diff(logits, answer_id=0, baseline_id=1) == pytest.approx(-3.0)


def test_normalized_recovery_endpoints_and_overshoot():
    # base=2, source=10: patched at base -> 0, at source -> 1, midpoint -> 0.5.
    assert patching.normalized_recovery(2.0, 2.0, 10.0) == pytest.approx(0.0)
    assert patching.normalized_recovery(10.0, 2.0, 10.0) == pytest.approx(1.0)
    assert patching.normalized_recovery(6.0, 2.0, 10.0) == pytest.approx(0.5)
    # NOT clamped: overshoot and backfire are visible on purpose.
    assert patching.normalized_recovery(14.0, 2.0, 10.0) == pytest.approx(1.5)
    assert patching.normalized_recovery(0.0, 2.0, 10.0) == pytest.approx(-0.25)


def test_normalized_recovery_undefined_when_runs_agree():
    assert patching.normalized_recovery(5.0, 3.0, 3.0) is None


def test_kl_divergence_zero_iff_identical():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.0, 0.0, 1.0])
    assert patching.kl_divergence(a, a) == pytest.approx(0.0, abs=1e-6)
    assert patching.kl_divergence(a, b) > 0.0


# -----------------------------------------------------------------------------
# Synthetic-model tests — real TransformerLens code paths, no download
# -----------------------------------------------------------------------------

def _synthetic_model():
    torch.manual_seed(0)
    cfg = tl.HookedTransformerConfig(
        n_layers=3, d_model=64, n_ctx=32, d_head=16, n_heads=4,
        d_vocab=100, act_fn="gelu", normalization_type="LN",
    )
    model = tl.HookedTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def synthetic(monkeypatch):
    model = _synthetic_model()
    monkeypatch.setattr(patching, "get_model", lambda: model)
    return model


def _distinct_prompt_pair(model, seq_len=6):
    """Two same-shape token sequences that differ, with two distinct answer ids."""
    torch.manual_seed(1)
    base = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
    source = base.clone()
    source[0, seq_len // 2] = (base[0, seq_len // 2] + 1) % model.cfg.d_vocab
    return base, source


def test_final_layer_final_position_patch_reproduces_source(synthetic):
    """
    Exact anchor: resid_post at the last layer + last position fully determines
    the final-position logits (ln_final @ W_U + b_U). Patching it from source
    into a base run must make that cell's logit_diff equal the source run's,
    i.e. normalized recovery == 1.
    """
    model = synthetic
    base, source = _distinct_prompt_pair(model)
    last_layer = model.cfg.n_layers - 1
    last_pos = base.shape[1] - 1

    out = patching.patch_grid(
        base, source, answer_id=7, baseline_id=3,
        component="resid_post", layers=[last_layer], positions=[last_pos],
    )
    cell = out["grid"][0]["cells"][0]
    assert cell["position"] == last_pos
    assert cell["logit_diff"] == pytest.approx(out["source_logit_diff"], abs=1e-4)
    assert cell["normalized"] == pytest.approx(1.0, abs=1e-4)
    assert cell["kl_to_source"] == pytest.approx(0.0, abs=1e-5)


def test_grid_dimensions_and_full_default_sweep(synthetic):
    model = synthetic
    base, source = _distinct_prompt_pair(model)
    out = patching.patch_grid(base, source, answer_id=7, baseline_id=3)
    assert out["component"] == "resid_post"
    assert len(out["grid"]) == model.cfg.n_layers
    assert out["positions"] == list(range(base.shape[1]))
    for row in out["grid"]:
        assert len(row["cells"]) == base.shape[1]


def test_head_component_grid_is_keyed_by_head(synthetic):
    model = synthetic
    base, source = _distinct_prompt_pair(model)
    out = patching.patch_grid(
        base, source, answer_id=7, baseline_id=3,
        component="head_z", layers=[0, 1],
    )
    assert out["heads"] == list(range(model.cfg.n_heads))
    assert len(out["grid"]) == 2
    for row in out["grid"]:
        assert [c["head"] for c in row["cells"]] == list(range(model.cfg.n_heads))


def test_negative_position_indexes_from_end(synthetic):
    model = synthetic
    base, source = _distinct_prompt_pair(model)
    out = patching.patch_grid(
        base, source, answer_id=7, baseline_id=3,
        layers=[0], positions=[-1],
    )
    assert out["grid"][0]["cells"][0]["position"] == base.shape[1] - 1


def test_shape_mismatch_raises(synthetic):
    model = synthetic
    base = torch.randint(0, model.cfg.d_vocab, (1, 6))
    source = torch.randint(0, model.cfg.d_vocab, (1, 7))
    with pytest.raises(ValueError, match="same shape"):
        patching.patch_grid(base, source, answer_id=7, baseline_id=3)


def test_out_of_range_and_cap_raise(synthetic):
    model = synthetic
    base, source = _distinct_prompt_pair(model)
    with pytest.raises(ValueError, match="layer .* out of range"):
        patching.patch_grid(base, source, 7, 3, layers=[99])
    with pytest.raises(ValueError, match="position .* out of range"):
        patching.patch_grid(base, source, 7, 3, layers=[0], positions=[99])
    with pytest.raises(ValueError, match="head .* out of range"):
        patching.patch_grid(base, source, 7, 3, component="head_z", layers=[0], heads=[99])
    with pytest.raises(ValueError, match="exceeds the cap"):
        patching.patch_grid(base, source, 7, 3, max_runs=1)


def test_invalid_component_and_direction_raise(synthetic):
    model = synthetic
    base, source = _distinct_prompt_pair(model)
    with pytest.raises(ValueError, match="component must be"):
        patching.patch_grid(base, source, 7, 3, component="nonsense")
    with pytest.raises(ValueError, match="direction must be"):
        patching.run_activation_patching("a", "b", "c", "d", direction="sideways")


def test_patch_grid_is_deterministic(synthetic):
    base, source = _distinct_prompt_pair(synthetic)
    first = patching.patch_grid(base, source, 7, 3, layers=[0, 2], positions=[0, -1])
    second = patching.patch_grid(base, source, 7, 3, layers=[0, 2], positions=[0, -1])
    assert first == second


# -----------------------------------------------------------------------------
# String wrapper with a faked tokenizer — direction semantics without a download
# -----------------------------------------------------------------------------

def _with_fake_tokenizer(model, clean_tokens, corrupted_tokens):
    """Give a config-only model just enough tokenizer surface for the wrapper."""
    vocab = {"A": 7, "B": 3, "A-alias": 7}
    model.to_tokens = lambda s: {"CLEAN": clean_tokens, "CORR": corrupted_tokens}[s]
    model.to_str_tokens = lambda t, prepend_bos=True: [str(int(x)) for x in t]
    model.to_single_token = lambda s: vocab[s]


def test_wrapper_direction_semantics(synthetic):
    """
    denoising: base = corrupted run, source = clean activations.
    noising:   base = clean run, source = corrupted activations.
    In both, the (last layer, last position) anchor must fully reach the
    source run's value (normalized == 1), and the clean/corrupted logit-diff
    labels must be assigned from the same runs regardless of direction.
    """
    model = synthetic
    clean_tokens, corrupted_tokens = _distinct_prompt_pair(model)
    _with_fake_tokenizer(model, clean_tokens, corrupted_tokens)
    last_layer = model.cfg.n_layers - 1

    den = patching.run_activation_patching(
        "CLEAN", "CORR", "A", "B",
        direction="denoising", layers=[last_layer], positions=[-1],
    )
    assert den["base_logit_diff"] == den["corrupted_logit_diff"]
    assert den["source_logit_diff"] == den["clean_logit_diff"]
    assert den["grid"][0]["cells"][0]["normalized"] == pytest.approx(1.0, abs=1e-4)

    noi = patching.run_activation_patching(
        "CLEAN", "CORR", "A", "B",
        direction="noising", layers=[last_layer], positions=[-1],
    )
    assert noi["base_logit_diff"] == noi["clean_logit_diff"]
    assert noi["source_logit_diff"] == noi["corrupted_logit_diff"]
    assert noi["grid"][0]["cells"][0]["normalized"] == pytest.approx(1.0, abs=1e-4)

    # The metric itself is direction-independent: same clean/corrupted values.
    assert den["clean_logit_diff"] == pytest.approx(noi["clean_logit_diff"], abs=1e-6)
    assert den["corrupted_logit_diff"] == pytest.approx(noi["corrupted_logit_diff"], abs=1e-6)


def test_wrapper_same_answer_token_raises(synthetic):
    model = synthetic
    clean_tokens, corrupted_tokens = _distinct_prompt_pair(model)
    _with_fake_tokenizer(model, clean_tokens, corrupted_tokens)
    with pytest.raises(ValueError, match="same token id"):
        patching.run_activation_patching(
            "CLEAN", "CORR", "A", "A-alias", layers=[0], positions=[0],
        )


def test_wrapper_notes_swapped_answers(synthetic):
    """Whichever answer orientation makes clean <= corrupted must carry a note."""
    model = synthetic
    clean_tokens, corrupted_tokens = _distinct_prompt_pair(model)
    _with_fake_tokenizer(model, clean_tokens, corrupted_tokens)

    kwargs = dict(direction="denoising", layers=[0], positions=[0])
    forward = patching.run_activation_patching("CLEAN", "CORR", "A", "B", **kwargs)
    swapped = patching.run_activation_patching("CLEAN", "CORR", "B", "A", **kwargs)

    # Swapping answers negates the metric, so exactly one orientation is
    # "well-formed" (unless the diff is exactly zero, excluded by seed choice).
    assert forward["clean_logit_diff"] != pytest.approx(forward["corrupted_logit_diff"])
    flagged = [r for r in (forward, swapped) if r["notes"]]
    assert len(flagged) == 1
    assert "swapped" in flagged[0]["notes"][0]


# -----------------------------------------------------------------------------
# End-to-end string wrapper — needs a real tokenizer, gated like test_determinism
# -----------------------------------------------------------------------------

def _load_gpt2_cpu(monkeypatch):
    """Load gpt2-small on CPU explicitly — the repo-documented MPS caveats
    (TransformerLens numeric warnings, ~11 GB RSS) apply to local runs."""
    import model as model_mod

    if (
        model_mod._model is None
        or model_mod._model_name != "gpt2-small"
        or str(model_mod._model.cfg.device) != "cpu"
    ):
        monkeypatch.setattr(model_mod, "get_device", lambda: "cpu")
        monkeypatch.setattr(model_mod, "_model", None)
        model_mod.load_model("gpt2-small")


@pytest.mark.skipif(
    os.environ.get("NEUROSCOPE_RUN_MODEL_TESTS") != "1",
    reason="loads gpt2-small; set NEUROSCOPE_RUN_MODEL_TESTS=1 to run",
)
def test_run_activation_patching_end_to_end_gpt2(monkeypatch):
    _load_gpt2_cpu(monkeypatch)

    # A classic IOI-style pair: same length, differ in the subject token.
    out = patching.run_activation_patching(
        clean_prompt="When John and Mary went to the store, John gave a drink to",
        corrupted_prompt="When John and Mary went to the store, Mary gave a drink to",
        clean_answer=" Mary",
        corrupted_answer=" John",
        direction="denoising",
        component="resid_post",
    )
    # Well-formed pair: clean favors the clean answer over the corrupted one.
    assert out["clean_logit_diff"] > out["corrupted_logit_diff"]
    assert out["notes"] == []
    # Patching the last layer's final position from clean into the corrupted run
    # recovers essentially all of the clean logit-diff.
    last = out["grid"][-1]["cells"][-1]
    assert last["normalized"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.skipif(
    os.environ.get("NEUROSCOPE_RUN_MODEL_TESTS") != "1",
    reason="loads gpt2-small; set NEUROSCOPE_RUN_MODEL_TESTS=1 to run",
)
def test_run_activation_patching_rejects_multitoken_answer_gpt2(monkeypatch):
    _load_gpt2_cpu(monkeypatch)
    with pytest.raises(ValueError, match="not a single token"):
        patching.run_activation_patching(
            clean_prompt="The capital is",
            corrupted_prompt="The capital was",
            clean_answer=" Reykjavik",  # multi-token in GPT-2
            corrupted_answer=" Paris",
        )
