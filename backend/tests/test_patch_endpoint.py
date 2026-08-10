"""
/patch endpoint contract tests.

Injects a tiny config-only HookedTransformer (with a faked tokenizer surface)
into the model singleton, then exercises the endpoint through TestClient: the
happy path, and each error path's status-code mapping (400 for research-level
ValueErrors/RuntimeErrors, 422 for per-model layer bounds).
"""

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("transformer_lens")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import model as model_mod  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)


def _synthetic_model():
    torch.manual_seed(0)
    cfg = tl.HookedTransformerConfig(
        n_layers=3, d_model=64, n_ctx=32, d_head=16, n_heads=4,
        d_vocab=100, act_fn="gelu", normalization_type="LN",
    )
    m = tl.HookedTransformer(cfg)
    m.eval()

    torch.manual_seed(1)
    clean = torch.randint(0, cfg.d_vocab, (1, 6))
    corrupted = clean.clone()
    corrupted[0, 3] = (clean[0, 3] + 1) % cfg.d_vocab
    short = torch.randint(0, cfg.d_vocab, (1, 4))

    vocab = {" A": 7, " B": 3}
    m.to_tokens = lambda s: {"CLEAN": clean, "CORR": corrupted, "SHORT": short}[s]
    m.to_str_tokens = lambda t, prepend_bos=True: [str(int(x)) for x in t]
    m.to_single_token = lambda s: vocab[s]
    return m


@pytest.fixture
def loaded_model(monkeypatch):
    m = _synthetic_model()
    monkeypatch.setattr(model_mod, "_model", m)
    monkeypatch.setattr(model_mod, "_model_name", "synthetic-test")
    return m


def _payload(**overrides):
    payload = {
        "clean_prompt": "CLEAN",
        "corrupted_prompt": "CORR",
        "clean_answer": " A",
        "corrupted_answer": " B",
        "layers": [2],
        "positions": [-1],
    }
    payload.update(overrides)
    return payload


def test_patch_happy_path(loaded_model):
    resp = client.post("/patch", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["direction"] == "denoising"
    assert body["component"] == "resid_post"
    cell = body["grid"][0]["cells"][0]
    # (last layer, last position) anchor: full recovery of the source value.
    assert abs(cell["normalized"] - 1.0) < 1e-3
    assert cell["kl_to_source"] < 1e-5


def test_patch_head_component(loaded_model):
    resp = client.post(
        "/patch", json=_payload(component="head_z", positions=None, heads=[0, 2])
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["heads"] == [0, 2]
    assert [c["head"] for c in body["grid"][0]["cells"]] == [0, 2]


def test_patch_no_model_loaded_is_400(monkeypatch):
    monkeypatch.setattr(model_mod, "_model", None)
    monkeypatch.setattr(model_mod, "_model_name", None)
    resp = client.post("/patch", json=_payload())
    assert resp.status_code == 400
    assert "No model loaded" in resp.json()["detail"]


def test_patch_length_mismatch_is_400(loaded_model):
    resp = client.post("/patch", json=_payload(corrupted_prompt="SHORT"))
    assert resp.status_code == 400
    assert "same shape" in resp.json()["detail"]


def test_patch_layer_out_of_range_is_422(loaded_model):
    resp = client.post("/patch", json=_payload(layers=[99]))
    assert resp.status_code == 422
    assert "n_layers" in resp.json()["detail"]


def test_patch_bad_direction_is_422_at_schema(loaded_model):
    # Literal["denoising", "noising"] — pydantic rejects before our code runs.
    resp = client.post("/patch", json=_payload(direction="sideways"))
    assert resp.status_code == 422
