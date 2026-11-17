"""End-to-end checks of the Qwen3 GGUF runtime. Skipped when the model files are not present."""

from pathlib import Path

import numpy as np
import pytest

from adakernel import reference as R

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"
# "The capital of France is" in the Qwen3 tokenizer; fixed so the test does not need llama-tokenize.
PROMPT = [785, 6722, 315, 9625, 374]

pytestmark = pytest.mark.skipif(not SMALL.exists(), reason="models/Qwen3-0.6B-Q8_0.gguf not present")


@pytest.fixture(scope="module")
def model():
    from adakernel.gguf import GgufModel

    m = GgufModel(SMALL, max_seq=64)
    yield m
    m.close()


def test_loads_qwen3_shape(model):
    info = model.info
    assert info["arch"] == "qwen3"
    assert (info["n_layers"], info["d_model"], info["n_heads"], info["n_kv_heads"], info["head_dim"]) == (28, 1024, 16, 8, 128)
    assert info["tied"] == 1
    assert info["vocab"] == 151936


def test_backends_agree(model):
    runs = {be: model.generate(PROMPT, 16, be)["tokens"].tolist() for be in ("eager", "cuda-graph", "persistent", "hybrid")}
    first = runs["eager"]
    for be, toks in runs.items():
        assert toks == first, be
    # Qwen3-0.6B answers this prompt with " Paris" (token 12095).
    assert first[0] == 12095


def test_logits_match_numpy_reference(model):
    from adakernel.gguf import reference_weights

    cfg, w = reference_weights(SMALL, model.native)
    cfg["max_seq"] = 16
    ref = R.prefill_logits(cfg, w, PROMPT).astype(np.float64)
    for be in ("eager", "hybrid"):
        ours = model.logits(PROMPT, be).astype(np.float64)
        # int8 activations (same scheme as llama.cpp) put the error in the low percent range.
        rel = np.linalg.norm(ours - ref) / np.linalg.norm(ref)
        assert rel < 5e-2, (be, rel)
        assert int(np.argmax(ours)) == int(np.argmax(ref))
