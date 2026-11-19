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


MID = ROOT / "models" / "Qwen3-4B-Q4_K_M.gguf"


@pytest.mark.parametrize("T", [1, 3, 9])
def test_verify_matches_stepwise_decode(model, T):
    greedy = model.generate(PROMPT, T + 1, "cuda-graph")["tokens"].tolist()
    model.logits(PROMPT, "cuda-graph")  # KV for the prompt; verify then places tokens at len(PROMPT)..
    got = model.verify(len(PROMPT), greedy[:T]).tolist()
    assert got == greedy[1 : T + 1]


@pytest.mark.parametrize("k", [1, 3, 8])
def test_self_speculation_is_greedy(model, k):
    # Draft == target: every drafted token is accepted and the output must be plain greedy.
    greedy = model.generate(PROMPT, 24, "cuda-graph")["tokens"].tolist()
    r = model.speculate(model, PROMPT, 24, k)
    assert r["tokens"].tolist() == greedy
    assert r["acceptance"] == 1.0


@pytest.mark.skipif(not MID.exists(), reason="models/Qwen3-4B-Q4_K_M.gguf not present")
@pytest.mark.parametrize("k", [2, 3, 5])
def test_speculative_4b_with_06b_draft_is_greedy(model, k):
    from adakernel.gguf import GgufModel

    with GgufModel(MID, max_seq=128) as target:
        greedy = target.generate(PROMPT, 40, "cuda-graph")["tokens"].tolist()
        r = target.speculate(model, PROMPT, 40, k)
        assert r["tokens"].tolist() == greedy
        assert 0.0 < r["acceptance"] <= 1.0
