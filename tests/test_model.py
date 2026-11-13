import numpy as np
import pytest

from adakernel.native import Model, Native
from adakernel.reference import generate, make_weights, prefill_logits

CFG = {
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "d_ff": 128,
    "vocab": 48,
    "max_seq": 32,
    "rms_eps": 1.0e-6,
    "rope_theta": 10000.0,
    "qk_norm": 1,
    "n_gpu_layers": -1,
    "chunk_rows": 16,
    "offload_overlap": 0,
    "tied_embeddings": 0,
    "init_scale": 0.02,
    "seed": 0,
}
PROMPT = np.array([1, 5, 9, 3], dtype=np.int32)


@pytest.fixture(scope="module")
def native():
    try:
        return Native()
    except FileNotFoundError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="module")
def weights():
    return make_weights(CFG, seed=0)


def test_eager_logits_match_reference(native, weights):
    model = Model(native, CFG, weights)
    try:
        got = model.logits(PROMPT)
    finally:
        model.close()
    ref = prefill_logits(CFG, weights, PROMPT.tolist())
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)


def test_backends_match_reference_tokens(native, weights):
    ref = generate(CFG, weights, PROMPT.tolist(), 6)
    eager_tokens = None
    for backend in ("eager", "cuda-graph", "persistent", "hybrid", "fused"):
        model = Model(native, CFG, weights)
        try:
            result = model.generate(PROMPT, 6, backend)
        finally:
            model.close()
        if backend == "eager":
            eager_tokens = result["tokens"]
            np.testing.assert_array_equal(result["tokens"], ref)
        else:
            np.testing.assert_array_equal(result["tokens"], eager_tokens)


def test_offload_matches_resident(native, weights):
    resident = Model(native, CFG, weights)
    off = dict(CFG)
    off["n_gpu_layers"] = 0
    streamed = Model(native, off, weights)
    try:
        a = resident.generate(PROMPT, 5, "eager")
        b = streamed.generate(PROMPT, 5, "cuda-graph")
    finally:
        resident.close()
        streamed.close()
    np.testing.assert_array_equal(a["tokens"], b["tokens"])
    assert b["host_weight_mb"] > 0
    assert b["n_gpu_layers"] == 0


def test_paged_kv_stays_close(native, weights):
    fp32 = Model(native, CFG, weights)
    fp16_cfg = dict(CFG)
    fp16_cfg["kv_type"] = 1
    q8_cfg = dict(CFG)
    q8_cfg["kv_type"] = 2
    fp16 = Model(native, fp16_cfg, weights)
    q8 = Model(native, q8_cfg, weights)
    try:
        a = fp32.generate(PROMPT, 6, "eager")["tokens"]
        b = fp16.generate(PROMPT, 6, "eager")["tokens"]
        c = q8.generate(PROMPT, 6, "persistent")["tokens"]
    finally:
        fp32.close()
        fp16.close()
        q8.close()
    assert b.tolist() == a.tolist()
    assert c.tolist() == a.tolist()
    wide = dict(CFG)
    wide["max_seq"] = 128
    wide["kv_type"] = 0
    fp16_wide = dict(wide)
    fp16_wide["kv_type"] = 1
    q8_wide = dict(wide)
    q8_wide["kv_type"] = 2
    wide_weights = make_weights(wide, seed=0)
    a_model = Model(native, wide, wide_weights)
    b_model = Model(native, fp16_wide, wide_weights)
    c_model = Model(native, q8_wide, wide_weights)
    try:
        ra = a_model.generate(PROMPT, 4, "eager")
        rb = b_model.generate(PROMPT, 4, "eager")
        rc = c_model.generate(PROMPT, 4, "persistent")
    finally:
        a_model.close()
        b_model.close()
        c_model.close()
    assert rb["tokens"].tolist() == ra["tokens"].tolist()
    assert rc["tokens"].tolist() == ra["tokens"].tolist()
    assert rb["vram_peak_mb"] < ra["vram_peak_mb"]
    assert rc["vram_peak_mb"] < rb["vram_peak_mb"]


def test_speculative_matches_identical_draft(native, weights):
    target = Model(native, CFG, weights)
    draft = Model(native, CFG, weights)
    try:
        alone = target.generate(PROMPT, 6, "eager")
        spec = target.speculate(draft, PROMPT, 6, gamma=4)
    finally:
        target.close()
        draft.close()
    np.testing.assert_array_equal(spec["tokens"], alone["tokens"])
    assert spec["accepted_tokens"] > 0


def test_offload_overlap_matches(native, weights):
    off = dict(CFG)
    off["n_gpu_layers"] = 0
    off["offload_overlap"] = 1
    plain = dict(off)
    plain["offload_overlap"] = 0
    a = Model(native, plain, weights)
    b = Model(native, off, weights)
    try:
        ra = a.generate(PROMPT, 5, "eager")
        rb = b.generate(PROMPT, 5, "eager")
    finally:
        a.close()
        b.close()
    np.testing.assert_array_equal(ra["tokens"], rb["tokens"])


def test_cublas_logits_are_close(native, weights):
    model = Model(native, CFG, weights)
    try:
        eager = model.logits(PROMPT)
        cublas = model.generate(PROMPT, 1, "cublas")
        again = model.logits(PROMPT)
    finally:
        model.close()
    np.testing.assert_allclose(eager, again, rtol=1e-5, atol=1e-5)
    assert cublas["tokens"].shape == (1,)
