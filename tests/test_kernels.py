import numpy as np
import pytest

from adakernel.native import Native


@pytest.fixture(scope="module")
def gpu():
    try:
        return Native()
    except FileNotFoundError as exc:
        pytest.fail(str(exc))


def test_rmsnorm_matches_numpy(gpu):
    rng = np.random.default_rng(1)
    for cols in (1, 7, 17, 32, 127, 256, 1024):
        x = rng.standard_normal((3, cols)).astype(np.float32)
        w = rng.standard_normal(cols).astype(np.float32) * 0.5 + 1
        got = gpu.rmsnorm(x, w, 1e-6)
        var = np.sum(x * x, axis=-1, keepdims=True) / cols
        ref = x * np.reciprocal(np.sqrt(var + 1e-6)) * w
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_residual_and_silu(gpu):
    rng = np.random.default_rng(2)
    a = rng.standard_normal(1000).astype(np.float32)
    b = rng.standard_normal(1000).astype(np.float32)
    np.testing.assert_allclose(gpu.residual(a, b), a + b, rtol=0, atol=0)
    gate = rng.standard_normal(513).astype(np.float32)
    up = rng.standard_normal(513).astype(np.float32)
    ref = gate * (1 / (1 + np.exp(-gate))) * up
    np.testing.assert_allclose(gpu.silu_mul(gate, up), ref, rtol=1e-5, atol=1e-5)


def test_fused_add_rmsnorm(gpu):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(255).astype(np.float32)
    r = rng.standard_normal(255).astype(np.float32)
    w = np.ones(255, dtype=np.float32)
    added, normed = gpu.fused_add_rmsnorm(x, r, w, 1e-6)
    expect_add = x + r
    var = np.sum(expect_add * expect_add) / 255
    expect_norm = expect_add / np.sqrt(var + 1e-6)
    np.testing.assert_allclose(added, expect_add, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(normed, expect_norm, rtol=1e-4, atol=1e-4)


def test_rope_and_qk_norm(gpu):
    rng = np.random.default_rng(4)
    q = rng.standard_normal((4, 32)).astype(np.float32)
    got = gpu.rope(q, 5, 10000.0)
    half = 16
    index = np.arange(half, dtype=np.float32)
    freq = np.exp(-np.log(np.float32(10000)) * (2 * index / 32))
    ang = 5 * freq
    cos, sin = np.cos(ang), np.sin(ang)
    ref = np.empty_like(q)
    ref[:, :half] = q[:, :half] * cos - q[:, half:] * sin
    ref[:, half:] = q[:, half:] * cos + q[:, :half] * sin
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
    w = rng.standard_normal(32).astype(np.float32)
    normed = gpu.qk_norm(q, w, 1e-6)
    var = np.sum(q * q, axis=-1, keepdims=True) / 32
    expect = q * np.reciprocal(np.sqrt(var + 1e-6)) * w
    np.testing.assert_allclose(normed, expect, rtol=1e-4, atol=1e-4)


def test_matvec_odd_sizes(gpu):
    rng = np.random.default_rng(5)
    for rows, k in ((1, 1), (3, 7), (17, 13), (64, 128), (128, 255)):
        w = rng.standard_normal((rows, k)).astype(np.float32)
        x = rng.standard_normal(k).astype(np.float32)
        got = gpu.matvec(w, x)
        ref = w @ x
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
        cublas = gpu.cublas_matvec(w, x)
        np.testing.assert_allclose(cublas, ref, rtol=1e-3, atol=1e-3)


def test_attention_gqa(gpu):
    rng = np.random.default_rng(6)
    q = rng.standard_normal((4, 16)).astype(np.float32)
    k = rng.standard_normal((5, 2, 16)).astype(np.float32)
    v = rng.standard_normal((5, 2, 16)).astype(np.float32)
    got = gpu.attn(q, k, v)
    scale = 1 / np.sqrt(16)
    ref = np.empty_like(q)
    for head in range(4):
        kv = head // 2
        scores = k[:, kv] @ q[head] * scale
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        ref[head] = weights @ v[:, kv]
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
