import numpy as np
import pytest

from adakernel.native import Native


@pytest.fixture(scope="module")
def gpu():
    try:
        return Native()
    except FileNotFoundError as exc:
        pytest.fail(str(exc))


def test_block_sizes(gpu):
    assert gpu.lib.ak_q8_block_size() == 34
    assert gpu.lib.ak_q4k_block_size() == 144
    assert gpu.lib.ak_q6k_block_size() == 210


def test_q8_roundtrip_and_matvec(gpu):
    rng = np.random.default_rng(7)
    rows, k = 4, 64
    weight = rng.standard_normal((rows, k)).astype(np.float32)
    packed = gpu.q8_quantize(weight)
    deq = gpu.q8_dequantize(packed, rows * k).reshape(rows, k)
    assert np.max(np.abs(deq - weight)) < 0.02
    vector = rng.standard_normal(k).astype(np.float32)
    got = gpu.q8_matvec(packed, vector, rows, k)
    ref = deq @ vector
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_q4k_known_block_and_matvec(gpu):
    block = np.zeros(144, dtype=np.uint8)
    block[0:2] = np.array([0x00, 0x3C], dtype=np.uint8)  # fp16 1.0
    block[4:8] = 1
    block[8:12] = 1
    values = gpu.q4k_dequantize(block, 256)
    # Groups 0..3 use low 6 bits of scales 0..3 (=1) and mins 4..7 (=0).
    # Groups 4..7 use the low nibble of scales 8..11 (=1).
    expect = np.zeros(256, dtype=np.float32)
    np.testing.assert_allclose(values, expect, atol=0)
    rng = np.random.default_rng(8)
    rows, k = 3, 256
    weight = rng.standard_normal((rows, k)).astype(np.float32)
    packed = gpu.q4k_quantize(weight)
    deq = gpu.q4k_dequantize(packed, rows * k).reshape(rows, k)
    vector = rng.standard_normal(k).astype(np.float32)
    got = gpu.q4k_matvec(packed, vector, rows, k)
    np.testing.assert_allclose(got, deq @ vector, rtol=1e-4, atol=1e-4)
    assert np.max(np.abs(deq)) > 0


def _q6k_dequantize(gpu, packed, n):
    raw = np.ascontiguousarray(packed, dtype=np.uint8)
    out = np.empty(n, dtype=np.float32)
    gpu.check(gpu.lib.ak_q6k_dequantize(gpu._ptr(raw), gpu._ptr(out), n))
    return out


def _random_q6k(rng, rows, k):
    blocks = rng.integers(0, 256, size=(rows * (k // 256), 210), dtype=np.uint8)
    # Keep d finite and small: fp16 0.01 is 0x211F.
    blocks[:, 208] = 0x1F
    blocks[:, 209] = 0x21
    return blocks.reshape(-1)


@pytest.mark.parametrize("qtype", ["q8_0", "q4_k", "q6_k"])
@pytest.mark.parametrize("rows,k", [(5, 256), (37, 1024), (64, 4096)])
def test_qgemv_dp4a(gpu, qtype, rows, k):
    rng = np.random.default_rng(rows * 1000 + k)
    vector = rng.standard_normal(k).astype(np.float32)
    vector[::97] *= 15.0  # outliers, so some blocks have a much larger scale than others
    if qtype == "q8_0":
        packed = gpu.q8_quantize(rng.standard_normal((rows, k)).astype(np.float32))
        deq = gpu.q8_dequantize(packed, rows * k).reshape(rows, k)
    elif qtype == "q4_k":
        packed = gpu.q4k_quantize(rng.standard_normal((rows, k)).astype(np.float32))
        deq = gpu.q4k_dequantize(packed, rows * k).reshape(rows, k)
    else:
        packed = _random_q6k(rng, rows, k)
        deq = _q6k_dequantize(gpu, packed, rows * k).reshape(rows, k)
    y, xq, xd = gpu.qgemv_dp4a(qtype, packed, vector, rows, k)

    # The activation quantizer is amax / 127 per 32 values, round to nearest even.
    blocks = vector.reshape(-1, 32)
    d_ref = (np.abs(blocks).max(axis=1) / np.float32(127)).astype(np.float32)
    np.testing.assert_array_equal(xd, d_ref)
    q_ref = np.rint(blocks / d_ref[:, None]).astype(np.int8).reshape(-1)
    np.testing.assert_array_equal(xq, q_ref)

    # Kernel arithmetic against the exact dequantized activation.
    x_int8 = (xq.reshape(-1, 32).astype(np.float64) * xd[:, None]).reshape(-1)
    exact = deq.astype(np.float64) @ x_int8
    scale = np.abs(exact).max()
    assert np.max(np.abs(y - exact)) <= 1e-5 * scale + 1e-6

    # End to end against the float activation: int8 rounding only.
    full = deq.astype(np.float64) @ vector.astype(np.float64)
    rel = np.linalg.norm(y - full) / np.linalg.norm(full)
    assert rel < 2e-2
