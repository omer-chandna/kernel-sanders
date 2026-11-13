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
