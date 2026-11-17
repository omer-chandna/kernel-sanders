#pragma once

// Bandwidth-oriented quantized GEMV for batch-1 decode.
//
// The activation vector is quantized once to Q8_1-style blocks (32 int8 values plus one float scale).
// Weight blocks are then dotted with __dp4a, four int8 products per instruction, instead of decoding every
// weight to float. Each warp owns one output row, and its lanes split a row's super-blocks so that one warp
// load covers contiguous weight bytes.
//
// Layout of x after quantization:  xq[K] int8, xd[K / 32] float, x[i] ~= xd[i / 32] * xq[i].
// xq must be 16-byte aligned. Q4_K rows must start 16-byte aligned (GGUF tensors are 32-aligned and 144 is a
// multiple of 16). Q6_K and Q8_0 blocks are only 2-byte aligned and use 16-bit loads.

#include <cuda_fp16.h>
#include <cstdint>

namespace akd {

__device__ __forceinline__ float qg_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float qg_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float qg_half(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// Little-endian 32-bit word from a 2-byte-aligned address.
__device__ __forceinline__ int qg_ld_b2(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return static_cast<int>(static_cast<uint32_t>(h[0]) | (static_cast<uint32_t>(h[1]) << 16));
}

// Block-wide: quantize x[0..K) into xq / xd. K must be a multiple of 32.
__device__ __forceinline__ void quantize_q8_1_n(const float* x, int8_t* xq, float* xd, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    for (int base = warp * 32; base < K; base += nwarps * 32) {
        const float v = x[base + lane];
        const float amax = qg_warp_max(fabsf(v));
        const float d = amax / 127.f;
        const int q = amax > 0.f ? __float2int_rn(v / d) : 0;
        xq[base + lane] = static_cast<int8_t>(q);
        if (lane == 0) xd[base >> 5] = d;
    }
    __syncthreads();
}

// Grid-wide quantize: one warp per 32-value block. Launch with ceil(K / 32 / warps_per_block) blocks.
static __global__ void quantize_q8_1_grid(const float* x, int8_t* xq, float* xd, int K) {
    const int lane = threadIdx.x & 31;
    const int base = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 32;
    if (base >= K) return;
    const float v = x[base + lane];
    const float amax = qg_warp_max(fabsf(v));
    const float d = amax / 127.f;
    const int q = amax > 0.f ? __float2int_rn(v / d) : 0;
    xq[base + lane] = static_cast<int8_t>(q);
    if (lane == 0) xd[base >> 5] = d;
}

// ---- Q4_K: 144-byte super-block, 256 weights. 8 lanes per super-block, 16 weight bytes per lane. ----
__device__ __forceinline__ float q4k_dp4a_sb(const uint8_t* blk, const int8_t* xq, const float* xd, int t) {
    const int j = t >> 1;           // 64-weight group
    const int off = (t & 1) << 4;   // first or second 16 bytes of the group's 32 bytes
    const int4 w = *reinterpret_cast<const int4*>(blk + 16 + (t << 4));
    const int4 xa = *reinterpret_cast<const int4*>(xq + j * 64 + off);
    const int4 xb = *reinterpret_cast<const int4*>(xq + j * 64 + 32 + off);
    const int ones = 0x01010101;
    int dl = 0, dh = 0, sl = 0, sh = 0;
    dl = __dp4a(w.x & 0x0F0F0F0F, xa.x, dl);
    dl = __dp4a(w.y & 0x0F0F0F0F, xa.y, dl);
    dl = __dp4a(w.z & 0x0F0F0F0F, xa.z, dl);
    dl = __dp4a(w.w & 0x0F0F0F0F, xa.w, dl);
    dh = __dp4a((w.x >> 4) & 0x0F0F0F0F, xb.x, dh);
    dh = __dp4a((w.y >> 4) & 0x0F0F0F0F, xb.y, dh);
    dh = __dp4a((w.z >> 4) & 0x0F0F0F0F, xb.z, dh);
    dh = __dp4a((w.w >> 4) & 0x0F0F0F0F, xb.w, dh);
    sl = __dp4a(ones, xa.x, sl);
    sl = __dp4a(ones, xa.y, sl);
    sl = __dp4a(ones, xa.z, sl);
    sl = __dp4a(ones, xa.w, sl);
    sh = __dp4a(ones, xb.x, sh);
    sh = __dp4a(ones, xb.y, sh);
    sh = __dp4a(ones, xb.z, sh);
    sh = __dp4a(ones, xb.w, sh);

    // 6-bit scales and mins for sub-blocks 2j and 2j+1 (same packing as get_scale_min_k4).
    const uint8_t* s = blk + 4;
    int sc0, m0, sc1, m1;
    const int is = 2 * j;
    if (is < 4) {
        sc0 = s[is] & 63;
        m0 = s[is + 4] & 63;
        sc1 = s[is + 1] & 63;
        m1 = s[is + 5] & 63;
    } else {
        sc0 = (s[is + 4] & 0xF) | ((s[is - 4] >> 6) << 4);
        m0 = (s[is + 4] >> 4) | ((s[is] >> 6) << 4);
        sc1 = (s[is + 5] & 0xF) | ((s[is - 3] >> 6) << 4);
        m1 = (s[is + 5] >> 4) | ((s[is + 1] >> 6) << 4);
    }
    const float2 dm = __half22float2(*reinterpret_cast<const __half2*>(blk));
    const float x0 = xd[is];
    const float x1 = xd[is + 1];
    return dm.x * (x0 * static_cast<float>(sc0 * dl) + x1 * static_cast<float>(sc1 * dh)) -
           dm.y * (x0 * static_cast<float>(m0 * sl) + x1 * static_cast<float>(m1 * sh));
}

__device__ __forceinline__ float q4k_dp4a_row(const uint8_t* rowp, const int8_t* xq, const float* xd, int nb, int lane) {
    const int t = lane & 7;
    float acc = 0.f;
#pragma unroll 4
    for (int b = lane >> 3; b < nb; b += 4) {
        acc += q4k_dp4a_sb(rowp + b * 144, xq + b * 256, xd + b * 8, t);
    }
    return qg_warp_sum(acc);
}

// ---- Q6_K: 210-byte super-block (ql[128], qh[64], scales[16], d). 8 lanes per super-block. ----
__device__ __forceinline__ float q6k_dp4a_sb(const uint8_t* blk, const int8_t* xq, const float* xd, int t) {
    const int h = t >> 2;            // 128-weight half
    const int l0 = (t & 3) << 3;     // 8 consecutive l values
    const uint8_t* ql = blk + h * 64 + l0;
    const uint8_t* qh = blk + 128 + h * 32 + l0;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + h * 8 + (l0 >> 4);
    const int8_t* xb = xq + h * 128 + l0;
    const float* xdb = xd + h * 4;
    const int ones = 0x01010101;
    int dot[4] = {0, 0, 0, 0};
    int sum[4] = {0, 0, 0, 0};
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const int a = qg_ld_b2(ql + 4 * w);
        const int b = qg_ld_b2(ql + 32 + 4 * w);
        const int c = qg_ld_b2(qh + 4 * w);
        const int q[4] = {
            (a & 0x0F0F0F0F) | ((c << 4) & 0x30303030),
            (b & 0x0F0F0F0F) | ((c << 2) & 0x30303030),
            ((a >> 4) & 0x0F0F0F0F) | (c & 0x30303030),
            ((b >> 4) & 0x0F0F0F0F) | ((c >> 2) & 0x30303030),
        };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int xv = *reinterpret_cast<const int*>(xb + k * 32 + 4 * w);
            dot[k] = __dp4a(q[k], xv, dot[k]);
            sum[k] = __dp4a(ones, xv, sum[k]);
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc += xdb[k] * static_cast<float>(sc[2 * k]) * static_cast<float>(dot[k] - 32 * sum[k]);
    }
    return qg_half(blk + 208) * acc;
}

__device__ __forceinline__ float q6k_dp4a_row(const uint8_t* rowp, const int8_t* xq, const float* xd, int nb, int lane) {
    const int t = lane & 7;
    float acc = 0.f;
#pragma unroll 4
    for (int b = lane >> 3; b < nb; b += 4) {
        acc += q6k_dp4a_sb(rowp + b * 210, xq + b * 256, xd + b * 8, t);
    }
    return qg_warp_sum(acc);
}

// ---- Q8_0: 34-byte block (d, 32 x int8). 4 lanes per block, 8 weights per lane. ----
__device__ __forceinline__ float q8_dp4a_row(const uint8_t* rowp, const int8_t* xq, const float* xd, int nb, int lane) {
    const int t = lane & 3;
    float acc = 0.f;
#pragma unroll 4
    for (int b = lane >> 2; b < nb; b += 8) {
        const uint8_t* blk = rowp + b * 34;
        const int8_t* xb = xq + b * 32 + t * 8;
        int dot = __dp4a(qg_ld_b2(blk + 2 + t * 8), *reinterpret_cast<const int*>(xb), 0);
        dot = __dp4a(qg_ld_b2(blk + 6 + t * 8), *reinterpret_cast<const int*>(xb + 4), dot);
        acc += qg_half(blk) * xd[b] * static_cast<float>(dot);
    }
    return qg_warp_sum(acc);
}

// Block-wide row loop, one warp per row. Usable both as a persistent-scheduler task and inside a grid kernel.
// type: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K.
template <int type>
__device__ __forceinline__ void qgemv_dp4a_rows(const uint8_t* W, const int8_t* xq, const float* xd, float* y, int rows,
                                                int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int nb = type == 0 ? (K >> 5) : (K >> 8);
    const int bb = type == 0 ? 34 : (type == 1 ? 144 : 210);
    const size_t row_bytes = static_cast<size_t>(nb) * bb;
    for (int row = warp; row < rows; row += nwarps) {
        const uint8_t* rowp = W + static_cast<size_t>(row) * row_bytes;
        float v;
        if (type == 0) v = q8_dp4a_row(rowp, xq, xd, nb, lane);
        else if (type == 1) v = q4k_dp4a_row(rowp, xq, xd, nb, lane);
        else v = q6k_dp4a_row(rowp, xq, xd, nb, lane);
        if (lane == 0) y[row] = v;
    }
    __syncthreads();
}

}  // namespace akd
