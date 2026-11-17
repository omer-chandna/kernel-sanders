#pragma once

// Fused single-token decode ops for the quantized GGUF runtime. Each one replaces a chain of two or three tasks
// from the FP32 engine, so the per-layer dependency chain in the persistent scheduler gets shorter. All are
// block-wide: they work both as a persistent-scheduler task and inside a one-block launch. Included by ops.cuh
// after block_sum / block_max.

namespace akd {

// x += res (if res), then y = rmsnorm(x) * w, written only in Q8_1 form (xq, xd). n % 32 == 0.
__device__ __forceinline__ void add_rmsnorm_quant(float* x, const float* res, const float* w, int8_t* xq, float* xd,
                                                  int n, float eps) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (res) {
            v += res[i];
            x[i] = v;
        }
        ss += v * v;
    }
    ss = block_sum(ss);  // syncs, so the x writes above are visible below
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    for (int base = warp * 32; base < n; base += nwarps * 32) {
        const float v = x[base + lane] * inv * w[base + lane];
        const float amax = qg_warp_max(fabsf(v));
        const float d = amax / 127.f;
        xq[base + lane] = static_cast<int8_t>(amax > 0.f ? __float2int_rn(v / d) : 0);
        if (lane == 0) xd[base >> 5] = d;
    }
    __syncthreads();
}

// q = silu(gate) * up over [0, n), written only in Q8_1 form. Every 32-block is independent, so callers may
// split the range. n % 32 == 0.
__device__ __forceinline__ void silu_quant(const float* gate, const float* up, int8_t* xq, float* xd, int n) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    for (int base = warp * 32; base < n; base += nwarps * 32) {
        const float g = gate[base + lane];
        const float v = g * (1.f / (1.f + expf(-g))) * up[base + lane];
        const float amax = qg_warp_max(fabsf(v));
        const float d = amax / 127.f;
        xq[base + lane] = static_cast<int8_t>(amax > 0.f ? __float2int_rn(v / d) : 0);
        if (lane == 0) xd[base >> 5] = d;
    }
    __syncthreads();
}

// Per-head RMSNorm then NeoX RoPE, one warp per head. With kc set, the rotated heads and v are also appended to
// the KV cache at *pos (layout [max_seq][n_heads][hd]). hd must be a multiple of 64 and at most 256.
__device__ __forceinline__ void qk_norm_rope(float* q, const float* w, int n_heads, int hd, float eps, const int* pos_p,
                                             float theta, float* kc, const float* v, float* vc, int max_seq) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int pos = *pos_p;
    const int nj = hd >> 5;
    const int hj = nj >> 1;
    const bool store = kc && pos >= 0 && pos < max_seq;
    for (int h = warp; h < n_heads; h += nwarps) {
        float* row = q + h * hd;
        float vals[8];
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < nj) {
                vals[j] = row[lane + 32 * j];
                ss += vals[j] * vals[j];
            }
        }
        ss = qg_warp_sum(ss);
        const float inv = rsqrtf(ss / static_cast<float>(hd) + eps);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < nj) vals[j] = vals[j] * inv * w[lane + 32 * j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < hj) {
                const int i = lane + 32 * j;
                const float freq = expf(-logf(theta) * (2.f * static_cast<float>(i) / static_cast<float>(hd)));
                const float ang = static_cast<float>(pos) * freq;
                const float c = cosf(ang);
                const float s = sinf(ang);
                const float x1 = vals[j];
                const float x2 = vals[j + hj];
                vals[j] = x1 * c - x2 * s;
                vals[j + hj] = x2 * c + x1 * s;
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < nj) {
                row[lane + 32 * j] = vals[j];
                if (store) {
                    const size_t off = (static_cast<size_t>(pos) * n_heads + h) * hd + lane + 32 * j;
                    kc[off] = vals[j];
                    vc[off] = v[h * hd + lane + 32 * j];
                }
            }
        }
    }
    __syncthreads();
}

// One query head against the cache, output written as Q8_1 blocks (oq[hd], od[hd / 32]).
// Scores: warp per key, lanes across head_dim (coalesced). Output: blockDim / hd groups split the keys.
// Requires hd % 32 == 0, hd <= 256, blockDim % hd == 0, blockDim <= 256.
__device__ __forceinline__ void attn_decode_head(const float* q, const float* kc, const float* vc, int8_t* oq, float* od,
                                                 float* scores, int kv_head, int hd, int seq, int n_kv) {
    __shared__ float part[256];
    __shared__ float outv[256];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int nj = hd >> 5;
    const float scale = rsqrtf(static_cast<float>(hd));
    float qr[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) qr[j] = j < nj ? q[lane + 32 * j] : 0.f;
    for (int t = warp; t < seq; t += nwarps) {
        const float* krow = kc + (static_cast<size_t>(t) * n_kv + kv_head) * hd;
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < nj) dot += qr[j] * krow[lane + 32 * j];
        }
        dot = qg_warp_sum(dot);
        if (lane == 0) scores[t] = dot * scale;
    }
    __syncthreads();
    float mx = -1.0e30f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) mx = fmaxf(mx, scores[t]);
    mx = block_max(mx);
    float sum = 0.f;
    for (int t = threadIdx.x; t < seq; t += blockDim.x) {
        const float e = expf(scores[t] - mx);
        scores[t] = e;
        sum += e;
    }
    sum = block_sum(sum);  // syncs: every score is final after this
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    const int groups = blockDim.x / hd;
    const int g = threadIdx.x / hd;
    const int dim = threadIdx.x - g * hd;
    float acc = 0.f;
    if (g < groups) {
        for (int t = g; t < seq; t += groups) {
            acc += scores[t] * vc[(static_cast<size_t>(t) * n_kv + kv_head) * hd + dim];
        }
    }
    part[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x < hd) {
        float o = 0.f;
        for (int gg = 0; gg < groups; ++gg) o += part[gg * hd + threadIdx.x];
        outv[threadIdx.x] = o * inv;
    }
    __syncthreads();
    if (warp < nj) {
        const float v = outv[warp * 32 + lane];
        const float amax = qg_warp_max(fabsf(v));
        const float d = amax / 127.f;
        oq[warp * 32 + lane] = static_cast<int8_t>(amax > 0.f ? __float2int_rn(v / d) : 0);
        if (lane == 0) od[warp] = d;
    }
    __syncthreads();
}

// Monotonic key: larger float -> larger key; ties -> smaller index wins.
__device__ __forceinline__ unsigned long long argmax_key(float v, int idx) {
    unsigned int u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | static_cast<unsigned int>(0xFFFFFFFFu - static_cast<unsigned>(idx));
}

__device__ __forceinline__ int argmax_key_index(unsigned long long key) {
    return static_cast<int>(0xFFFFFFFFu - static_cast<unsigned int>(key & 0xFFFFFFFFull));
}

// Folds v[0, n) (global indices base + i) into *best with atomicMax.
__device__ __forceinline__ void argmax_part(const float* v, int n, int base, unsigned long long* best) {
    unsigned long long k = 0ull;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned long long c = argmax_key(v[i], base + i);
        k = c > k ? c : k;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long other = __shfl_xor_sync(0xffffffff, k, o);
        k = other > k ? other : k;
    }
    if ((threadIdx.x & 31) == 0 && k) atomicMax(best, k);
    __syncthreads();
}

}  // namespace akd
