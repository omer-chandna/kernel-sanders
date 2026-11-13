#include "prefill.cuh"

#include "error.h"
#include "ops.cuh"

namespace {

__global__ void embed_rows(const float* table, const int* tokens, float* x, int n, int d) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < d) {
        x[static_cast<size_t>(row) * d + col] = table[static_cast<size_t>(tokens[row]) * d + col];
    }
}

__global__ void rmsnorm_rows_kernel(const float* x, const float* w, float* y, int n, int d, float eps) {
    if (blockIdx.x < n) {
        akd::rmsnorm_row(x + static_cast<size_t>(blockIdx.x) * d, w, y + static_cast<size_t>(blockIdx.x) * d, d, eps);
    }
}

__global__ void qknorm_rows_kernel(float* q, const float* w, int rows, int groups, int width, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float* base = q + static_cast<size_t>(row) * groups * width;
    for (int g = 0; g < groups; ++g) {
        akd::rmsnorm_row(base + static_cast<size_t>(g) * width, w, base + static_cast<size_t>(g) * width, width, eps);
    }
}

__global__ void rope_rows_kernel(float* q, int rows, int n_heads, int hd, int pos0, float theta) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    akd::rope_neox(q + static_cast<size_t>(row) * n_heads * hd, n_heads, hd, pos0 + row, theta);
}

__global__ void add_rows_kernel(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += y[i];
    }
}

__global__ void silu_rows_kernel(const float* gate, const float* up, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        y[i] = g * (1.f / (1.f + expf(-g))) * up[i];
    }
}

__global__ void kv_store_kernel(const float* k, const float* v, float* kc, float* vc, int n, int pos0, int width) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < width) {
        kc[static_cast<size_t>(pos0 + row) * width + col] = k[static_cast<size_t>(row) * width + col];
        vc[static_cast<size_t>(pos0 + row) * width + col] = v[static_cast<size_t>(row) * width + col];
    }
}

__global__ void kv_store_view_kernel(const float* k, const float* v, akd::AkKvView* view, int n, int pos0) {
    const int row = blockIdx.x;
    if (row >= n || !view) {
        return;
    }
    const akd::AkKvView local = *view;
    const int token = akd::kv_token_index(local, pos0 + row);
    const float* krow = k + static_cast<size_t>(row) * local.n_kv * local.hd;
    const float* vrow = v + static_cast<size_t>(row) * local.n_kv * local.hd;
    for (int kv = 0; kv < local.n_kv; ++kv) {
        akd::kv_store_head(local.k, local.kscale, local.type, token, local.n_kv, local.hd, kv, krow);
        __syncthreads();
        akd::kv_store_head(local.v, local.vscale, local.type, token, local.n_kv, local.hd, kv, vrow);
        __syncthreads();
    }
}

__global__ void attn_view_kernel(const float* q, akd::AkKvView* view, float* o, int n, int pos0, int n_heads) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (!view || head >= n_heads || row >= n) {
        return;
    }
    const akd::AkKvView local = *view;
    const int kv = head / (n_heads / local.n_kv);
    const float* qq = q + (static_cast<size_t>(row) * n_heads + head) * local.hd;
    const int end = pos0 + row;
    const float scale = rsqrtf(static_cast<float>(local.hd));
    float local_max = -1.0e30f;
    for (int t = threadIdx.x; t <= end; t += blockDim.x) {
        const int token = akd::kv_token_index(local, t);
        float dot = 0.f;
        for (int i = 0; i < local.hd; ++i) {
            dot += qq[i] * akd::kv_load(local.k, local.kscale, local.type, token, local.n_kv, local.hd, kv, i);
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    const float mx = akd::block_max(local_max);
    float local_sum = 0.f;
    for (int t = threadIdx.x; t <= end; t += blockDim.x) {
        const int token = akd::kv_token_index(local, t);
        float dot = 0.f;
        for (int i = 0; i < local.hd; ++i) {
            dot += qq[i] * akd::kv_load(local.k, local.kscale, local.type, token, local.n_kv, local.hd, kv, i);
        }
        local_sum += expf(dot * scale - mx);
    }
    const float sum = akd::block_sum(local_sum);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int i = threadIdx.x; i < local.hd; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t <= end; ++t) {
            const int token = akd::kv_token_index(local, t);
            float dot = 0.f;
            for (int j = 0; j < local.hd; ++j) {
                dot += qq[j] * akd::kv_load(local.k, local.kscale, local.type, token, local.n_kv, local.hd, kv, j);
            }
            acc += expf(dot * scale - mx) * inv *
                   akd::kv_load(local.v, local.vscale, local.type, token, local.n_kv, local.hd, kv, i);
        }
        o[(static_cast<size_t>(row) * n_heads + head) * local.hd + i] = acc;
    }
}

__global__ void attn_cache_kernel(const float* q, const float* kc, const float* vc, float* o, int n, int pos0,
                                  int n_heads, int n_kv, int hd) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= n_heads || row >= n) {
        return;
    }
    const int kv = head / (n_heads / n_kv);
    const float* qq = q + (static_cast<size_t>(row) * n_heads + head) * hd;
    const int end = pos0 + row;
    const float scale = rsqrtf(static_cast<float>(hd));
    float local_max = -1.0e30f;
    for (int t = threadIdx.x; t <= end; t += blockDim.x) {
        const float* krow = kc + (static_cast<size_t>(t) * n_kv + kv) * hd;
        float dot = 0.f;
        for (int i = 0; i < hd; ++i) {
            dot += qq[i] * krow[i];
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    const float mx = akd::block_max(local_max);
    float local_sum = 0.f;
    for (int t = threadIdx.x; t <= end; t += blockDim.x) {
        const float* krow = kc + (static_cast<size_t>(t) * n_kv + kv) * hd;
        float dot = 0.f;
        for (int i = 0; i < hd; ++i) {
            dot += qq[i] * krow[i];
        }
        local_sum += expf(dot * scale - mx);
    }
    const float sum = akd::block_sum(local_sum);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t <= end; ++t) {
            const float* krow = kc + (static_cast<size_t>(t) * n_kv + kv) * hd;
            const float* vrow = vc + (static_cast<size_t>(t) * n_kv + kv) * hd;
            float dot = 0.f;
            for (int j = 0; j < hd; ++j) {
                dot += qq[j] * krow[j];
            }
            acc += expf(dot * scale - mx) * inv * vrow[i];
        }
        o[(static_cast<size_t>(row) * n_heads + head) * hd + i] = acc;
    }
}

ak_status gemm_nt(cublasHandle_t blas, const float* W, const float* X, float* Y, int rows, int S, int K) {
    const float alpha = 1.f;
    const float beta = 0.f;
    cublasStatus_t st = cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, S, K, &alpha, W, K, X, K, &beta, Y, rows);
    if (st != CUBLAS_STATUS_SUCCESS) {
        ak_set_error("cublasSgemm failed");
        return AK_ERR_CUDA;
    }
    return AK_OK;
}

}  // namespace

ak_status ak_prefill_extend(cublasHandle_t blas, cudaStream_t stream, const ak_model_config* cfg,
                            const PrefillLayer* layers, const float* tok_embd, const float* out_norm, const float* wout,
                            const int* d_tokens, int n_tokens, int pos0, PrefillBuffers buf, bool store_every_logits) {
    if (!cfg || !layers || n_tokens < 1 || pos0 < 0) {
        ak_set_error("prefill: bad arguments");
        return AK_ERR_INVALID;
    }
    cublasSetStream(blas, stream);
    const int d = cfg->d_model;
    const int q_rows = cfg->n_heads * cfg->head_dim;
    const int kv_rows = cfg->n_kv_heads * cfg->head_dim;
    const int ff = cfg->d_ff;
    const int S = n_tokens;
    dim3 embed_grid((d + 255) / 256, S);
    embed_rows<<<embed_grid, 256, 0, stream>>>(tok_embd, d_tokens, buf.x, S, d);
    float* cur = buf.x;
    for (int layer = 0; layer < cfg->n_layers; ++layer) {
        const PrefillLayer& ly = layers[layer];
        rmsnorm_rows_kernel<<<S, 256, 0, stream>>>(cur, ly.attn_norm, buf.y, S, d, cfg->rms_eps);
        if (gemm_nt(blas, ly.wq, buf.y, buf.q, q_rows, S, d) != AK_OK) return AK_ERR_CUDA;
        if (gemm_nt(blas, ly.wk, buf.y, buf.k, kv_rows, S, d) != AK_OK) return AK_ERR_CUDA;
        if (gemm_nt(blas, ly.wv, buf.y, buf.v, kv_rows, S, d) != AK_OK) return AK_ERR_CUDA;
        if (cfg->qk_norm) {
            qknorm_rows_kernel<<<S, 128, 0, stream>>>(buf.q, ly.q_norm, S, cfg->n_heads, cfg->head_dim, cfg->rms_eps);
            qknorm_rows_kernel<<<S, 128, 0, stream>>>(buf.k, ly.k_norm, S, cfg->n_kv_heads, cfg->head_dim, cfg->rms_eps);
        }
        rope_rows_kernel<<<S, 128, 0, stream>>>(buf.q, S, cfg->n_heads, cfg->head_dim, pos0, cfg->rope_theta);
        rope_rows_kernel<<<S, 128, 0, stream>>>(buf.k, S, cfg->n_kv_heads, cfg->head_dim, pos0, cfg->rope_theta);
        if (ly.kv_view) {
            kv_store_view_kernel<<<S, 128, 0, stream>>>(buf.k, buf.v, reinterpret_cast<akd::AkKvView*>(ly.kv_view), S, pos0);
            dim3 attn_grid(cfg->n_heads, S);
            attn_view_kernel<<<attn_grid, 128, 0, stream>>>(buf.q, reinterpret_cast<akd::AkKvView*>(ly.kv_view), buf.attn, S, pos0, cfg->n_heads);
        } else {
            dim3 kv_grid((kv_rows + 255) / 256, S);
            kv_store_kernel<<<kv_grid, 256, 0, stream>>>(buf.k, buf.v, ly.kc, ly.vc, S, pos0, kv_rows);
            dim3 attn_grid(cfg->n_heads, S);
            attn_cache_kernel<<<attn_grid, 128, 0, stream>>>(buf.q, ly.kc, ly.vc, buf.attn, S, pos0, cfg->n_heads,
                                                            cfg->n_kv_heads, cfg->head_dim);
        }
        if (gemm_nt(blas, ly.wo, buf.attn, buf.proj, d, S, q_rows) != AK_OK) return AK_ERR_CUDA;
        add_rows_kernel<<<(S * d + 255) / 256, 256, 0, stream>>>(cur, buf.proj, S * d);
        rmsnorm_rows_kernel<<<S, 256, 0, stream>>>(cur, ly.ffn_norm, buf.y, S, d, cfg->rms_eps);
        if (gemm_nt(blas, ly.wgate, buf.y, buf.gate, ff, S, d) != AK_OK) return AK_ERR_CUDA;
        if (gemm_nt(blas, ly.wup, buf.y, buf.up, ff, S, d) != AK_OK) return AK_ERR_CUDA;
        silu_rows_kernel<<<(S * ff + 255) / 256, 256, 0, stream>>>(buf.gate, buf.up, buf.ffn, S * ff);
        if (gemm_nt(blas, ly.wdown, buf.ffn, buf.proj, d, S, ff) != AK_OK) return AK_ERR_CUDA;
        add_rows_kernel<<<(S * d + 255) / 256, 256, 0, stream>>>(cur, buf.proj, S * d);
    }
    rmsnorm_rows_kernel<<<S, 256, 0, stream>>>(cur, out_norm, buf.y, S, d, cfg->rms_eps);
    if (gemm_nt(blas, wout, buf.y, buf.step_logits ? buf.step_logits : buf.logits, cfg->vocab, S, d) != AK_OK) {
        return AK_ERR_CUDA;
    }
    if (buf.step_logits && buf.logits) {
        cudaMemcpyAsync(buf.logits, buf.step_logits + static_cast<size_t>(S - 1) * cfg->vocab,
                        sizeof(float) * cfg->vocab, cudaMemcpyDeviceToDevice, stream);
    }
    (void)store_every_logits;
    return AK_OK;
}
