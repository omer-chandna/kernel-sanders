#pragma once

#include <adakernel/api.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

struct PrefillLayer {
    const float* attn_norm;
    const float* wq;
    const float* wk;
    const float* wv;
    const float* wo;
    const float* q_norm;
    const float* k_norm;
    const float* ffn_norm;
    const float* wgate;
    const float* wup;
    const float* wdown;
    float* kc;
    float* vc;
    void* kv_view;
};

struct PrefillBuffers {
    float* x;
    float* y;
    float* q;
    float* k;
    float* v;
    float* attn;
    float* gate;
    float* up;
    float* ffn;
    float* proj;
    float* logits;
    float* step_logits;
};

ak_status ak_prefill_extend(cublasHandle_t blas, cudaStream_t stream, const ak_model_config* cfg,
                            const PrefillLayer* layers, const float* tok_embd, const float* out_norm, const float* wout,
                            const int* d_tokens, int n_tokens, int pos0, PrefillBuffers buf, bool store_every_logits);
