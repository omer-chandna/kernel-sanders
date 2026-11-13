#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define AK_API __declspec(dllexport)
#else
#define AK_API __attribute__((visibility("default")))
#endif

typedef enum ak_status {
    AK_OK = 0,
    AK_ERR_CUDA = 1,
    AK_ERR_INVALID = 2,
    AK_ERR_OOM = 3,
    AK_ERR_TIMEOUT = 4,
    AK_ERR_UNSUPPORTED = 5
} ak_status;

enum {
    AK_BACKEND_EAGER = 0,
    AK_BACKEND_CUDA_GRAPH = 1,
    AK_BACKEND_PERSISTENT = 2,
    AK_BACKEND_HYBRID = 3,
    AK_BACKEND_FUSED = 4,
    AK_BACKEND_CUBLAS = 5,
    AK_BACKEND_CUBLAS_GRAPH = 6
};

enum {
    AK_OP_NOP = 0,
    AK_OP_RMSNORM = 1,
    AK_OP_RESIDUAL = 2,
    AK_OP_FUSED_ADD_RMSNORM = 3,
    AK_OP_ROPE = 4,
    AK_OP_MATVEC = 5,
    AK_OP_SILU_MUL = 6,
    AK_OP_ATTN_HEAD = 7,
    AK_OP_Q8_MATVEC = 8,
    AK_OP_Q4K_MATVEC = 9,
    AK_OP_EMBED = 10,
    AK_OP_ARGMAX = 11,
    AK_OP_QK_NORM = 12,
    AK_OP_KV_WRITE = 13,
    AK_OP_FINALIZE = 14
};

typedef struct ak_task {
    int32_t op;
    int32_t preds_left;
    int32_t preds_init;
    int32_t succ_begin;
    int32_t n_succ;
    int32_t n0;
    int32_t n1;
    int32_t n2;
    int32_t n3;
    float f0;
    int32_t i0;
    int32_t i1;
    uint64_t p0;
    uint64_t p1;
    uint64_t p2;
    uint64_t p3;
    uint64_t p4;
    uint64_t p5;
    uint64_t p6;
} ak_task;

typedef struct ak_device_info {
    char name[256];
    char pci_bus_id[32];
    int32_t cc_major;
    int32_t cc_minor;
    int32_t sm_count;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t max_threads_per_sm;
    int32_t regs_per_sm;
    int32_t shared_mem_per_sm;
    int32_t shared_mem_per_block;
    int32_t shared_mem_per_block_optin;
    int32_t l2_cache_bytes;
    int32_t memory_clock_khz;
    int32_t memory_bus_width_bits;
    int32_t cooperative_launch;
    int32_t persistent_max_blocks;
    int32_t pci_bus;
    int32_t pci_device;
    int32_t clock_sm_khz;
    float total_vram_mb;
    float free_vram_mb;
    float theoretical_bandwidth_gbps;
} ak_device_info;

typedef struct ak_model_config {
    int32_t n_layers;
    int32_t d_model;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t d_ff;
    int32_t vocab;
    int32_t max_seq;
    float rms_eps;
    float rope_theta;
    int32_t qk_norm;
    int32_t n_gpu_layers;
    int32_t chunk_rows;
    int32_t offload_overlap;
    int32_t persistent_blocks;
    int32_t kv_type;
    int32_t page_tokens;
} ak_model_config;

typedef struct ak_fp32_weights {
    const float* tok_embd;
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
    const float* out_norm;
    const float* wout;
    int32_t tied_embeddings;
} ak_fp32_weights;

typedef struct ak_gen_stats {
    float prefill_ms;
    float decode_ms;
    float ttft_ms;
    float graph_capture_ms;
    int32_t prompt_tokens;
    int32_t generated_tokens;
    int32_t backend;
    int32_t launches_per_token;
    float decode_tps;
    float prefill_tps;
    float vram_peak_mb;
    float weight_vram_mb;
    float host_weight_mb;
    int32_t n_gpu_layers;
    int32_t persistent_blocks;
    int32_t status;
    int32_t graph_nodes;
} ak_gen_stats;

#define AK_BENCH_CAP 32

typedef struct ak_bench_row {
    char name[32];
    float median_us;
    float min_us;
    float p95_us;
    float max_abs_err;
    float rel_err;
    int32_t correct;
    int32_t iters;
} ak_bench_row;

typedef struct ak_micro_report {
    ak_bench_row rows[AK_BENCH_CAP];
    int32_t n_rows;
    int32_t sm_count;
    int32_t regs_persistent;
    int32_t persistent_blocks;
    float achieved_bandwidth_gbps;
    float theoretical_bandwidth_gbps;
    float launch_eager_us;
    float launch_graph_us;
    float launch_persistent_us;
    float coop_supported;
} ak_micro_report;

typedef struct ak_model ak_model;

AK_API const char* ak_last_error(void);
AK_API ak_status ak_init(int device);
AK_API ak_status ak_device_query(ak_device_info* out);
AK_API ak_status ak_sync(void);

AK_API ak_status ak_q8_quantize(const float* x, void* dst, int n);
AK_API ak_status ak_q8_dequantize(const void* src, float* y, int n);
AK_API ak_status ak_q4k_quantize(const float* x, void* dst, int n);
AK_API ak_status ak_q4k_dequantize(const void* src, float* y, int n);
AK_API ak_status ak_q6k_dequantize(const void* src, float* y, int n);
AK_API int ak_q8_block_size(void);
AK_API int ak_q4k_block_size(void);
AK_API int ak_q6k_block_size(void);

AK_API ak_status ak_test_rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps);
AK_API ak_status ak_test_residual(const float* a, const float* b, float* y, int n);
AK_API ak_status ak_test_fused_add_rmsnorm(const float* x, const float* residual, const float* w,
                                           float* y_add, float* y_norm, int cols, float eps);
AK_API ak_status ak_test_rope(float* q, int n_heads, int head_dim, int pos, float theta);
AK_API ak_status ak_test_qk_norm(float* q, const float* w, int n_heads, int head_dim, float eps);
AK_API ak_status ak_test_matvec(const float* W, const float* x, float* y, int rows, int K);
AK_API ak_status ak_test_cublas_matvec(const float* W, const float* x, float* y, int rows, int K);
AK_API ak_status ak_test_silu_mul(const float* gate, const float* up, float* y, int n);
AK_API ak_status ak_test_attn(const float* q, const float* k, const float* v, float* out,
                              int n_heads, int n_kv_heads, int head_dim, int seq, int max_seq);
AK_API ak_status ak_test_q8_matvec(const void* Wq, const float* x, float* y, int rows, int K);
AK_API ak_status ak_test_q4k_matvec(const void* Wq, const float* x, float* y, int rows, int K);
AK_API ak_status ak_persistent_execute(const ak_task* tasks, int n_tasks, const int32_t* succ, int n_edges,
                                       int n_blocks, int max_idle);

AK_API ak_status ak_model_create(const ak_model_config* cfg, const ak_fp32_weights* weights, ak_model** out);
AK_API ak_status ak_model_destroy(ak_model* model);
AK_API ak_status ak_model_logits(ak_model* model, const int32_t* prompt, int n_prompt, float* logits);
AK_API ak_status ak_generate(ak_model* model, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                             int n_gen, int backend, float* per_token_ms, int per_token_cap, ak_gen_stats* stats);
AK_API ak_status ak_speculate(ak_model* target, ak_model* draft, const int32_t* prompt, int n_prompt,
                              int32_t* out_tokens, int n_gen, int gamma, int32_t* out_accepts, int accept_cap,
                              ak_gen_stats* stats);
AK_API ak_status ak_microbench(int rows, int K, int seq, int n_heads, int iters, int warmup, ak_micro_report* out);

#ifdef __cplusplus
}
#endif
