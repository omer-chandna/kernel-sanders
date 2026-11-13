#include "error.h"
#include "launch.h"
#include "ops.cuh"

#include <adakernel/api.h>

#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

__global__ void nop_kernel() {}

__global__ void copy4_kernel(const float4* in, float4* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        out[i] = in[i];
    }
}

namespace cg = cooperative_groups;

__global__ void coop_touch(int* counter) {
    cg::grid_group grid = cg::this_grid();
    if (threadIdx.x == 0) {
        atomicAdd(counter, 1);
    }
    grid.sync();
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t bytes) {
        if (cudaMalloc(&p, bytes == 0 ? 4 : bytes) != cudaSuccess) p = nullptr;
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    template <typename T>
    T* as() const { return reinterpret_cast<T*>(p); }
};

void stats_us(std::vector<float>& samples, float* median, float* min_v, float* p95) {
    std::sort(samples.begin(), samples.end());
    *min_v = samples.front();
    *median = samples[samples.size() / 2];
    const size_t i95 = std::min(samples.size() - 1, static_cast<size_t>(samples.size() * 95 / 100));
    *p95 = samples[i95];
}

template <typename Fn>
void time_gpu(Fn&& fn, cudaStream_t stream, int warmup, int iters, float* median, float* min_v, float* p95) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaStreamSynchronize(stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<float> samples(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start, stream);
        fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        samples[i] = ms * 1000.f;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    stats_us(samples, median, min_v, p95);
}

void push_row(ak_micro_report* out, const char* name, float median, float min_v, float p95, float err, float rel,
              int correct, int iters) {
    if (out->n_rows >= AK_BENCH_CAP) return;
    ak_bench_row& row = out->rows[out->n_rows++];
    std::memset(&row, 0, sizeof(row));
    std::snprintf(row.name, sizeof(row.name), "%s", name);
    row.median_us = median;
    row.min_us = min_v;
    row.p95_us = p95;
    row.max_abs_err = err;
    row.rel_err = rel;
    row.correct = correct;
    row.iters = iters;
}

float max_abs(const std::vector<float>& a, const std::vector<float>& b, float* rel) {
    float m = 0.f;
    float r = 0.f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        m = std::max(m, d);
        const float den = std::max(1.0e-6f, std::fabs(b[i]));
        r = std::max(r, d / den);
    }
    *rel = r;
    return m;
}

__global__ void q8_grid_kernel(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) {
        return;
    }
    int nrows = rows - row0;
    if (nrows > nwarps) {
        nrows = nwarps;
    }
    const int nb = K >> 5;
    akd::q8_matvec_rows(W + static_cast<size_t>(row0) * nb * 34, x, y + row0, nrows, K);
}

__global__ void q4k_grid_kernel(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) {
        return;
    }
    int nrows = rows - row0;
    if (nrows > nwarps) {
        nrows = nwarps;
    }
    const int nb = K >> 8;
    akd::q4k_matvec_rows(W + static_cast<size_t>(row0) * nb * 144, x, y + row0, nrows, K);
}

__global__ void q8_dequant_kernel(const uint8_t* W, float* out, int rows, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * K) {
        return;
    }
    const int row = idx / K;
    const int col = idx - row * K;
    const int nb = K >> 5;
    const uint8_t* blk = W + (static_cast<size_t>(row) * nb + (col >> 5)) * 34;
    uint16_t bits;
    memcpy(&bits, blk, 2);
    const int8_t q = reinterpret_cast<const int8_t*>(blk + 2)[col & 31];
    out[idx] = akd::half_to_f(bits) * static_cast<float>(q);
}

__global__ void q4k_dequant_kernel(const uint8_t* W, float* out, int rows, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * K) {
        return;
    }
    const int row = idx / K;
    const int col = idx - row * K;
    const int nb = K >> 8;
    const uint8_t* blk = W + (static_cast<size_t>(row) * nb + (col >> 8)) * 144;
    uint16_t d_bits, m_bits;
    memcpy(&d_bits, blk, 2);
    memcpy(&m_bits, blk + 2, 2);
    out[idx] = akd::q4k_at(blk + 4, blk + 16, akd::half_to_f(d_bits), akd::half_to_f(m_bits), col & 255);
}

}  // namespace

extern "C" ak_status ak_microbench(int rows, int K, int seq, int n_heads, int iters, int warmup, ak_micro_report* out) {
    if (!out || rows < 1 || K < 1 || iters < 1 || warmup < 0) {
        ak_set_error("microbench: bad arguments");
        return AK_ERR_INVALID;
    }
    ak_status st = ak_init(0);
    if (st != AK_OK) return st;
    std::memset(out, 0, sizeof(*out));
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    out->sm_count = prop.multiProcessorCount;
    out->regs_persistent = ak_persistent_regs();
    out->persistent_blocks = ak_persistent_occupancy();
    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
    out->coop_supported = static_cast<float>(coop);
    if (seq < 1) seq = 1;
    if (n_heads < 1) n_heads = 1;
    const int head_dim = 64;
    const int n_kv = n_heads;

    std::vector<float> hW(static_cast<size_t>(rows) * K);
    std::vector<float> hx(K), hy(rows), href(rows), hnorm(K, 1.f), hy2(K);
    for (int i = 0; i < K; ++i) {
        hx[i] = 0.01f * static_cast<float>((i % 17) - 8);
        hnorm[i] = 0.8f + 0.01f * static_cast<float>(i % 5);
    }
    for (int r = 0; r < rows; ++r) {
        float acc = 0.f;
        for (int c = 0; c < K; ++c) {
            const float w = 0.002f * static_cast<float>(((r * 13 + c * 3) % 11) - 5);
            hW[static_cast<size_t>(r) * K + c] = w;
            acc += w * hx[c];
        }
        href[r] = acc;
    }
    Dev dW(sizeof(float) * hW.size()), dx(sizeof(float) * K), dy(sizeof(float) * rows), dnorm(sizeof(float) * K);
    Dev dy2(sizeof(float) * K);
    if (!dW.p || !dx.p || !dy.p || !dnorm.p || !dy2.p) return AK_ERR_OOM;
    cudaMemcpy(dW.p, hW.data(), sizeof(float) * hW.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx.p, hx.data(), sizeof(float) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dnorm.p, hnorm.data(), sizeof(float) * K, cudaMemcpyHostToDevice);

    auto rms = [&]() {
        ak_task task{};
        task.op = AK_OP_RMSNORM;
        task.p0 = reinterpret_cast<uint64_t>(dx.p);
        task.p1 = reinterpret_cast<uint64_t>(dnorm.p);
        task.p2 = reinterpret_cast<uint64_t>(dy2.p);
        task.n0 = K;
        task.f0 = 1.0e-6f;
        ak_launch_task(&task, nullptr);
    };
    float med, mn, p95;
    time_gpu(rms, nullptr, warmup, iters, &med, &mn, &p95);
    rms();
    cudaDeviceSynchronize();
    cudaMemcpy(hy2.data(), dy2.p, sizeof(float) * K, cudaMemcpyDeviceToHost);
    float ss = 0.f;
    for (float v : hx) ss += v * v;
    const float inv = 1.f / std::sqrt(ss / static_cast<float>(K) + 1.0e-6f);
    std::vector<float> rms_ref(K);
    for (int i = 0; i < K; ++i) rms_ref[i] = hx[i] * inv * hnorm[i];
    float rel = 0.f;
    float err = max_abs(hy2, rms_ref, &rel);
    push_row(out, "rmsnorm", med, mn, p95, err, rel, err < 2e-4f && rel < 2e-4f, iters);

    auto residual = [&]() {
        ak_task task{};
        task.op = AK_OP_RESIDUAL;
        task.p0 = reinterpret_cast<uint64_t>(dx.p);
        task.p1 = reinterpret_cast<uint64_t>(dnorm.p);
        task.p2 = reinterpret_cast<uint64_t>(dy2.p);
        task.n0 = K;
        ak_launch_task(&task, nullptr);
    };
    time_gpu(residual, nullptr, warmup, iters, &med, &mn, &p95);
    residual();
    cudaDeviceSynchronize();
    cudaMemcpy(hy2.data(), dy2.p, sizeof(float) * K, cudaMemcpyDeviceToHost);
    std::vector<float> add_ref(K);
    for (int i = 0; i < K; ++i) add_ref[i] = hx[i] + hnorm[i];
    err = max_abs(hy2, add_ref, &rel);
    push_row(out, "residual", med, mn, p95, err, rel, err < 1e-5f, iters);

    auto matvec = [&]() {
        ak_task task{};
        task.op = AK_OP_MATVEC;
        task.p0 = reinterpret_cast<uint64_t>(dW.p);
        task.p1 = reinterpret_cast<uint64_t>(dx.p);
        task.p2 = reinterpret_cast<uint64_t>(dy.p);
        task.n0 = rows;
        task.n1 = K;
        ak_launch_task(&task, nullptr);
    };
    time_gpu(matvec, nullptr, warmup, iters, &med, &mn, &p95);
    matvec();
    cudaDeviceSynchronize();
    cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
    err = max_abs(hy, href, &rel);
    push_row(out, "matvec", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);

    const int gemv_blocks = ak_persistent_occupancy();
    auto gemv_grid = [&]() {
        ak_launch_gemv_grid(dW.as<float>(), dx.as<float>(), dy.as<float>(), rows, K, gemv_blocks, nullptr);
    };
    time_gpu(gemv_grid, nullptr, warmup, iters, &med, &mn, &p95);
    gemv_grid();
    cudaDeviceSynchronize();
    cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
    err = max_abs(hy, href, &rel);
    push_row(out, "gemv_grid", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    auto gemv = [&]() {
        const float alpha = 1.f, beta = 0.f;
        cublasSgemv(handle, CUBLAS_OP_T, K, rows, &alpha, dW.as<float>(), K, dx.as<float>(), 1, &beta, dy.as<float>(), 1);
    };
    time_gpu(gemv, nullptr, warmup, iters, &med, &mn, &p95);
    gemv();
    cudaDeviceSynchronize();
    cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
    err = max_abs(hy, href, &rel);
    push_row(out, "cublas_gemv", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);
    cublasDestroy(handle);

    const int qk = (K / 32) * 32;
    if (qk >= 32) {
        std::vector<float> xq(hx.begin(), hx.begin() + qk);
        std::vector<unsigned char> qbytes(static_cast<size_t>(rows) * (qk / 32) * 34);
        std::vector<float> deq(static_cast<size_t>(rows) * qk);
        for (int r = 0; r < rows; ++r) {
            ak_q8_quantize(hW.data() + static_cast<size_t>(r) * K, qbytes.data() + static_cast<size_t>(r) * (qk / 32) * 34, qk);
            ak_q8_dequantize(qbytes.data() + static_cast<size_t>(r) * (qk / 32) * 34, deq.data() + static_cast<size_t>(r) * qk, qk);
        }
        std::vector<float> qref(rows);
        for (int r = 0; r < rows; ++r) {
            float acc = 0.f;
            for (int c = 0; c < qk; ++c) acc += deq[static_cast<size_t>(r) * qk + c] * xq[c];
            qref[r] = acc;
        }
        Dev dq(qbytes.size());
        Dev dxq(sizeof(float) * qk);
        cudaMemcpy(dq.p, qbytes.data(), qbytes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dxq.p, xq.data(), sizeof(float) * qk, cudaMemcpyHostToDevice);
        auto q8 = [&]() {
            ak_task task{};
            task.op = AK_OP_Q8_MATVEC;
            task.p0 = reinterpret_cast<uint64_t>(dq.p);
            task.p1 = reinterpret_cast<uint64_t>(dxq.p);
            task.p2 = reinterpret_cast<uint64_t>(dy.p);
            task.n0 = rows;
            task.n1 = qk;
            ak_launch_task(&task, nullptr);
        };
        time_gpu(q8, nullptr, warmup, iters, &med, &mn, &p95);
        q8();
        cudaDeviceSynchronize();
        cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
        err = max_abs(hy, qref, &rel);
        push_row(out, "q8_matvec", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);
        const int q8_blocks = (rows + 7) / 8;
        auto q8_grid = [&]() { q8_grid_kernel<<<q8_blocks, 256>>>(dq.as<uint8_t>(), dxq.as<float>(), dy.as<float>(), rows, qk); };
        time_gpu(q8_grid, nullptr, warmup, iters, &med, &mn, &p95);
        q8_grid();
        cudaDeviceSynchronize();
        cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
        err = max_abs(hy, qref, &rel);
        push_row(out, "q8_grid", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);
        Dev q8fp(sizeof(float) * static_cast<size_t>(rows) * qk);
        cublasHandle_t q8blas = nullptr;
        if (q8fp.p && cublasCreate(&q8blas) == CUBLAS_STATUS_SUCCESS) {
            const int deq_blocks = static_cast<int>((static_cast<size_t>(rows) * qk + 255) / 256);
            auto q8_base = [&]() {
                q8_dequant_kernel<<<deq_blocks, 256>>>(dq.as<uint8_t>(), q8fp.as<float>(), rows, qk);
                const float alpha = 1.f, beta = 0.f;
                cublasSgemv(q8blas, CUBLAS_OP_T, qk, rows, &alpha, q8fp.as<float>(), qk, dxq.as<float>(), 1, &beta, dy.as<float>(), 1);
            };
            time_gpu(q8_base, nullptr, warmup, iters, &med, &mn, &p95);
            q8_base();
            cudaDeviceSynchronize();
            cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
            err = max_abs(hy, qref, &rel);
            push_row(out, "q8_dequant_cublas", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);
            cublasDestroy(q8blas);
        }
    }

    const int q4k = (K / 256) * 256;
    if (q4k >= 256) {
        std::vector<unsigned char> qbytes(static_cast<size_t>(rows) * (q4k / 256) * 144);
        std::vector<float> deq(static_cast<size_t>(rows) * q4k), qref(rows), xq(hx.begin(), hx.begin() + q4k);
        for (int r = 0; r < rows; ++r) {
            ak_q4k_quantize(hW.data() + static_cast<size_t>(r) * K, qbytes.data() + static_cast<size_t>(r) * (q4k / 256) * 144, q4k);
            ak_q4k_dequantize(qbytes.data() + static_cast<size_t>(r) * (q4k / 256) * 144, deq.data() + static_cast<size_t>(r) * q4k, q4k);
            float acc = 0.f;
            for (int c = 0; c < q4k; ++c) acc += deq[static_cast<size_t>(r) * q4k + c] * xq[c];
            qref[r] = acc;
        }
        Dev dq(qbytes.size()), dxq(sizeof(float) * q4k);
        cudaMemcpy(dq.p, qbytes.data(), qbytes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dxq.p, xq.data(), sizeof(float) * q4k, cudaMemcpyHostToDevice);
        auto q4 = [&]() {
            ak_task task{};
            task.op = AK_OP_Q4K_MATVEC;
            task.p0 = reinterpret_cast<uint64_t>(dq.p);
            task.p1 = reinterpret_cast<uint64_t>(dxq.p);
            task.p2 = reinterpret_cast<uint64_t>(dy.p);
            task.n0 = rows;
            task.n1 = q4k;
            ak_launch_task(&task, nullptr);
        };
        time_gpu(q4, nullptr, warmup, iters, &med, &mn, &p95);
        q4();
        cudaDeviceSynchronize();
        cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
        err = max_abs(hy, qref, &rel);
        push_row(out, "q4k_matvec", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);
        const int q4_blocks = (rows + 7) / 8;
        auto q4_grid = [&]() { q4k_grid_kernel<<<q4_blocks, 256>>>(dq.as<uint8_t>(), dxq.as<float>(), dy.as<float>(), rows, q4k); };
        time_gpu(q4_grid, nullptr, warmup, iters, &med, &mn, &p95);
        q4_grid();
        cudaDeviceSynchronize();
        cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
        err = max_abs(hy, qref, &rel);
        push_row(out, "q4k_grid", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);
        Dev q4fp(sizeof(float) * static_cast<size_t>(rows) * q4k);
        cublasHandle_t q4blas = nullptr;
        if (q4fp.p && cublasCreate(&q4blas) == CUBLAS_STATUS_SUCCESS) {
            const int deq_blocks = static_cast<int>((static_cast<size_t>(rows) * q4k + 255) / 256);
            auto q4_base = [&]() {
                q4k_dequant_kernel<<<deq_blocks, 256>>>(dq.as<uint8_t>(), q4fp.as<float>(), rows, q4k);
                const float alpha = 1.f, beta = 0.f;
                cublasSgemv(q4blas, CUBLAS_OP_T, q4k, rows, &alpha, q4fp.as<float>(), q4k, dxq.as<float>(), 1, &beta, dy.as<float>(), 1);
            };
            time_gpu(q4_base, nullptr, warmup, iters, &med, &mn, &p95);
            q4_base();
            cudaDeviceSynchronize();
            cudaMemcpy(hy.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost);
            err = max_abs(hy, qref, &rel);
            push_row(out, "q4k_dequant_cublas", med, mn, p95, err, rel, err < 1e-4f || rel < 5e-3f, iters);
            cublasDestroy(q4blas);
        }
    }

    auto silu = [&]() {
        ak_task task{};
        task.op = AK_OP_SILU_MUL;
        task.p0 = reinterpret_cast<uint64_t>(dx.p);
        task.p1 = reinterpret_cast<uint64_t>(dnorm.p);
        task.p2 = reinterpret_cast<uint64_t>(dy2.p);
        task.n0 = K;
        ak_launch_task(&task, nullptr);
    };
    time_gpu(silu, nullptr, warmup, iters, &med, &mn, &p95);
    silu();
    cudaDeviceSynchronize();
    cudaMemcpy(hy2.data(), dy2.p, sizeof(float) * K, cudaMemcpyDeviceToHost);
    std::vector<float> silu_ref(K);
    for (int i = 0; i < K; ++i) {
        const float g = hx[i];
        silu_ref[i] = g * (1.f / (1.f + std::exp(-g))) * hnorm[i];
    }
    err = max_abs(hy2, silu_ref, &rel);
    push_row(out, "silu_mul", med, mn, p95, err, rel, err < 2e-4f && rel < 2e-4f, iters);

    const int launches = 32;
    auto nops = [&]() {
        for (int i = 0; i < launches; ++i) nop_kernel<<<1, 32>>>();
    };
    time_gpu(nops, nullptr, warmup, iters, &med, &mn, &p95);
    out->launch_eager_us = med / launches;
    push_row(out, "launch_eager_x32", med, mn, p95, 0.f, 0.f, 1, iters);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < launches; ++i) nop_kernel<<<1, 32, 0, stream>>>();
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, 0);
    auto replay = [&]() { cudaGraphLaunch(exec, stream); };
    time_gpu(replay, stream, warmup, iters, &med, &mn, &p95);
    out->launch_graph_us = med / launches;
    push_row(out, "launch_graph_x32", med, mn, p95, 0.f, 0.f, 1, iters);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);

    std::vector<ak_task> chain(launches);
    std::vector<int> succ(launches - 1);
    std::vector<int> ready{0};
    for (int i = 0; i < launches; ++i) {
        chain[i] = ak_task{};
        chain[i].op = AK_OP_NOP;
        chain[i].preds_init = i == 0 ? 0 : 1;
        chain[i].n_succ = i + 1 == launches ? 0 : 1;
        chain[i].succ_begin = i;
        if (i + 1 < launches) succ[i] = i + 1;
    }
    ak_persist_buf buf{};
    if (ak_persist_alloc(&buf, launches, launches - 1, 1, 0) == AK_OK &&
        ak_persist_upload(&buf, chain.data(), succ.data(), ready.data()) == AK_OK) {
        auto persistent = [&]() { ak_persist_launch(&buf, 1000000000, stream); };
        time_gpu(persistent, stream, warmup, iters, &med, &mn, &p95);
        out->launch_persistent_us = med / launches;
        push_row(out, "launch_persistent_x32", med, mn, p95, 0.f, 0.f, 1, iters);
        ak_persist_free(&buf);
    }
    cudaStreamDestroy(stream);

    const size_t n4 = (32ull << 20) / sizeof(float4);
    Dev a(n4 * sizeof(float4)), b(n4 * sizeof(float4));
    if (a.p && b.p) {
        auto copy = [&]() { copy4_kernel<<<256, 256>>>(a.as<float4>(), b.as<float4>(), n4); };
        time_gpu(copy, nullptr, 2, std::max(3, iters / 2), &med, &mn, &p95);
        const double bytes = static_cast<double>(n4) * sizeof(float4) * 2.0;
        const double seconds = med * 1.0e-6;
        out->achieved_bandwidth_gbps = seconds > 0 ? static_cast<float>(bytes / seconds / 1.0e9) : 0.f;
        push_row(out, "copy_bandwidth", med, mn, p95, 0.f, 0.f, 1, std::max(3, iters / 2));
    }
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0);
    if (clock_khz > 0 && prop.memoryBusWidth > 0) {
        out->theoretical_bandwidth_gbps =
            static_cast<float>(2.0 * (clock_khz * 1000.0) * (prop.memoryBusWidth / 8.0) / 1.0e9);
    }

    int* dcount = nullptr;
    cudaMalloc(&dcount, sizeof(int));
    cudaMemset(dcount, 0, sizeof(int));
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, coop_touch, 256, 0);
    const int grid = std::max(1, std::min(prop.multiProcessorCount, blocks * prop.multiProcessorCount));
    if (coop && blocks > 0 && grid <= blocks * prop.multiProcessorCount) {
        void* args[] = {&dcount};
        cudaError_t err = cudaLaunchCooperativeKernel(reinterpret_cast<void*>(coop_touch), dim3(grid), dim3(256), args, 0, nullptr);
        if (err != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            out->coop_supported = 0.f;
            cudaGetLastError();
        }
    } else {
        out->coop_supported = 0.f;
    }
    cudaFree(dcount);

    void* pinned = nullptr;
    const size_t expert_bytes = sizeof(float) * static_cast<size_t>(rows) * K;
    if (cudaHostAlloc(&pinned, expert_bytes, cudaHostAllocDefault) == cudaSuccess && pinned) {
        std::memcpy(pinned, hW.data(), expert_bytes);
        Dev slot_a(expert_bytes), slot_b(expert_bytes);
        cudaStream_t copy_stream = nullptr;
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
        cudaEvent_t copied = nullptr;
        cudaEventCreateWithFlags(&copied, cudaEventDisableTiming);
        const int expert_blocks = std::max(1, ak_persistent_occupancy());
        auto serial_experts = [&]() {
            for (int e = 0; e < 4; ++e) {
                cudaMemcpyAsync(slot_a.p, pinned, expert_bytes, cudaMemcpyHostToDevice, nullptr);
                ak_launch_gemv_grid(slot_a.as<float>(), dx.as<float>(), dy.as<float>(), rows, K, expert_blocks, nullptr);
            }
        };
        auto overlap_experts = [&]() {
            float* curr = slot_a.as<float>();
            float* next = slot_b.as<float>();
            cudaMemcpyAsync(curr, pinned, expert_bytes, cudaMemcpyHostToDevice, copy_stream);
            cudaEventRecord(copied, copy_stream);
            for (int e = 0; e < 4; ++e) {
                cudaStreamWaitEvent(nullptr, copied, 0);
                if (e + 1 < 4) {
                    cudaMemcpyAsync(next, pinned, expert_bytes, cudaMemcpyHostToDevice, copy_stream);
                    cudaEventRecord(copied, copy_stream);
                }
                ak_launch_gemv_grid(curr, dx.as<float>(), dy.as<float>(), rows, K, expert_blocks, nullptr);
                std::swap(curr, next);
            }
        };
        if (slot_a.p && slot_b.p && copy_stream && copied) {
            time_gpu(serial_experts, nullptr, 1, std::max(3, iters / 4), &med, &mn, &p95);
            push_row(out, "expert_serial", med, mn, p95, 0.f, 0.f, 1, std::max(3, iters / 4));
            time_gpu(overlap_experts, nullptr, 1, std::max(3, iters / 4), &med, &mn, &p95);
            cudaStreamSynchronize(copy_stream);
            push_row(out, "expert_overlap", med, mn, p95, 0.f, 0.f, 1, std::max(3, iters / 4));
        }
        cudaEventDestroy(copied);
        cudaStreamDestroy(copy_stream);
        cudaFreeHost(pinned);
    }
    return AK_OK;
}
