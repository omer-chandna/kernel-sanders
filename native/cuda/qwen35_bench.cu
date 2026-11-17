#include "error.h"
#include "ops.cuh"
#include "qgemv.cuh"

#include <adakernel/api.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

enum { kQ8 = 8, kQ4 = 12, kQ6 = 14 };

struct Proj {
    int type;
    int rows;
    int K;
    size_t offset;
    size_t bytes;
};

uint64_t ru64(std::ifstream& in) {
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), 8);
    return v;
}

uint32_t ru32(std::ifstream& in) {
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), 4);
    return v;
}

std::string rstr(std::ifstream& in) {
    const uint64_t n = ru64(in);
    std::string s(n, '\0');
    if (n) {
        in.read(s.data(), static_cast<std::streamsize>(n));
    }
    return s;
}

void skip_value(std::ifstream& in, uint32_t t) {
    if (t == 0 || t == 1 || t == 7) {
        in.seekg(1, std::ios::cur);
    } else if (t == 4 || t == 5 || t == 6) {
        in.seekg(4, std::ios::cur);
    } else if (t == 10 || t == 11 || t == 12) {
        in.seekg(8, std::ios::cur);
    } else if (t == 8) {
        rstr(in);
    } else if (t == 9) {
        const uint32_t et = ru32(in);
        const uint64_t n = ru64(in);
        for (uint64_t i = 0; i < n; ++i) {
            skip_value(in, et);
        }
    }
}

int block_bytes(int type) {
    if (type == kQ8) return 34;
    if (type == kQ4) return 144;
    if (type == kQ6) return 210;
    return 0;
}

int block_elems(int type) {
    return type == kQ8 ? 32 : 256;
}

__global__ void tile_q4(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) return;
    int nrows = rows - row0;
    if (nrows > nwarps) nrows = nwarps;
    akd::q4k_matvec_rows(W + static_cast<size_t>(row0) * ((K >> 8) * 144), x, y + row0, nrows, K);
}

__global__ void tile_q6(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) return;
    int nrows = rows - row0;
    if (nrows > nwarps) nrows = nwarps;
    akd::q6k_matvec_rows(W + static_cast<size_t>(row0) * ((K >> 8) * 210), x, y + row0, nrows, K);
}

__global__ void tile_q8(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) return;
    int nrows = rows - row0;
    if (nrows > nwarps) nrows = nwarps;
    akd::q8_matvec_rows(W + static_cast<size_t>(row0) * ((K >> 5) * 34), x, y + row0, nrows, K);
}

constexpr int kFastThreads = 128;

template <int T>
__global__ void fast_q(const uint8_t* W, const int8_t* xq, const float* xd, float* y, int rows, int K) {
    constexpr int bb = T == 0 ? 34 : (T == 1 ? 144 : 210);
    const int nb = T == 0 ? (K >> 5) : (K >> 8);
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) return;
    int nrows = rows - row0;
    if (nrows > nwarps) nrows = nwarps;
    akd::qgemv_dp4a_rows<T>(W + static_cast<size_t>(row0) * nb * bb, xq, xd, y + row0, nrows, K);
}

void quant_x(const float* x, int8_t* xq, float* xd, int K) {
    akd::quantize_q8_1_grid<<<(K / 32 + 7) / 8, 256>>>(x, xq, xd, K);
}

__global__ void dequant_x(const int8_t* xq, const float* xd, float* x, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) x[i] = xd[i >> 5] * static_cast<float>(xq[i]);
}

struct Bufs {
    const uint8_t* w;
    const float* x;
    const int8_t* xq;
    const float* xd;
    float* y;
};

void launch_old(const Proj& p, const Bufs& b) {
    const int blocks = (p.rows + 7) / 8;
    const uint8_t* w = b.w + p.offset;
    if (p.type == kQ4) tile_q4<<<blocks, 256>>>(w, b.x, b.y, p.rows, p.K);
    else if (p.type == kQ6) tile_q6<<<blocks, 256>>>(w, b.x, b.y, p.rows, p.K);
    else tile_q8<<<blocks, 256>>>(w, b.x, b.y, p.rows, p.K);
}

void launch_fast(const Proj& p, const Bufs& b) {
    const int rpb = kFastThreads / 32;
    const int blocks = (p.rows + rpb - 1) / rpb;
    const uint8_t* w = b.w + p.offset;
    if (p.type == kQ4) fast_q<1><<<blocks, kFastThreads>>>(w, b.xq, b.xd, b.y, p.rows, p.K);
    else if (p.type == kQ6) fast_q<2><<<blocks, kFastThreads>>>(w, b.xq, b.xd, b.y, p.rows, p.K);
    else fast_q<0><<<blocks, kFastThreads>>>(w, b.xq, b.xd, b.y, p.rows, p.K);
}

template <class F>
float time_ms(F fn, int iters) {
    fn();
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / static_cast<float>(iters);
}

double gbps(size_t bytes, float ms) { return ms > 0 ? bytes / (ms * 1.0e-3) / 1.0e9 : 0.0; }

}  // namespace

extern "C" ak_status ak_bench_qwen35_projections(const char* path, int iters, ak_qwen_proj_report* out) {
    if (!path || !out || iters < 1) {
        ak_set_error("qwen35 bench: bad arguments");
        return AK_ERR_INVALID;
    }
    std::memset(out, 0, sizeof(*out));
    ak_status st = ak_init(0);
    if (st != AK_OK) return st;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        ak_set_error("qwen35 bench: cannot open gguf");
        return AK_ERR_INVALID;
    }
    uint32_t magic = 0, version = 0;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    const uint64_t n_tensors = ru64(in);
    const uint64_t n_kv = ru64(in);
    if (magic != 0x46554747u) {
        ak_set_error("qwen35 bench: bad magic");
        return AK_ERR_INVALID;
    }
    for (uint64_t i = 0; i < n_kv; ++i) {
        rstr(in);
        skip_value(in, ru32(in));
    }
    std::vector<Proj> proj;
    proj.reserve(static_cast<size_t>(n_tensors));
    size_t weight_bytes = 0;
    int max_k = 1;
    int max_rows = 1;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        const std::string name = rstr(in);
        const uint32_t nd = ru32(in);
        std::vector<uint64_t> dims(nd);
        for (uint32_t d = 0; d < nd; ++d) dims[d] = ru64(in);
        const uint32_t gtype = ru32(in);
        const uint64_t offset = ru64(in);
        if (name == "token_embd.weight" || nd != 2 || (gtype != kQ4 && gtype != kQ6 && gtype != kQ8)) continue;
        const int K = static_cast<int>(dims[0]);
        const int rows = static_cast<int>(dims[1]);
        const int be = block_elems(static_cast<int>(gtype));
        if (K <= 0 || rows <= 0 || (K % be) != 0) continue;
        Proj p;
        p.type = static_cast<int>(gtype);
        p.rows = rows;
        p.K = K;
        p.offset = static_cast<size_t>(offset);
        p.bytes = static_cast<size_t>(rows) * (static_cast<size_t>(K) / be) * block_bytes(p.type);
        weight_bytes += p.bytes;
        if (K > max_k) max_k = K;
        if (rows > max_rows) max_rows = rows;
        proj.push_back(p);
        if (gtype == kQ4) out->n_q4 += 1;
        else if (gtype == kQ6) out->n_q6 += 1;
        else out->n_q8 += 1;
    }
    const std::streamoff header_end = in.tellg();
    const std::streamoff data_off = (header_end + 31) & ~std::streamoff{31};
    in.seekg(0, std::ios::end);
    const std::streamoff file_end = in.tellg();
    const size_t blob = static_cast<size_t>(file_end - data_off);
    std::vector<char> host(blob);
    in.seekg(data_off);
    in.read(host.data(), static_cast<std::streamsize>(blob));
    if (!in) {
        ak_set_error("qwen35 bench: short read");
        return AK_ERR_INVALID;
    }
    uint8_t* d_w = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    if (cudaMalloc(&d_w, blob) != cudaSuccess || cudaMalloc(&d_x, sizeof(float) * max_k) != cudaSuccess ||
        cudaMalloc(&d_y, sizeof(float) * max_rows) != cudaSuccess) {
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
        ak_set_error("qwen35 bench: out of memory");
        return AK_ERR_OOM;
    }
    cudaMemcpy(d_w, host.data(), blob, cudaMemcpyHostToDevice);
    // Deterministic, non-trivial activation with a few outliers, like a post-RMSNorm hidden state.
    std::vector<float> hx(max_k);
    uint32_t seed = 12345u;
    for (int i = 0; i < max_k; ++i) {
        seed = seed * 1664525u + 1013904223u;
        float v = (static_cast<float>(seed >> 8) / 16777216.f) * 2.f - 1.f;
        if ((i % 509) == 0) v *= 20.f;
        hx[i] = v;
    }
    cudaMemcpy(d_x, hx.data(), sizeof(float) * max_k, cudaMemcpyHostToDevice);
    int8_t* d_xq = nullptr;
    float* d_xd = nullptr;
    float* d_yref = nullptr;
    cudaMalloc(&d_xq, max_k);
    cudaMalloc(&d_xd, sizeof(float) * (max_k / 32));
    cudaMalloc(&d_yref, sizeof(float) * max_rows);
    Bufs bufs{d_w, d_x, d_xq, d_xd, d_y};
    // Quantizing a prefix of x gives the same blocks as quantizing x and taking the prefix, so one pass serves
    // every K.
    quant_x(d_x, d_xq, d_xd, max_k);

    // Correctness: every projection, new kernel against the old FP32-activation kernel. Pass 0 feeds the old
    // kernel the original x (total error, including int8 activation rounding). Pass 1 feeds it the dequantized
    // int8 x, so any remaining difference is the dp4a kernel itself.
    float* d_xdq = nullptr;
    cudaMalloc(&d_xdq, sizeof(float) * max_k);
    dequant_x<<<(max_k + 255) / 256, 256>>>(d_xq, d_xd, d_xdq, max_k);
    std::vector<float> yr(max_rows), yf(max_rows);
    float worst_pass[2] = {0.f, 0.f};
    for (int pass = 0; pass < 2; ++pass)
    for (const Proj& p : proj) {
        float& worst = worst_pass[pass];
        Bufs rb = bufs;
        rb.y = d_yref;
        if (pass == 1) rb.x = d_xdq;
        launch_old(p, rb);
        launch_fast(p, bufs);
        cudaMemcpy(yr.data(), d_yref, sizeof(float) * p.rows, cudaMemcpyDeviceToHost);
        cudaMemcpy(yf.data(), d_y, sizeof(float) * p.rows, cudaMemcpyDeviceToHost);
        double num = 0.0, den = 0.0;
        for (int r = 0; r < p.rows; ++r) {
            const double e = static_cast<double>(yf[r]) - yr[r];
            num += e * e;
            den += static_cast<double>(yr[r]) * yr[r];
        }
        const float rel = den > 0 ? static_cast<float>(std::sqrt(num / den)) : 0.f;
        if (rel > worst) worst = rel;
    }
    out->max_rel_err = worst_pass[0];
    out->kernel_rel_err = worst_pass[1];
    cudaFree(d_xdq);

    out->n_proj = static_cast<int>(proj.size());
    out->weight_gb = static_cast<float>(weight_bytes / 1.0e9);
    out->old_ms = time_ms([&]() { for (const Proj& p : proj) launch_old(p, bufs); }, iters);
    out->fast_ms = time_ms([&]() {
        quant_x(d_x, d_xq, d_xd, max_k);
        for (const Proj& p : proj) launch_fast(p, bufs);
    }, iters);
    out->quant_us = 1000.f * time_ms([&]() { quant_x(d_x, d_xq, d_xd, max_k); }, iters * 20);
    out->old_gbps = static_cast<float>(gbps(weight_bytes, out->old_ms));
    out->fast_gbps = static_cast<float>(gbps(weight_bytes, out->fast_ms));
    float* per_type[3] = {&out->q8_gbps, &out->q4_gbps, &out->q6_gbps};
    const int types[3] = {kQ8, kQ4, kQ6};
    for (int ti = 0; ti < 3; ++ti) {
        std::vector<Proj> sub;
        size_t bytes = 0;
        for (const Proj& p : proj) {
            if (p.type == types[ti]) {
                sub.push_back(p);
                bytes += p.bytes;
            }
        }
        if (sub.empty()) continue;
        const float ms = time_ms([&]() { for (const Proj& p : sub) launch_fast(p, bufs); }, iters);
        *per_type[ti] = static_cast<float>(gbps(bytes, ms));
    }
    cudaFree(d_xq);
    cudaFree(d_xd);
    cudaFree(d_yref);
    cudaError_t err = cudaDeviceSynchronize();
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return AK_ERR_CUDA;
    }
    return AK_OK;
}
