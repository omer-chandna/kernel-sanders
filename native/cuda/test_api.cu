#include "error.h"
#include "launch.h"

#include <adakernel/api.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

ak_status ck(cudaError_t err) {
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return err == cudaErrorMemoryAllocation ? AK_ERR_OOM : AK_ERR_CUDA;
    }
    return AK_OK;
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t bytes) {
        if (bytes == 0) {
            bytes = 4;
        }
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            p = nullptr;
        }
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    template <typename T>
    T* as() const {
        return reinterpret_cast<T*>(p);
    }
};

ak_status run(const ak_task& task) {
    ak_status st = ak_launch_task(&task, nullptr);
    if (st != AK_OK) {
        return st;
    }
    return ck(cudaDeviceSynchronize());
}

cublasHandle_t blas() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}

}  // namespace

extern "C" ak_status ak_test_rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps) {
    if (!x || !y || rows < 1 || cols < 1) {
        ak_set_error("rmsnorm: bad args");
        return AK_ERR_INVALID;
    }
    Dev dx(sizeof(float) * rows * cols);
    Dev dw(sizeof(float) * cols);
    Dev dy(sizeof(float) * rows * cols);
    if (!dx.p || !dw.p || !dy.p) {
        return AK_ERR_OOM;
    }
    ak_status st = ck(cudaMemcpy(dx.p, x, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    if (w) {
        st = ck(cudaMemcpy(dw.p, w, sizeof(float) * cols, cudaMemcpyHostToDevice));
        if (st != AK_OK) return st;
    } else {
        std::vector<float> ones(cols, 1.f);
        st = ck(cudaMemcpy(dw.p, ones.data(), sizeof(float) * cols, cudaMemcpyHostToDevice));
        if (st != AK_OK) return st;
    }
    for (int row = 0; row < rows; ++row) {
        ak_task task{};
        task.op = AK_OP_RMSNORM;
        task.p0 = reinterpret_cast<uint64_t>(dx.as<float>() + row * cols);
        task.p1 = reinterpret_cast<uint64_t>(dw.as<float>());
        task.p2 = reinterpret_cast<uint64_t>(dy.as<float>() + row * cols);
        task.n0 = cols;
        task.f0 = eps;
        st = ak_launch_task(&task, nullptr);
        if (st != AK_OK) return st;
    }
    st = ck(cudaDeviceSynchronize());
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(y, dy.p, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_residual(const float* a, const float* b, float* y, int n) {
    Dev da(sizeof(float) * n), db(sizeof(float) * n), dy(sizeof(float) * n);
    if (!da.p || !db.p || !dy.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(da.p, a, sizeof(float) * n, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(db.p, b, sizeof(float) * n, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task task{};
    task.op = AK_OP_RESIDUAL;
    task.p0 = reinterpret_cast<uint64_t>(da.p);
    task.p1 = reinterpret_cast<uint64_t>(db.p);
    task.p2 = reinterpret_cast<uint64_t>(dy.p);
    task.n0 = n;
    st = run(task);
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(y, dy.p, sizeof(float) * n, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_fused_add_rmsnorm(const float* x, const float* residual, const float* w, float* y_add,
                                               float* y_norm, int cols, float eps) {
    Dev dx(sizeof(float) * cols), dr(sizeof(float) * cols), dw(sizeof(float) * cols), da(sizeof(float) * cols),
        dn(sizeof(float) * cols);
    if (!dx.p || !dr.p || !dw.p || !da.p || !dn.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(dx.p, x, sizeof(float) * cols, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dr.p, residual, sizeof(float) * cols, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dw.p, w, sizeof(float) * cols, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task task{};
    task.op = AK_OP_FUSED_ADD_RMSNORM;
    task.p0 = reinterpret_cast<uint64_t>(dx.p);
    task.p1 = reinterpret_cast<uint64_t>(dr.p);
    task.p2 = reinterpret_cast<uint64_t>(dw.p);
    task.p3 = reinterpret_cast<uint64_t>(da.p);
    task.p4 = reinterpret_cast<uint64_t>(dn.p);
    task.n0 = cols;
    task.f0 = eps;
    st = run(task);
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(y_add, da.p, sizeof(float) * cols, cudaMemcpyDeviceToHost));
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(y_norm, dn.p, sizeof(float) * cols, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_rope(float* q, int n_heads, int head_dim, int pos, float theta) {
    const int n = n_heads * head_dim;
    Dev d(sizeof(float) * n);
    Dev dpos(sizeof(int));
    if (!d.p || !dpos.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(d.p, q, sizeof(float) * n, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dpos.p, &pos, sizeof(int), cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task task{};
    task.op = AK_OP_ROPE;
    task.p0 = reinterpret_cast<uint64_t>(d.p);
    task.p3 = reinterpret_cast<uint64_t>(dpos.p);
    task.n0 = n_heads;
    task.n1 = head_dim;
    task.f0 = theta;
    st = run(task);
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(q, d.p, sizeof(float) * n, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_qk_norm(float* q, const float* w, int n_heads, int head_dim, float eps) {
    const int n = n_heads * head_dim;
    Dev d(sizeof(float) * n);
    Dev dw(sizeof(float) * head_dim);
    if (!d.p || !dw.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(d.p, q, sizeof(float) * n, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dw.p, w, sizeof(float) * head_dim, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task task{};
    task.op = AK_OP_QK_NORM;
    task.p0 = reinterpret_cast<uint64_t>(d.p);
    task.p1 = reinterpret_cast<uint64_t>(dw.p);
    task.n0 = n_heads;
    task.n1 = head_dim;
    task.f0 = eps;
    st = run(task);
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(q, d.p, sizeof(float) * n, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_matvec(const float* W, const float* x, float* y, int rows, int K) {
    Dev dW(sizeof(float) * rows * K), dx(sizeof(float) * K), dy(sizeof(float) * rows);
    if (!dW.p || !dx.p || !dy.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(dW.p, W, sizeof(float) * rows * K, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dx.p, x, sizeof(float) * K, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task task{};
    task.op = AK_OP_MATVEC;
    task.p0 = reinterpret_cast<uint64_t>(dW.p);
    task.p1 = reinterpret_cast<uint64_t>(dx.p);
    task.p2 = reinterpret_cast<uint64_t>(dy.p);
    task.n0 = rows;
    task.n1 = K;
    st = run(task);
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(y, dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_cublas_matvec(const float* W, const float* x, float* y, int rows, int K) {
    Dev dW(sizeof(float) * rows * K), dx(sizeof(float) * K), dy(sizeof(float) * rows);
    if (!dW.p || !dx.p || !dy.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(dW.p, W, sizeof(float) * rows * K, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dx.p, x, sizeof(float) * K, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    const float alpha = 1.f;
    const float beta = 0.f;
    cublasStatus_t cs = cublasSgemv(blas(), CUBLAS_OP_T, K, rows, &alpha, dW.as<float>(), K, dx.as<float>(), 1, &beta,
                                    dy.as<float>(), 1);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        ak_set_error("cublasSgemv failed");
        return AK_ERR_CUDA;
    }
    st = ck(cudaDeviceSynchronize());
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(y, dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_silu_mul(const float* gate, const float* up, float* y, int n) {
    Dev dg(sizeof(float) * n), du(sizeof(float) * n), dy(sizeof(float) * n);
    if (!dg.p || !du.p || !dy.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(dg.p, gate, sizeof(float) * n, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(du.p, up, sizeof(float) * n, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task task{};
    task.op = AK_OP_SILU_MUL;
    task.p0 = reinterpret_cast<uint64_t>(dg.p);
    task.p1 = reinterpret_cast<uint64_t>(du.p);
    task.p2 = reinterpret_cast<uint64_t>(dy.p);
    task.n0 = n;
    st = run(task);
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(y, dy.p, sizeof(float) * n, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_attn(const float* q, const float* k, const float* v, float* out, int n_heads, int n_kv_heads,
                                  int head_dim, int seq, int max_seq) {
    if (seq < 1 || seq > max_seq || n_heads < 1 || (n_heads % n_kv_heads) != 0) {
        ak_set_error("attn: bad shape");
        return AK_ERR_INVALID;
    }
    const int qn = n_heads * head_dim;
    const int kvn = max_seq * n_kv_heads * head_dim;
    Dev dq(sizeof(float) * qn), dk(sizeof(float) * kvn), dv(sizeof(float) * kvn), dout(sizeof(float) * qn);
    Dev dscores(sizeof(float) * n_heads * max_seq);
    Dev dpos(sizeof(int));
    if (!dq.p || !dk.p || !dv.p || !dout.p || !dscores.p || !dpos.p) return AK_ERR_OOM;
    const int pos = seq - 1;
    ak_status st = ck(cudaMemcpy(dq.p, q, sizeof(float) * qn, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dk.p, k, sizeof(float) * kvn, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dv.p, v, sizeof(float) * kvn, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dpos.p, &pos, sizeof(int), cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    const int nrep = n_heads / n_kv_heads;
    for (int head = 0; head < n_heads; ++head) {
        ak_task task{};
        task.op = AK_OP_ATTN_HEAD;
        task.p0 = reinterpret_cast<uint64_t>(dq.as<float>() + head * head_dim);
        task.p1 = reinterpret_cast<uint64_t>(dk.p);
        task.p2 = reinterpret_cast<uint64_t>(dv.p);
        task.p3 = reinterpret_cast<uint64_t>(dout.as<float>() + head * head_dim);
        task.p4 = reinterpret_cast<uint64_t>(dpos.p);
        task.p5 = reinterpret_cast<uint64_t>(dscores.as<float>() + head * max_seq);
        task.n0 = head / nrep;
        task.n1 = head_dim;
        task.n2 = max_seq;
        task.n3 = n_kv_heads;
        st = ak_launch_task(&task, nullptr);
        if (st != AK_OK) return st;
    }
    st = ck(cudaDeviceSynchronize());
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(out, dout.p, sizeof(float) * qn, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_q8_matvec(const void* Wq, const float* x, float* y, int rows, int K) {
    if ((K % 32) != 0) {
        ak_set_error("q8 matvec: K must be a multiple of 32");
        return AK_ERR_INVALID;
    }
    const size_t bytes = static_cast<size_t>(rows) * (K / 32) * 34;
    Dev dW(bytes), dx(sizeof(float) * K), dy(sizeof(float) * rows);
    if (!dW.p || !dx.p || !dy.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(dW.p, Wq, bytes, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dx.p, x, sizeof(float) * K, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task task{};
    task.op = AK_OP_Q8_MATVEC;
    task.p0 = reinterpret_cast<uint64_t>(dW.p);
    task.p1 = reinterpret_cast<uint64_t>(dx.p);
    task.p2 = reinterpret_cast<uint64_t>(dy.p);
    task.n0 = rows;
    task.n1 = K;
    st = run(task);
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(y, dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost));
}

// type: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K. Runs the quantize task and the dp4a GEMV task through ak_launch_task.
// xq_out / xd_out receive the device-quantized activation so the caller can rebuild the exact reference.
extern "C" ak_status ak_test_qgemv_dp4a(int type, const void* Wq, const float* x, float* y, int8_t* xq_out, float* xd_out,
                                        int rows, int K) {
    const int be = type == 0 ? 32 : 256;
    const int bb = type == 0 ? 34 : (type == 1 ? 144 : 210);
    if (type < 0 || type > 2 || rows < 1 || K < be || (K % be) != 0) {
        ak_set_error("qgemv dp4a: bad type or K");
        return AK_ERR_INVALID;
    }
    const size_t bytes = static_cast<size_t>(rows) * (K / be) * bb;
    Dev dW(bytes), dx(sizeof(float) * K), dxq(K), dxd(sizeof(float) * (K / 32)), dy(sizeof(float) * rows);
    if (!dW.p || !dx.p || !dxq.p || !dxd.p || !dy.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(dW.p, Wq, bytes, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dx.p, x, sizeof(float) * K, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task q{};
    q.op = AK_OP_QUANT_Q8_1;
    q.p0 = reinterpret_cast<uint64_t>(dx.p);
    q.p1 = reinterpret_cast<uint64_t>(dxq.p);
    q.p3 = reinterpret_cast<uint64_t>(dxd.p);
    q.n0 = K;
    st = ak_launch_task(&q, nullptr);
    if (st != AK_OK) return st;
    ak_task g{};
    g.op = type == 0 ? AK_OP_Q8_DP4A : (type == 1 ? AK_OP_Q4K_DP4A : AK_OP_Q6K_DP4A);
    g.p0 = reinterpret_cast<uint64_t>(dW.p);
    g.p1 = reinterpret_cast<uint64_t>(dxq.p);
    g.p2 = reinterpret_cast<uint64_t>(dy.p);
    g.p3 = reinterpret_cast<uint64_t>(dxd.p);
    g.n0 = rows;
    g.n1 = K;
    st = ak_launch_task(&g, nullptr);
    if (st != AK_OK) return st;
    // Same task through the block-wide dispatch used by the persistent scheduler; results must agree.
    Dev dy2(sizeof(float) * rows);
    if (!dy2.p) return AK_ERR_OOM;
    g.p2 = reinterpret_cast<uint64_t>(dy2.p);
    st = run(g);
    if (st != AK_OK) return st;
    std::vector<float> a(rows), b(rows);
    st = ck(cudaMemcpy(a.data(), dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(b.data(), dy2.p, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    if (st != AK_OK) return st;
    for (int r = 0; r < rows; ++r) {
        if (a[r] != b[r]) {
            ak_set_error("qgemv dp4a: grid and single-block dispatch disagree");
            return AK_ERR_INVALID;
        }
    }
    std::memcpy(y, a.data(), sizeof(float) * rows);
    st = ck(cudaMemcpy(xq_out, dxq.p, K, cudaMemcpyDeviceToHost));
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(xd_out, dxd.p, sizeof(float) * (K / 32), cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_test_q4k_matvec(const void* Wq, const float* x, float* y, int rows, int K) {
    if ((K % 256) != 0) {
        ak_set_error("q4_K matvec: K must be a multiple of 256");
        return AK_ERR_INVALID;
    }
    const size_t bytes = static_cast<size_t>(rows) * (K / 256) * 144;
    Dev dW(bytes), dx(sizeof(float) * K), dy(sizeof(float) * rows);
    if (!dW.p || !dx.p || !dy.p) return AK_ERR_OOM;
    ak_status st = ck(cudaMemcpy(dW.p, Wq, bytes, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpy(dx.p, x, sizeof(float) * K, cudaMemcpyHostToDevice));
    if (st != AK_OK) return st;
    ak_task task{};
    task.op = AK_OP_Q4K_MATVEC;
    task.p0 = reinterpret_cast<uint64_t>(dW.p);
    task.p1 = reinterpret_cast<uint64_t>(dx.p);
    task.p2 = reinterpret_cast<uint64_t>(dy.p);
    task.n0 = rows;
    task.n1 = K;
    st = run(task);
    if (st != AK_OK) return st;
    return ck(cudaMemcpy(y, dy.p, sizeof(float) * rows, cudaMemcpyDeviceToHost));
}
