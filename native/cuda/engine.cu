#include "error.h"
#include "launch.h"
#include "ops.cuh"
#include "prefill.cuh"

#include <adakernel/api.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

struct Layer {
    float* attn_norm = nullptr;
    float* wq = nullptr;
    float* wk = nullptr;
    float* wv = nullptr;
    float* wo = nullptr;
    float* q_norm = nullptr;
    float* k_norm = nullptr;
    float* ffn_norm = nullptr;
    float* wgate = nullptr;
    float* wup = nullptr;
    float* wdown = nullptr;
    float* host_wq = nullptr;
    float* host_wk = nullptr;
    float* host_wv = nullptr;
    float* host_wo = nullptr;
    float* host_wgate = nullptr;
    float* host_wup = nullptr;
    float* host_wdown = nullptr;
    float* kc = nullptr;
    float* vc = nullptr;
    akd::AkKvView* kv_view = nullptr;
    uint16_t* kscale = nullptr;
    uint16_t* vscale = nullptr;
    size_t cache_bytes = 0;
    size_t scale_bytes = 0;
    int on_device = 1;
    size_t q_bytes = 0;
    size_t kv_bytes = 0;
    size_t o_bytes = 0;
    size_t gu_bytes = 0;
    size_t down_bytes = 0;
};

struct Schedule {
    std::vector<ak_task> tasks;
    std::vector<int> succ;
    std::vector<int> ready;
};

}  // namespace

struct ak_model {
    ak_model_config cfg{};
    Layer* layers = nullptr;
    float* tok_embd = nullptr;
    float* out_norm = nullptr;
    float* wout = nullptr;
    int tied = 0;
    float* x = nullptr;
    float* h = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn = nullptr;
    float* proj = nullptr;
    float* gate = nullptr;
    float* up = nullptr;
    float* ffn = nullptr;
    float* logits = nullptr;
    float* scores = nullptr;
    float* scratch = nullptr;
    float* pref_base = nullptr;
    float* pref_x = nullptr;
    float* pref_y = nullptr;
    float* pref_q = nullptr;
    float* pref_k = nullptr;
    float* pref_v = nullptr;
    float* pref_attn = nullptr;
    float* pref_gate = nullptr;
    float* pref_up = nullptr;
    float* pref_ffn = nullptr;
    float* pref_proj = nullptr;
    float* pref_logits = nullptr;
    int* d_prompt = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t copy_event[2] = {nullptr, nullptr};
    cudaEvent_t compute_event[2] = {nullptr, nullptr};
    float* scratch_b = nullptr;
    float* sc2_wq = nullptr;
    float* sc2_wk = nullptr;
    float* sc2_wv = nullptr;
    float* sc2_wo = nullptr;
    float* sc2_wgate = nullptr;
    float* sc2_wup = nullptr;
    float* sc2_wdown = nullptr;
    float* sc_wq = nullptr;
    float* sc_wk = nullptr;
    float* sc_wv = nullptr;
    float* sc_wo = nullptr;
    float* sc_wgate = nullptr;
    float* sc_wup = nullptr;
    float* sc_wdown = nullptr;
    int* d_token = nullptr;
    int* d_pos = nullptr;
    int* d_sample = nullptr;
    int* d_out = nullptr;
    int* d_count = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    std::vector<void*> device_mem;
    std::vector<void*> host_mem;
    size_t total_alloc = 0;
    size_t weight_bytes = 0;
    size_t host_bytes = 0;
    int n_gpu_layers = 0;
    int all_resident = 0;
    Schedule sched;
    Schedule fused_sched;
    ak_persist_buf persist{};
    int persist_ready = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int graph_kind = -1;
    int graph_nodes = 0;
    float graph_capture_ms = 0.f;
};

namespace {

ak_status ck(cudaError_t err) {
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return err == cudaErrorMemoryAllocation ? AK_ERR_OOM : AK_ERR_CUDA;
    }
    return AK_OK;
}

ak_status alloc_device(ak_model* m, void** ptr, size_t bytes, bool weight) {
    *ptr = nullptr;
    if (bytes == 0) {
        return AK_OK;
    }
    ak_status st = ck(cudaMalloc(ptr, bytes));
    if (st != AK_OK) {
        return st;
    }
    m->device_mem.push_back(*ptr);
    m->total_alloc += bytes;
    if (weight) {
        m->weight_bytes += bytes;
    }
    return AK_OK;
}

ak_status alloc_host(ak_model* m, void** ptr, size_t bytes) {
    *ptr = nullptr;
    if (bytes == 0) {
        return AK_OK;
    }
    ak_status st = ck(cudaHostAlloc(ptr, bytes, cudaHostAllocDefault));
    if (st != AK_OK) {
        return st;
    }
    m->host_mem.push_back(*ptr);
    m->host_bytes += bytes;
    return AK_OK;
}

ak_status copy_to(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    if (bytes == 0) {
        return AK_OK;
    }
    return ck(cudaMemcpy(dst, src, bytes, kind));
}

ak_status finalize_schedule(const std::vector<ak_task>& tasks, const std::vector<std::pair<int, int>>& edges, Schedule* out) {
    const int n = static_cast<int>(tasks.size());
    out->tasks = tasks;
    std::vector<int> counts(n, 0);
    for (const auto& edge : edges) {
        if (edge.first < 0 || edge.second < 0 || edge.first >= n || edge.second >= n || edge.first >= edge.second) {
            ak_set_error("task graph edge is invalid");
            return AK_ERR_INVALID;
        }
        counts[edge.first] += 1;
        out->tasks[edge.second].preds_init += 1;
    }
    int cursor = 0;
    for (int i = 0; i < n; ++i) {
        out->tasks[i].succ_begin = cursor;
        out->tasks[i].n_succ = counts[i];
        out->tasks[i].preds_left = out->tasks[i].preds_init;
        cursor += counts[i];
    }
    out->succ.assign(cursor, 0);
    std::vector<int> fill(n, 0);
    for (const auto& edge : edges) {
        out->succ[out->tasks[edge.first].succ_begin + fill[edge.first]++] = edge.second;
    }
    out->ready.clear();
    std::vector<int> left(n);
    std::vector<int> queue;
    for (int i = 0; i < n; ++i) {
        left[i] = out->tasks[i].preds_init;
        if (left[i] == 0) {
            out->ready.push_back(i);
            queue.push_back(i);
        }
    }
    size_t qh = 0;
    int seen = 0;
    while (qh < queue.size()) {
        const int id = queue[qh++];
        ++seen;
        const ak_task& task = out->tasks[id];
        for (int s = 0; s < task.n_succ; ++s) {
            const int child = out->succ[task.succ_begin + s];
            if (--left[child] == 0) {
                queue.push_back(child);
            }
        }
    }
    if (seen != n) {
        ak_set_error("task graph is not a complete DAG");
        return AK_ERR_INVALID;
    }
    return AK_OK;
}

template <typename Rec>
ak_status emit_forward(ak_model* m, Rec& rec, bool fused) {
    const ak_model_config& cfg = m->cfg;
    const int layers_n = cfg.n_layers;
    const int d = cfg.d_model;
    const int n_heads = cfg.n_heads;
    const int n_kv = cfg.n_kv_heads;
    const int hd = cfg.head_dim;
    const int ff = cfg.d_ff;
    const int vocab = cfg.vocab;
    const int q_rows = n_heads * hd;
    const int kv_rows = n_kv * hd;
    const int n_rep = n_heads / n_kv;
    const float eps = cfg.rms_eps;
    const float theta = cfg.rope_theta;
    int chunk = cfg.chunk_rows > 0 ? cfg.chunk_rows : 64;
    if (rec.full_chunks) {
        chunk = 1 << 30;
    }

    auto add = [&](ak_task task, const std::vector<int>& preds) -> int {
        task.preds_init = 0;
        return rec.add(task, preds);
    };
    auto matvec = [&](const float* W, const float* x, float* y, int rows, int K, const std::vector<int>& preds) {
        std::vector<int> ids;
        const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(K) * sizeof(float);
        // One task covering the whole matrix. Resident blocks steal row tiles.
        // Smaller products stay as 64-row chunks so independent Q, K, and V still overlap.
        if (!rec.full_chunks && bytes >= (16ull << 20)) {
            ak_task task{};
            task.op = AK_OP_MATVEC;
            task.i0 = 1;
            task.p0 = reinterpret_cast<uint64_t>(W);
            task.p1 = reinterpret_cast<uint64_t>(x);
            task.p2 = reinterpret_cast<uint64_t>(y);
            task.n0 = rows;
            task.n1 = K;
            ids.push_back(add(task, preds));
            return ids;
        }
        const int step = std::max(1, std::min(chunk, rows));
        for (int row = 0; row < rows; row += step) {
            const int rc = std::min(step, rows - row);
            ak_task task{};
            task.op = AK_OP_MATVEC;
            task.p0 = reinterpret_cast<uint64_t>(W + static_cast<size_t>(row) * static_cast<size_t>(K));
            task.p1 = reinterpret_cast<uint64_t>(x);
            task.p2 = reinterpret_cast<uint64_t>(y + row);
            task.n0 = rc;
            task.n1 = K;
            ids.push_back(add(task, preds));
        }
        return ids;
    };

    ak_task embed{};
    embed.op = AK_OP_EMBED;
    embed.p0 = reinterpret_cast<uint64_t>(m->tok_embd);
    embed.p1 = reinterpret_cast<uint64_t>(m->d_token);
    embed.p2 = reinterpret_cast<uint64_t>(m->x);
    embed.n0 = d;
    int prev = add(embed, {});
    bool h_ready = false;

    for (int layer = 0; layer < layers_n; ++layer) {
        Layer ly = rec.view(layer);
        std::vector<int> produced;
        if (!h_ready) {
            ak_task norm{};
            norm.op = AK_OP_RMSNORM;
            norm.p0 = reinterpret_cast<uint64_t>(m->x);
            norm.p1 = reinterpret_cast<uint64_t>(ly.attn_norm);
            norm.p2 = reinterpret_cast<uint64_t>(m->h);
            norm.n0 = d;
            norm.f0 = eps;
            prev = add(norm, {prev});
        }
        h_ready = false;
        const std::vector<int> h_pred{prev};
        auto q_ids = matvec(ly.wq, m->h, m->q, q_rows, d, h_pred);
        auto k_ids = matvec(ly.wk, m->h, m->k, kv_rows, d, h_pred);
        auto v_ids = matvec(ly.wv, m->h, m->v, kv_rows, d, h_pred);

        std::vector<int> q_for_rope = q_ids;
        std::vector<int> k_for_rope = k_ids;
        if (cfg.qk_norm) {
            ak_task qn{};
            qn.op = AK_OP_QK_NORM;
            qn.p0 = reinterpret_cast<uint64_t>(m->q);
            qn.p1 = reinterpret_cast<uint64_t>(ly.q_norm);
            qn.n0 = n_heads;
            qn.n1 = hd;
            qn.f0 = eps;
            q_for_rope = {add(qn, q_ids)};
            ak_task kn = qn;
            kn.p0 = reinterpret_cast<uint64_t>(m->k);
            kn.p1 = reinterpret_cast<uint64_t>(ly.k_norm);
            kn.n0 = n_kv;
            k_for_rope = {add(kn, k_ids)};
        }
        ak_task rope{};
        rope.op = AK_OP_ROPE;
        rope.p0 = reinterpret_cast<uint64_t>(m->q);
        rope.p3 = reinterpret_cast<uint64_t>(m->d_pos);
        rope.n0 = n_heads;
        rope.n1 = hd;
        rope.f0 = theta;
        const int rope_q = add(rope, q_for_rope);
        rope.p0 = reinterpret_cast<uint64_t>(m->k);
        rope.n0 = n_kv;
        const int rope_k = add(rope, k_for_rope);

        ak_task kv{};
        kv.op = AK_OP_KV_WRITE;
        kv.p0 = reinterpret_cast<uint64_t>(m->k);
        kv.p1 = reinterpret_cast<uint64_t>(m->v);
        kv.p2 = reinterpret_cast<uint64_t>(ly.kc);
        kv.p3 = reinterpret_cast<uint64_t>(ly.vc);
        kv.p4 = reinterpret_cast<uint64_t>(m->d_pos);
        kv.p6 = reinterpret_cast<uint64_t>(ly.kv_view);
        kv.n0 = n_kv;
        kv.n1 = hd;
        kv.n2 = cfg.max_seq;
        std::vector<int> kv_preds = v_ids;
        kv_preds.push_back(rope_k);
        const int kv_id = add(kv, kv_preds);

        std::vector<int> head_ids;
        head_ids.reserve(n_heads);
        for (int head = 0; head < n_heads; ++head) {
            ak_task attn{};
            attn.op = AK_OP_ATTN_HEAD;
            attn.p0 = reinterpret_cast<uint64_t>(m->q + head * hd);
            attn.p1 = reinterpret_cast<uint64_t>(ly.kc);
            attn.p2 = reinterpret_cast<uint64_t>(ly.vc);
            attn.p3 = reinterpret_cast<uint64_t>(m->attn + head * hd);
            attn.p4 = reinterpret_cast<uint64_t>(m->d_pos);
            attn.p5 = reinterpret_cast<uint64_t>(m->scores + static_cast<size_t>(head) * cfg.max_seq);
            attn.p6 = reinterpret_cast<uint64_t>(ly.kv_view);
            attn.n0 = head / n_rep;
            attn.n1 = hd;
            attn.n2 = cfg.max_seq;
            attn.n3 = n_kv;
            head_ids.push_back(add(attn, {rope_q, kv_id}));
        }
        auto o_ids = matvec(ly.wo, m->attn, m->proj, d, q_rows, head_ids);
        const bool fuse_ffn = fused;
        int ffn_src = 0;
        if (fuse_ffn) {
            ak_task fused_task{};
            fused_task.op = AK_OP_FUSED_ADD_RMSNORM;
            fused_task.p0 = reinterpret_cast<uint64_t>(m->x);
            fused_task.p1 = reinterpret_cast<uint64_t>(m->proj);
            fused_task.p2 = reinterpret_cast<uint64_t>(ly.ffn_norm);
            fused_task.p3 = reinterpret_cast<uint64_t>(m->x);
            fused_task.p4 = reinterpret_cast<uint64_t>(m->h);
            fused_task.n0 = d;
            fused_task.f0 = eps;
            ffn_src = add(fused_task, o_ids);
        } else {
            ak_task residual{};
            residual.op = AK_OP_RESIDUAL;
            residual.p0 = reinterpret_cast<uint64_t>(m->x);
            residual.p1 = reinterpret_cast<uint64_t>(m->proj);
            residual.p2 = reinterpret_cast<uint64_t>(m->x);
            residual.n0 = d;
            const int res = add(residual, o_ids);
            ak_task norm{};
            norm.op = AK_OP_RMSNORM;
            norm.p0 = reinterpret_cast<uint64_t>(m->x);
            norm.p1 = reinterpret_cast<uint64_t>(ly.ffn_norm);
            norm.p2 = reinterpret_cast<uint64_t>(m->h);
            norm.n0 = d;
            norm.f0 = eps;
            ffn_src = add(norm, {res});
        }
        auto gate_ids = matvec(ly.wgate, m->h, m->gate, ff, d, {ffn_src});
        auto up_ids = matvec(ly.wup, m->h, m->up, ff, d, {ffn_src});
        ak_task silu{};
        silu.op = AK_OP_SILU_MUL;
        silu.p0 = reinterpret_cast<uint64_t>(m->gate);
        silu.p1 = reinterpret_cast<uint64_t>(m->up);
        silu.p2 = reinterpret_cast<uint64_t>(m->ffn);
        silu.n0 = ff;
        std::vector<int> silu_preds = gate_ids;
        silu_preds.insert(silu_preds.end(), up_ids.begin(), up_ids.end());
        const int silu_id = add(silu, silu_preds);
        auto down_ids = matvec(ly.wdown, m->ffn, m->proj, d, ff, {silu_id});

        const bool last = layer + 1 == layers_n;
        if (fused) {
            ak_task fused_task{};
            fused_task.op = AK_OP_FUSED_ADD_RMSNORM;
            fused_task.p0 = reinterpret_cast<uint64_t>(m->x);
            fused_task.p1 = reinterpret_cast<uint64_t>(m->proj);
            fused_task.p2 = reinterpret_cast<uint64_t>(last ? m->out_norm : m->layers[layer + 1].attn_norm);
            fused_task.p3 = reinterpret_cast<uint64_t>(m->x);
            fused_task.p4 = reinterpret_cast<uint64_t>(m->h);
            fused_task.n0 = d;
            fused_task.f0 = eps;
            prev = add(fused_task, down_ids);
            h_ready = !last;
        } else {
            ak_task residual{};
            residual.op = AK_OP_RESIDUAL;
            residual.p0 = reinterpret_cast<uint64_t>(m->x);
            residual.p1 = reinterpret_cast<uint64_t>(m->proj);
            residual.p2 = reinterpret_cast<uint64_t>(m->x);
            residual.n0 = d;
            prev = add(residual, down_ids);
        }
    }

    if (!fused) {
        ak_task norm{};
        norm.op = AK_OP_RMSNORM;
        norm.p0 = reinterpret_cast<uint64_t>(m->x);
        norm.p1 = reinterpret_cast<uint64_t>(m->out_norm);
        norm.p2 = reinterpret_cast<uint64_t>(m->h);
        norm.n0 = d;
        norm.f0 = eps;
        prev = add(norm, {prev});
    }
    auto head_ids = matvec(m->wout, m->h, m->logits, vocab, d, {prev});
    ak_task fin{};
    fin.op = AK_OP_FINALIZE;
    fin.p0 = reinterpret_cast<uint64_t>(m->logits);
    fin.p1 = reinterpret_cast<uint64_t>(m->d_pos);
    fin.p2 = reinterpret_cast<uint64_t>(m->d_token);
    fin.p3 = reinterpret_cast<uint64_t>(m->d_out);
    fin.p4 = reinterpret_cast<uint64_t>(m->d_sample);
    fin.p5 = reinterpret_cast<uint64_t>(m->d_count);
    fin.n0 = vocab;
    add(fin, head_ids);
    return rec.status;
}

struct Recorder {
    ak_model* model = nullptr;
    bool full_chunks = false;
    ak_status status = AK_OK;
    std::vector<ak_task> tasks;
    std::vector<std::pair<int, int>> edges;
    int add(const ak_task& task, const std::vector<int>& preds) {
        const int id = static_cast<int>(tasks.size());
        tasks.push_back(task);
        for (int pred : preds) {
            if (pred < 0 || pred >= id) {
                status = AK_ERR_INVALID;
                ak_set_error("recorder received a forward edge");
            } else {
                edges.emplace_back(pred, id);
            }
        }
        return id;
    }
    Layer view(int layer) const { return model->layers[layer]; }
};

struct Immediate {
    ak_model* model = nullptr;
    cudaStream_t stream = nullptr;
    bool full_chunks = false;
    bool use_cublas = false;
    ak_status status = AK_OK;
    int launches = 0;
    int staged_layer[2] = {-1, -1};
    void bind_slot(Layer& ly, int slot) const {
        ly.wq = slot ? model->sc2_wq : model->sc_wq;
        ly.wk = slot ? model->sc2_wk : model->sc_wk;
        ly.wv = slot ? model->sc2_wv : model->sc_wv;
        ly.wo = slot ? model->sc2_wo : model->sc_wo;
        ly.wgate = slot ? model->sc2_wgate : model->sc_wgate;
        ly.wup = slot ? model->sc2_wup : model->sc_wup;
        ly.wdown = slot ? model->sc2_wdown : model->sc_wdown;
        ly.on_device = 1;
    }
    ak_status stage_slot(int layer, int slot, cudaStream_t copy_stream, bool wait_for_reader) {
        if (wait_for_reader && model->compute_event[slot]) {
            cudaError_t err = cudaEventRecord(model->compute_event[slot], stream);
            if (err != cudaSuccess) {
                status = AK_ERR_CUDA;
                ak_set_error(cudaGetErrorString(err));
                return status;
            }
            err = cudaStreamWaitEvent(copy_stream, model->compute_event[slot], 0);
            if (err != cudaSuccess) {
                status = AK_ERR_CUDA;
                ak_set_error(cudaGetErrorString(err));
                return status;
            }
        }
        const Layer& src = model->layers[layer];
        auto stage = [&](float* dst, float* host, size_t bytes) {
            if (status == AK_OK && bytes > 0) {
                status = ck(cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, copy_stream));
            }
        };
        stage(slot ? model->sc2_wq : model->sc_wq, src.host_wq, src.q_bytes);
        stage(slot ? model->sc2_wk : model->sc_wk, src.host_wk, src.kv_bytes);
        stage(slot ? model->sc2_wv : model->sc_wv, src.host_wv, src.kv_bytes);
        stage(slot ? model->sc2_wo : model->sc_wo, src.host_wo, src.o_bytes);
        stage(slot ? model->sc2_wgate : model->sc_wgate, src.host_wgate, src.gu_bytes);
        stage(slot ? model->sc2_wup : model->sc_wup, src.host_wup, src.gu_bytes);
        stage(slot ? model->sc2_wdown : model->sc_wdown, src.host_wdown, src.down_bytes);
        if (status == AK_OK) {
            cudaError_t err = cudaEventRecord(model->copy_event[slot], copy_stream);
            if (err != cudaSuccess) {
                status = AK_ERR_CUDA;
                ak_set_error(cudaGetErrorString(err));
            } else {
                staged_layer[slot] = layer;
            }
        }
        return status;
    }
    int add(const ak_task& task, const std::vector<int>&) {
        if (status != AK_OK) {
            return 0;
        }
        if (use_cublas && task.op == AK_OP_MATVEC) {
            const int rows = task.n0;
            const int K = task.n1;
            const float alpha = 1.f;
            const float beta = 0.f;
            cublasSetStream(model->cublas, stream);
            cublasStatus_t cs = cublasSgemv(model->cublas, CUBLAS_OP_T, K, rows, &alpha,
                                            reinterpret_cast<const float*>(task.p0), K,
                                            reinterpret_cast<const float*>(task.p1), 1, &beta,
                                            reinterpret_cast<float*>(task.p2), 1);
            ++launches;
            if (cs != CUBLAS_STATUS_SUCCESS) {
                status = AK_ERR_CUDA;
                ak_set_error("cublasSgemv failed");
            }
            return 0;
        }
        status = ak_launch_task(&task, stream);
        ++launches;
        return 0;
    }
    Layer view(int layer) {
        Layer ly = model->layers[layer];
        if (ly.on_device || status != AK_OK) {
            return ly;
        }
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        const bool overlap = model->cfg.offload_overlap && model->copy_stream && model->copy_event[0] &&
                             cap == cudaStreamCaptureStatusNone;
        if (overlap) {
            const int slot = layer & 1;
            if (staged_layer[slot] != layer) {
                stage_slot(layer, slot, model->copy_stream, true);
            }
            if (status == AK_OK) {
                cudaError_t err = cudaStreamWaitEvent(stream, model->copy_event[slot], 0);
                if (err != cudaSuccess) {
                    status = AK_ERR_CUDA;
                    ak_set_error(cudaGetErrorString(err));
                }
            }
            bind_slot(ly, slot);
            int nxt = layer + 1;
            while (nxt < model->cfg.n_layers && model->layers[nxt].on_device) {
                ++nxt;
            }
            if (nxt < model->cfg.n_layers && staged_layer[nxt & 1] != nxt) {
                stage_slot(nxt, nxt & 1, model->copy_stream, true);
            }
            return ly;
        }
        auto stage = [&](float* dst, float* src, size_t bytes) {
            if (status == AK_OK && bytes > 0) {
                status = ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
            }
        };
        stage(model->sc_wq, ly.host_wq, ly.q_bytes);
        stage(model->sc_wk, ly.host_wk, ly.kv_bytes);
        stage(model->sc_wv, ly.host_wv, ly.kv_bytes);
        stage(model->sc_wo, ly.host_wo, ly.o_bytes);
        stage(model->sc_wgate, ly.host_wgate, ly.gu_bytes);
        stage(model->sc_wup, ly.host_wup, ly.gu_bytes);
        stage(model->sc_wdown, ly.host_wdown, ly.down_bytes);
        ly.wq = model->sc_wq;
        ly.wk = model->sc_wk;
        ly.wv = model->sc_wv;
        ly.wo = model->sc_wo;
        ly.wgate = model->sc_wgate;
        ly.wup = model->sc_wup;
        ly.wdown = model->sc_wdown;
        ly.on_device = 1;
        return ly;
    }
};

ak_status build_recorded(ak_model* m, bool fused, Schedule* sched) {
    Recorder rec;
    rec.model = m;
    rec.full_chunks = false;
    ak_status st = emit_forward(m, rec, fused);
    if (st != AK_OK) {
        return st;
    }
    if (rec.status != AK_OK) {
        return rec.status;
    }
    return finalize_schedule(rec.tasks, rec.edges, sched);
}

ak_status launch_immediate(ak_model* m, cudaStream_t stream, bool fused, bool cublas, int* launches) {
    Immediate now;
    now.model = m;
    now.stream = stream;
    now.full_chunks = cublas;
    now.use_cublas = cublas;
    ak_status st = emit_forward(m, now, fused);
    if (launches) {
        *launches = now.launches;
    }
    if (st != AK_OK) {
        return st;
    }
    return now.status;
}

ak_status capture_graph(ak_model* m, int kind) {
    if (m->graph_exec && m->graph_kind == kind) {
        return AK_OK;
    }
    if (m->graph_exec) {
        cudaGraphExecDestroy(m->graph_exec);
        m->graph_exec = nullptr;
    }
    if (m->graph) {
        cudaGraphDestroy(m->graph);
        m->graph = nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ak_status st = ck(cudaStreamSynchronize(m->stream));
    if (st != AK_OK) {
        return st;
    }
    st = ck(cudaStreamBeginCapture(m->stream, cudaStreamCaptureModeGlobal));
    if (st != AK_OK) {
        return st;
    }
    int launches = 0;
    if (kind == AK_BACKEND_HYBRID) {
        st = ak_persist_launch(&m->persist, 1000000000, m->stream);
    } else if (kind == AK_BACKEND_CUBLAS_GRAPH) {
        st = launch_immediate(m, m->stream, false, true, &launches);
    } else {
        st = launch_immediate(m, m->stream, false, false, &launches);
    }
    cudaError_t end = cudaStreamEndCapture(m->stream, &m->graph);
    if (st != AK_OK) {
        cudaGetLastError();
        return st;
    }
    if (end != cudaSuccess) {
        ak_set_error(cudaGetErrorString(end));
        return AK_ERR_CUDA;
    }
    end = cudaGraphInstantiate(&m->graph_exec, m->graph, 0);
    if (end != cudaSuccess) {
        ak_set_error(cudaGetErrorString(end));
        cudaGraphDestroy(m->graph);
        m->graph = nullptr;
        return AK_ERR_CUDA;
    }
    m->graph_kind = kind;
    m->graph_nodes = launches > 0 ? launches : m->sched.tasks.size();
    if (kind == AK_BACKEND_HYBRID) {
        m->graph_nodes = 2;
    }
    const auto t1 = std::chrono::steady_clock::now();
    m->graph_capture_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return AK_OK;
}

ak_status ensure_persistent(ak_model* m) {
    if (m->persist_ready) {
        return AK_OK;
    }
    if (!m->all_resident) {
        ak_set_error("persistent backend needs every weight layer resident on the GPU");
        return AK_ERR_UNSUPPORTED;
    }
    if (m->sched.tasks.empty()) {
        ak_status st = build_recorded(m, false, &m->sched);
        if (st != AK_OK) {
            return st;
        }
    }
    ak_status st = ak_persist_alloc(&m->persist, static_cast<int>(m->sched.tasks.size()),
                                    static_cast<int>(m->sched.succ.size()), static_cast<int>(m->sched.ready.size()),
                                    m->cfg.persistent_blocks);
    if (st != AK_OK) {
        return st;
    }
    st = ak_persist_upload(&m->persist, m->sched.tasks.data(), m->sched.succ.data(), m->sched.ready.data());
    if (st != AK_OK) {
        return st;
    }
    m->persist_ready = 1;
    return AK_OK;
}

ak_status run_backend(ak_model* m, int backend, int* launches) {
    if (backend == AK_BACKEND_PERSISTENT || backend == AK_BACKEND_HYBRID) {
        ak_status st = ensure_persistent(m);
        if (st != AK_OK) {
            return st;
        }
    }
    if (backend == AK_BACKEND_CUDA_GRAPH || backend == AK_BACKEND_CUBLAS_GRAPH || backend == AK_BACKEND_HYBRID) {
        ak_status st = capture_graph(m, backend);
        if (st != AK_OK) {
            return st;
        }
        st = ck(cudaGraphLaunch(m->graph_exec, m->stream));
        if (launches) {
            *launches = 1;
        }
        return st;
    }
    if (backend == AK_BACKEND_PERSISTENT) {
        ak_status st = ak_persist_launch(&m->persist, 1000000000, m->stream);
        if (launches) {
            *launches = 2;
        }
        return st;
    }
    const bool fused = backend == AK_BACKEND_FUSED;
    const bool cublas = backend == AK_BACKEND_CUBLAS;
    return launch_immediate(m, m->stream, fused, cublas, launches);
}

ak_status reset_token_state(ak_model* m) {
    const size_t kv_fallback = static_cast<size_t>(m->cfg.max_seq) * m->cfg.n_kv_heads * m->cfg.head_dim * sizeof(float);
    for (int i = 0; i < m->cfg.n_layers; ++i) {
        const size_t bytes = m->layers[i].cache_bytes > 0 ? m->layers[i].cache_bytes : kv_fallback;
        ak_status st = ck(cudaMemsetAsync(m->layers[i].kc, 0, bytes, m->stream));
        if (st != AK_OK) {
            return st;
        }
        st = ck(cudaMemsetAsync(m->layers[i].vc, 0, bytes, m->stream));
        if (st != AK_OK) {
            return st;
        }
        if (m->layers[i].scale_bytes > 0) {
            st = ck(cudaMemsetAsync(m->layers[i].kscale, 0, m->layers[i].scale_bytes, m->stream));
            if (st != AK_OK) return st;
            st = ck(cudaMemsetAsync(m->layers[i].vscale, 0, m->layers[i].scale_bytes, m->stream));
            if (st != AK_OK) return st;
        }
    }
    ak_status st = ck(cudaMemsetAsync(m->d_pos, 0, sizeof(int), m->stream));
    if (st != AK_OK) {
        return st;
    }
    return ck(cudaMemsetAsync(m->d_count, 0, sizeof(int), m->stream));
}

float elapsed_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

}  // namespace

extern "C" ak_status ak_model_destroy(ak_model* model) {
    if (!model) {
        return AK_OK;
    }
    if (model->graph_exec) {
        cudaGraphExecDestroy(model->graph_exec);
    }
    if (model->graph) {
        cudaGraphDestroy(model->graph);
    }
    ak_persist_free(&model->persist);
    if (model->cublas) {
        cublasDestroy(model->cublas);
    }
    if (model->stream) {
        cudaStreamDestroy(model->stream);
    }
    if (model->copy_stream) {
        cudaStreamSynchronize(model->copy_stream);
        cudaStreamDestroy(model->copy_stream);
    }
    for (cudaEvent_t event : model->copy_event) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    for (cudaEvent_t event : model->compute_event) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    for (void* ptr : model->device_mem) {
        cudaFree(ptr);
    }
    for (void* ptr : model->host_mem) {
        cudaFreeHost(ptr);
    }
    delete[] model->layers;
    delete model;
    return AK_OK;
}

extern "C" ak_status ak_model_create(const ak_model_config* cfg, const ak_fp32_weights* weights, ak_model** out) {
    if (!cfg || !weights || !out) {
        ak_set_error("model_create: null argument");
        return AK_ERR_INVALID;
    }
    *out = nullptr;
    if (cfg->n_layers < 1 || cfg->n_layers > 128 || cfg->d_model < 1 || cfg->n_heads < 1 || cfg->n_kv_heads < 1 ||
        cfg->head_dim < 1 || cfg->d_ff < 1 || cfg->vocab < 1 || cfg->max_seq < 2) {
        ak_set_error("model_create: configuration is out of range");
        return AK_ERR_INVALID;
    }
    if (cfg->n_heads * cfg->head_dim != cfg->d_model || (cfg->n_heads % cfg->n_kv_heads) != 0) {
        ak_set_error("model_create: head configuration does not match d_model");
        return AK_ERR_INVALID;
    }
    if ((cfg->head_dim % 2) != 0) {
        ak_set_error("model_create: head_dim must be even for RoPE");
        return AK_ERR_INVALID;
    }
    if (!weights->tok_embd || !weights->attn_norm || !weights->wq || !weights->wk || !weights->wv || !weights->wo ||
        !weights->ffn_norm || !weights->wgate || !weights->wup || !weights->wdown || !weights->out_norm) {
        ak_set_error("model_create: missing weights");
        return AK_ERR_INVALID;
    }
    if (cfg->qk_norm && (!weights->q_norm || !weights->k_norm)) {
        ak_set_error("model_create: qk norm weights are required");
        return AK_ERR_INVALID;
    }
    if (!weights->tied_embeddings && !weights->wout) {
        ak_set_error("model_create: output weights are required when embeddings are not tied");
        return AK_ERR_INVALID;
    }
    ak_status st = ak_init(0);
    if (st != AK_OK) {
        return st;
    }
    ak_model* model = new ak_model();
    model->cfg = *cfg;
    if (model->cfg.rms_eps <= 0.f) {
        model->cfg.rms_eps = 1.0e-6f;
    }
    if (model->cfg.rope_theta <= 0.f) {
        model->cfg.rope_theta = 10000.f;
    }
    if (model->cfg.chunk_rows <= 0) {
        model->cfg.chunk_rows = 64;
    }
    if (model->cfg.persistent_blocks < 0) {
        model->cfg.persistent_blocks = 0;
    }
    const int q_rows = cfg->n_heads * cfg->head_dim;
    const int kv_rows = cfg->n_kv_heads * cfg->head_dim;
    const size_t q_bytes = static_cast<size_t>(q_rows) * cfg->d_model * sizeof(float);
    const size_t kv_w_bytes = static_cast<size_t>(kv_rows) * cfg->d_model * sizeof(float);
    const size_t o_bytes = static_cast<size_t>(cfg->d_model) * q_rows * sizeof(float);
    const size_t gu_bytes = static_cast<size_t>(cfg->d_ff) * cfg->d_model * sizeof(float);
    const size_t down_bytes = static_cast<size_t>(cfg->d_model) * cfg->d_ff * sizeof(float);
    const size_t layer_bytes = q_bytes + 2 * kv_w_bytes + o_bytes + 2 * gu_bytes + down_bytes;
    const size_t emb_bytes = static_cast<size_t>(cfg->vocab) * cfg->d_model * sizeof(float);
    const size_t norm_bytes = static_cast<size_t>(cfg->d_model) * sizeof(float);
    const size_t qk_bytes = static_cast<size_t>(cfg->head_dim) * sizeof(float);
    const size_t kv_cache_one = static_cast<size_t>(cfg->max_seq) * cfg->n_kv_heads * cfg->head_dim * sizeof(float);

    size_t free_b = 0;
    size_t total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t budget = static_cast<size_t>(static_cast<double>(free_b) * 0.82);
    const size_t fixed = emb_bytes + (weights->tied_embeddings ? 0 : emb_bytes) +
                         static_cast<size_t>(cfg->n_layers) * (norm_bytes * 2 + (cfg->qk_norm ? 2 * qk_bytes : 0)) +
                         norm_bytes + static_cast<size_t>(cfg->n_layers) * 2 * kv_cache_one + layer_bytes +
                         (static_cast<size_t>(cfg->d_model) * 8 + static_cast<size_t>(cfg->d_ff) * 3 + cfg->vocab +
                          static_cast<size_t>(cfg->n_heads) * cfg->max_seq) *
                             sizeof(float);
    int fit = 0;
    if (fixed < budget && layer_bytes > 0) {
        fit = static_cast<int>(std::min<size_t>(cfg->n_layers, (budget - fixed) / layer_bytes));
    }
    int want = cfg->n_gpu_layers < 0 ? cfg->n_layers : cfg->n_gpu_layers;
    if (want > cfg->n_layers) {
        want = cfg->n_layers;
    }
    if (want < 0) {
        want = 0;
    }
    model->n_gpu_layers = std::min(want, std::max(fit, 0));
    if (fixed >= budget) {
        ak_set_error("not enough free VRAM for activations, KV cache, and one layer of scratch");
        ak_model_destroy(model);
        return AK_ERR_OOM;
    }

    auto fail = [&](ak_status code) {
        ak_model_destroy(model);
        return code;
    };
    if (cublasCreate(&model->cublas) != CUBLAS_STATUS_SUCCESS) {
        ak_set_error("cublasCreate failed");
        return fail(AK_ERR_CUDA);
    }
    st = ck(cudaStreamCreateWithFlags(&model->stream, cudaStreamNonBlocking));
    if (st != AK_OK) {
        return fail(st);
    }
    st = ck(cudaStreamCreateWithFlags(&model->copy_stream, cudaStreamNonBlocking));
    if (st != AK_OK) {
        return fail(st);
    }
    for (int i = 0; i < 2; ++i) {
        st = ck(cudaEventCreateWithFlags(&model->copy_event[i], cudaEventDisableTiming));
        if (st != AK_OK) {
            return fail(st);
        }
        st = ck(cudaEventCreateWithFlags(&model->compute_event[i], cudaEventDisableTiming));
        if (st != AK_OK) {
            return fail(st);
        }
    }
    cublasSetStream(model->cublas, model->stream);

    model->layers = new Layer[cfg->n_layers]();
    st = alloc_device(model, reinterpret_cast<void**>(&model->tok_embd), emb_bytes, true);
    if (st != AK_OK) {
        return fail(st);
    }
    st = copy_to(model->tok_embd, weights->tok_embd, emb_bytes, cudaMemcpyHostToDevice);
    if (st != AK_OK) {
        return fail(st);
    }
    model->tied = weights->tied_embeddings ? 1 : 0;
    if (model->tied) {
        model->wout = model->tok_embd;
    } else {
        st = alloc_device(model, reinterpret_cast<void**>(&model->wout), emb_bytes, true);
        if (st != AK_OK) {
            return fail(st);
        }
        st = copy_to(model->wout, weights->wout, emb_bytes, cudaMemcpyHostToDevice);
        if (st != AK_OK) {
            return fail(st);
        }
    }
    st = alloc_device(model, reinterpret_cast<void**>(&model->out_norm), norm_bytes, true);
    if (st != AK_OK) {
        return fail(st);
    }
    st = copy_to(model->out_norm, weights->out_norm, norm_bytes, cudaMemcpyHostToDevice);
    if (st != AK_OK) {
        return fail(st);
    }

    auto upload_small = [&](float** dst, const float* src, size_t bytes) -> ak_status {
        ak_status s = alloc_device(model, reinterpret_cast<void**>(dst), bytes, true);
        if (s != AK_OK) {
            return s;
        }
        return copy_to(*dst, src, bytes, cudaMemcpyHostToDevice);
    };
    auto upload_big = [&](float** dev, float** host, const float* src, size_t bytes, bool resident) -> ak_status {
        if (resident) {
            ak_status s = alloc_device(model, reinterpret_cast<void**>(dev), bytes, true);
            if (s != AK_OK) {
                return s;
            }
            return copy_to(*dev, src, bytes, cudaMemcpyHostToDevice);
        }
        ak_status s = alloc_host(model, reinterpret_cast<void**>(host), bytes);
        if (s != AK_OK) {
            return s;
        }
        std::memcpy(*host, src, bytes);
        return AK_OK;
    };

    for (int layer = 0; layer < cfg->n_layers; ++layer) {
        Layer& ly = model->layers[layer];
        ly.on_device = layer < model->n_gpu_layers ? 1 : 0;
        ly.q_bytes = q_bytes;
        ly.kv_bytes = kv_w_bytes;
        ly.o_bytes = o_bytes;
        ly.gu_bytes = gu_bytes;
        ly.down_bytes = down_bytes;
        const bool resident = ly.on_device != 0;
        st = upload_small(&ly.attn_norm, weights->attn_norm + static_cast<size_t>(layer) * cfg->d_model, norm_bytes);
        if (st != AK_OK) {
            return fail(st);
        }
        st = upload_small(&ly.ffn_norm, weights->ffn_norm + static_cast<size_t>(layer) * cfg->d_model, norm_bytes);
        if (st != AK_OK) {
            return fail(st);
        }
        if (cfg->qk_norm) {
            st = upload_small(&ly.q_norm, weights->q_norm + static_cast<size_t>(layer) * cfg->head_dim, qk_bytes);
            if (st != AK_OK) {
                return fail(st);
            }
            st = upload_small(&ly.k_norm, weights->k_norm + static_cast<size_t>(layer) * cfg->head_dim, qk_bytes);
            if (st != AK_OK) {
                return fail(st);
            }
        }
        st = upload_big(&ly.wq, &ly.host_wq, weights->wq + static_cast<size_t>(layer) * q_rows * cfg->d_model, q_bytes, resident);
        if (st != AK_OK) {
            return fail(st);
        }
        st = upload_big(&ly.wk, &ly.host_wk, weights->wk + static_cast<size_t>(layer) * kv_rows * cfg->d_model, kv_w_bytes, resident);
        if (st != AK_OK) {
            return fail(st);
        }
        st = upload_big(&ly.wv, &ly.host_wv, weights->wv + static_cast<size_t>(layer) * kv_rows * cfg->d_model, kv_w_bytes, resident);
        if (st != AK_OK) {
            return fail(st);
        }
        st = upload_big(&ly.wo, &ly.host_wo, weights->wo + static_cast<size_t>(layer) * cfg->d_model * q_rows, o_bytes, resident);
        if (st != AK_OK) {
            return fail(st);
        }
        st = upload_big(&ly.wgate, &ly.host_wgate, weights->wgate + static_cast<size_t>(layer) * cfg->d_ff * cfg->d_model, gu_bytes, resident);
        if (st != AK_OK) {
            return fail(st);
        }
        st = upload_big(&ly.wup, &ly.host_wup, weights->wup + static_cast<size_t>(layer) * cfg->d_ff * cfg->d_model, gu_bytes, resident);
        if (st != AK_OK) {
            return fail(st);
        }
        st = upload_big(&ly.wdown, &ly.host_wdown, weights->wdown + static_cast<size_t>(layer) * cfg->d_model * cfg->d_ff, down_bytes, resident);
        if (st != AK_OK) {
            return fail(st);
        }
        const int kv_type = model->cfg.kv_type;
        const int page = model->cfg.page_tokens > 0 ? model->cfg.page_tokens : 16;
        const int n_pages = (cfg->max_seq + page - 1) / page;
        if (kv_type == 0) {
            ly.cache_bytes = kv_cache_one;
            st = alloc_device(model, reinterpret_cast<void**>(&ly.kc), ly.cache_bytes, false);
            if (st != AK_OK) return fail(st);
            st = alloc_device(model, reinterpret_cast<void**>(&ly.vc), ly.cache_bytes, false);
            if (st != AK_OK) return fail(st);
        } else {
            int* pages = nullptr;
            st = alloc_device(model, reinterpret_cast<void**>(&pages), sizeof(int) * n_pages, false);
            if (st != AK_OK) return fail(st);
            std::vector<int> identity(n_pages);
            for (int p = 0; p < n_pages; ++p) identity[p] = p;
            st = copy_to(pages, identity.data(), sizeof(int) * n_pages, cudaMemcpyHostToDevice);
            if (st != AK_OK) return fail(st);
            uint16_t* kscale = nullptr;
            uint16_t* vscale = nullptr;
            const size_t elems = static_cast<size_t>(cfg->max_seq) * cfg->n_kv_heads * cfg->head_dim;
            if (kv_type == 1) {
                ly.cache_bytes = elems * sizeof(uint16_t);
                st = alloc_device(model, reinterpret_cast<void**>(&ly.kc), ly.cache_bytes, false);
                if (st != AK_OK) return fail(st);
                st = alloc_device(model, reinterpret_cast<void**>(&ly.vc), ly.cache_bytes, false);
                if (st != AK_OK) return fail(st);
            } else {
                ly.cache_bytes = elems;
                st = alloc_device(model, reinterpret_cast<void**>(&ly.kc), ly.cache_bytes, false);
                if (st != AK_OK) return fail(st);
                st = alloc_device(model, reinterpret_cast<void**>(&ly.vc), ly.cache_bytes, false);
                if (st != AK_OK) return fail(st);
                ly.scale_bytes = static_cast<size_t>(cfg->max_seq) * cfg->n_kv_heads * sizeof(uint16_t);
                st = alloc_device(model, reinterpret_cast<void**>(&kscale), ly.scale_bytes, false);
                if (st != AK_OK) return fail(st);
                st = alloc_device(model, reinterpret_cast<void**>(&vscale), ly.scale_bytes, false);
                if (st != AK_OK) return fail(st);
                ly.kscale = kscale;
                ly.vscale = vscale;
            }
            akd::AkKvView host_view{};
            host_view.type = kv_type;
            host_view.page = page;
            host_view.n_kv = cfg->n_kv_heads;
            host_view.hd = cfg->head_dim;
            host_view.n_pages = n_pages;
            host_view.pages = pages;
            host_view.k = ly.kc;
            host_view.v = ly.vc;
            host_view.kscale = kscale;
            host_view.vscale = vscale;
            st = alloc_device(model, reinterpret_cast<void**>(&ly.kv_view), sizeof(akd::AkKvView), false);
            if (st != AK_OK) return fail(st);
            st = copy_to(ly.kv_view, &host_view, sizeof(akd::AkKvView), cudaMemcpyHostToDevice);
            if (st != AK_OK) return fail(st);
        }
    }

    const size_t scratch_bytes = layer_bytes + 4096;
    st = alloc_device(model, reinterpret_cast<void**>(&model->scratch), scratch_bytes, false);
    if (st != AK_OK) {
        return fail(st);
    }
    size_t off = 0;
    auto take = [&](size_t bytes) -> float* {
        off = (off + 255u) & ~size_t{255};
        float* ptr = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(model->scratch) + off);
        off += bytes;
        return ptr;
    };
    model->sc_wq = take(q_bytes);
    model->sc_wk = take(kv_w_bytes);
    model->sc_wv = take(kv_w_bytes);
    model->sc_wo = take(o_bytes);
    model->sc_wgate = take(gu_bytes);
    model->sc_wup = take(gu_bytes);
    model->sc_wdown = take(down_bytes);
    if (off > scratch_bytes) {
        ak_set_error("scratch arena overflow");
        return fail(AK_ERR_INVALID);
    }

    auto act = [&](float** ptr, size_t n) -> ak_status {
        return alloc_device(model, reinterpret_cast<void**>(ptr), n * sizeof(float), false);
    };
    st = act(&model->x, cfg->d_model);
    if (st == AK_OK) st = act(&model->h, cfg->d_model);
    if (st == AK_OK) st = act(&model->q, q_rows);
    if (st == AK_OK) st = act(&model->k, kv_rows);
    if (st == AK_OK) st = act(&model->v, kv_rows);
    if (st == AK_OK) st = act(&model->attn, q_rows);
    if (st == AK_OK) st = act(&model->proj, cfg->d_model);
    if (st == AK_OK) st = act(&model->gate, cfg->d_ff);
    if (st == AK_OK) st = act(&model->up, cfg->d_ff);
    if (st == AK_OK) st = act(&model->ffn, cfg->d_ff);
    if (st == AK_OK) st = act(&model->logits, cfg->vocab);
    if (st == AK_OK) st = act(&model->scores, static_cast<size_t>(cfg->n_heads) * cfg->max_seq);
    if (st != AK_OK) {
        return fail(st);
    }
    auto di = [&](int** ptr) -> ak_status {
        return alloc_device(model, reinterpret_cast<void**>(ptr), sizeof(int) * (ptr == &model->d_out ? cfg->max_seq : 1), false);
    };
    st = di(&model->d_token);
    if (st == AK_OK) st = di(&model->d_pos);
    if (st == AK_OK) st = di(&model->d_sample);
    if (st == AK_OK) st = di(&model->d_count);
    if (st == AK_OK) st = di(&model->d_out);
    if (st == AK_OK) {
        st = alloc_device(model, reinterpret_cast<void**>(&model->d_prompt), sizeof(int) * cfg->max_seq, false);
    }
    if (st == AK_OK) st = act(&model->pref_x, static_cast<size_t>(cfg->max_seq) * cfg->d_model);
    if (st == AK_OK) st = act(&model->pref_y, static_cast<size_t>(cfg->max_seq) * cfg->d_model);
    if (st == AK_OK) st = act(&model->pref_q, static_cast<size_t>(cfg->max_seq) * q_rows);
    if (st == AK_OK) st = act(&model->pref_k, static_cast<size_t>(cfg->max_seq) * kv_rows);
    if (st == AK_OK) st = act(&model->pref_v, static_cast<size_t>(cfg->max_seq) * kv_rows);
    if (st == AK_OK) st = act(&model->pref_attn, static_cast<size_t>(cfg->max_seq) * q_rows);
    if (st == AK_OK) st = act(&model->pref_gate, static_cast<size_t>(cfg->max_seq) * cfg->d_ff);
    if (st == AK_OK) st = act(&model->pref_up, static_cast<size_t>(cfg->max_seq) * cfg->d_ff);
    if (st == AK_OK) st = act(&model->pref_ffn, static_cast<size_t>(cfg->max_seq) * cfg->d_ff);
    if (st == AK_OK) st = act(&model->pref_proj, static_cast<size_t>(cfg->max_seq) * cfg->d_model);
    if (st == AK_OK) st = act(&model->pref_logits, static_cast<size_t>(cfg->max_seq) * cfg->vocab);
    if (st != AK_OK) {
        return fail(st);
    }
    st = alloc_device(model, reinterpret_cast<void**>(&model->scratch_b), scratch_bytes, false);
    if (st != AK_OK) {
        return fail(st);
    }
    {
        size_t off2 = 0;
        auto take2 = [&](size_t bytes) -> float* {
            off2 = (off2 + 255u) & ~size_t{255};
            float* ptr = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(model->scratch_b) + off2);
            off2 += bytes;
            return ptr;
        };
        model->sc2_wq = take2(q_bytes);
        model->sc2_wk = take2(kv_w_bytes);
        model->sc2_wv = take2(kv_w_bytes);
        model->sc2_wo = take2(o_bytes);
        model->sc2_wgate = take2(gu_bytes);
        model->sc2_wup = take2(gu_bytes);
        model->sc2_wdown = take2(down_bytes);
    }
    model->all_resident = model->n_gpu_layers == cfg->n_layers ? 1 : 0;
    model->cfg.n_gpu_layers = model->n_gpu_layers;
    {
        const float alpha = 1.f;
        const float beta = 0.f;
        const float* warm_w = model->layers[0].on_device ? model->layers[0].wq : model->sc_wq;
        if (!model->layers[0].on_device) {
            cudaMemcpyAsync(model->sc_wq, model->layers[0].host_wq, q_bytes, cudaMemcpyHostToDevice, model->stream);
        }
        cublasSgemv(model->cublas, CUBLAS_OP_T, cfg->d_model, q_rows, &alpha, warm_w, cfg->d_model, model->x, 1, &beta,
                    model->proj, 1);
        cudaStreamSynchronize(model->stream);
        cudaGetLastError();
    }
    *out = model;
    return AK_OK;
}

ak_status run_prefill(ak_model* model, const int32_t* prompt, int n_prompt, bool sample_token) {
    if (!model->all_resident) {
        return AK_ERR_UNSUPPORTED;
    }
    std::vector<PrefillLayer> layers(model->cfg.n_layers);
    for (int i = 0; i < model->cfg.n_layers; ++i) {
        const Layer& src = model->layers[i];
        layers[i] = PrefillLayer{src.attn_norm, src.wq, src.wk, src.wv, src.wo, src.q_norm, src.k_norm,
                                 src.ffn_norm, src.wgate, src.wup, src.wdown, src.kc, src.vc, src.kv_view};
    }
    ak_status st = ck(cudaMemcpyAsync(model->d_prompt, prompt, sizeof(int) * n_prompt, cudaMemcpyHostToDevice, model->stream));
    if (st != AK_OK) {
        return st;
    }
    PrefillBuffers buf{model->pref_x, model->pref_y, model->pref_q, model->pref_k, model->pref_v, model->pref_attn,
                       model->pref_gate, model->pref_up, model->pref_ffn, model->pref_proj, model->logits,
                       model->pref_logits};
    st = ak_prefill_extend(model->cublas, model->stream, &model->cfg, layers.data(), model->tok_embd, model->out_norm,
                           model->wout, model->d_prompt, n_prompt, 0, buf, true);
    if (st != AK_OK) {
        return st;
    }
    const int pos = sample_token ? n_prompt - 1 : n_prompt;
    const int sample = sample_token ? 1 : 0;
    const int zero = 0;
    st = ck(cudaMemcpyAsync(model->d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, model->stream));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpyAsync(model->d_sample, &sample, sizeof(int), cudaMemcpyHostToDevice, model->stream));
    if (st != AK_OK) return st;
    st = ck(cudaMemcpyAsync(model->d_count, &zero, sizeof(int), cudaMemcpyHostToDevice, model->stream));
    if (st != AK_OK) return st;
    if (!sample_token) {
        return AK_OK;
    }
    ak_task fin{};
    fin.op = AK_OP_FINALIZE;
    fin.p0 = reinterpret_cast<uint64_t>(model->logits);
    fin.p1 = reinterpret_cast<uint64_t>(model->d_pos);
    fin.p2 = reinterpret_cast<uint64_t>(model->d_token);
    fin.p3 = reinterpret_cast<uint64_t>(model->d_out);
    fin.p4 = reinterpret_cast<uint64_t>(model->d_sample);
    fin.p5 = reinterpret_cast<uint64_t>(model->d_count);
    fin.n0 = model->cfg.vocab;
    return ak_launch_task(&fin, model->stream);
}

extern "C" ak_status ak_model_logits(ak_model* model, const int32_t* prompt, int n_prompt, float* logits) {
    if (!model || !prompt || n_prompt < 1 || !logits) {
        ak_set_error("logits: bad arguments");
        return AK_ERR_INVALID;
    }
    if (n_prompt > model->cfg.max_seq) {
        ak_set_error("logits: prompt exceeds max_seq");
        return AK_ERR_INVALID;
    }
    ak_status st = reset_token_state(model);
    if (st != AK_OK) {
        return st;
    }
    for (int i = 0; i < n_prompt; ++i) {
        const int sample = 0;
        st = ck(cudaMemcpyAsync(model->d_token, &prompt[i], sizeof(int), cudaMemcpyHostToDevice, model->stream));
        if (st != AK_OK) {
            return st;
        }
        st = ck(cudaMemcpyAsync(model->d_sample, &sample, sizeof(int), cudaMemcpyHostToDevice, model->stream));
        if (st != AK_OK) {
            return st;
        }
        int launches = 0;
        st = run_backend(model, AK_BACKEND_EAGER, &launches);
        if (st != AK_OK) {
            return st;
        }
    }
    st = ck(cudaStreamSynchronize(model->stream));
    if (st != AK_OK) {
        return st;
    }
    return ck(cudaMemcpy(logits, model->logits, sizeof(float) * model->cfg.vocab, cudaMemcpyDeviceToHost));
}

extern "C" ak_status ak_generate(ak_model* model, const int32_t* prompt, int n_prompt, int32_t* out_tokens, int n_gen,
                                 int backend, float* per_token_ms, int per_token_cap, ak_gen_stats* stats) {
    if (!model || !prompt || n_prompt < 1 || n_gen < 1 || !out_tokens) {
        ak_set_error("generate: bad arguments");
        return AK_ERR_INVALID;
    }
    if (n_prompt + n_gen - 1 > model->cfg.max_seq) {
        ak_set_error("generate: prompt plus generation exceeds max_seq");
        return AK_ERR_INVALID;
    }
    if (stats) {
        std::memset(stats, 0, sizeof(*stats));
        stats->backend = backend;
        stats->prompt_tokens = n_prompt;
        stats->n_gpu_layers = model->n_gpu_layers;
        stats->weight_vram_mb = static_cast<float>(model->weight_bytes / (1024.0 * 1024.0));
        stats->host_weight_mb = static_cast<float>(model->host_bytes / (1024.0 * 1024.0));
        stats->vram_peak_mb = static_cast<float>(model->total_alloc / (1024.0 * 1024.0));
    }
    ak_status st = reset_token_state(model);
    if (st != AK_OK) {
        if (stats) stats->status = st;
        return st;
    }
    const auto wall0 = std::chrono::steady_clock::now();
    int launches = 0;
    bool prefilled = false;
    if (model->all_resident) {
        st = run_prefill(model, prompt, n_prompt, true);
        if (st == AK_OK) {
            prefilled = true;
            if (per_token_ms) {
                st = ck(cudaStreamSynchronize(model->stream));
                if (st != AK_OK) {
                    if (stats) stats->status = st;
                    return st;
                }
                if (per_token_cap > 0) {
                    per_token_ms[0] = elapsed_ms(wall0, std::chrono::steady_clock::now());
                }
            }
        } else if (st != AK_ERR_UNSUPPORTED) {
            if (stats) stats->status = st;
            return st;
        }
    }
    if (!prefilled) {
    for (int i = 0; i < n_prompt; ++i) {
        const int sample = (i + 1 == n_prompt) ? 1 : 0;
        const auto t0 = std::chrono::steady_clock::now();
        st = ck(cudaMemcpyAsync(model->d_token, &prompt[i], sizeof(int), cudaMemcpyHostToDevice, model->stream));
        if (st == AK_OK) {
            st = ck(cudaMemcpyAsync(model->d_sample, &sample, sizeof(int), cudaMemcpyHostToDevice, model->stream));
        }
        if (st == AK_OK) {
            st = run_backend(model, backend, &launches);
        }
        if (st == AK_OK && per_token_ms) {
            st = ck(cudaStreamSynchronize(model->stream));
        }
        if (st != AK_OK) {
            if (stats) stats->status = st;
            return st;
        }
        if (i + 1 == n_prompt && per_token_ms && per_token_cap > 0) {
            per_token_ms[0] = elapsed_ms(wall0, std::chrono::steady_clock::now());
        }
        (void)t0;
    }
    }
    if (!per_token_ms) {
        st = ck(cudaStreamSynchronize(model->stream));
        if (st != AK_OK) {
            if (stats) stats->status = st;
            return st;
        }
    }
    const auto after_prefill = std::chrono::steady_clock::now();
    const float ttft = elapsed_ms(wall0, after_prefill);
    for (int g = 1; g < n_gen; ++g) {
        const auto t0 = std::chrono::steady_clock::now();
        st = run_backend(model, backend, &launches);
        if (st != AK_OK) {
            if (stats) stats->status = st;
            return st;
        }
        if (per_token_ms) {
            st = ck(cudaStreamSynchronize(model->stream));
            if (st != AK_OK) {
                if (stats) stats->status = st;
                return st;
            }
            if (g < per_token_cap) {
                per_token_ms[g] = elapsed_ms(t0, std::chrono::steady_clock::now());
            }
        }
    }
    st = ck(cudaStreamSynchronize(model->stream));
    if (st != AK_OK) {
        if (stats) stats->status = st;
        return st;
    }
    const auto wall1 = std::chrono::steady_clock::now();
    int count = 0;
    st = ck(cudaMemcpy(&count, model->d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (st != AK_OK) {
        if (stats) stats->status = st;
        return st;
    }
    if (count != n_gen) {
        ak_set_error("generate: device token count does not match the request");
        if (stats) stats->status = AK_ERR_CUDA;
        return AK_ERR_CUDA;
    }
    st = ck(cudaMemcpy(out_tokens, model->d_out, sizeof(int) * n_gen, cudaMemcpyDeviceToHost));
    if (st != AK_OK) {
        if (stats) stats->status = st;
        return st;
    }
    if (backend == AK_BACKEND_PERSISTENT || backend == AK_BACKEND_HYBRID) {
        int completed = 0;
        int error = 0;
        cudaMemcpy(&completed, model->persist.d_completed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&error, model->persist.d_error, sizeof(int), cudaMemcpyDeviceToHost);
        if (error != 0 || completed != model->persist.n_tasks) {
            ak_set_error(error == AK_ERR_TIMEOUT ? "persistent kernel timed out" : "persistent kernel did not finish");
            if (stats) stats->status = error ? error : AK_ERR_TIMEOUT;
            return error ? static_cast<ak_status>(error) : AK_ERR_TIMEOUT;
        }
    }
    if (stats) {
        stats->prefill_ms = ttft;
        stats->ttft_ms = ttft;
        stats->decode_ms = elapsed_ms(after_prefill, wall1);
        stats->generated_tokens = n_gen;
        stats->launches_per_token = launches;
        stats->graph_nodes = model->graph_nodes;
        stats->graph_capture_ms = model->graph_capture_ms;
        stats->persistent_blocks = model->persist.n_blocks;
        stats->decode_tps = stats->decode_ms > 0.f && n_gen > 1 ? (n_gen - 1) * 1000.f / stats->decode_ms : 0.f;
        stats->prefill_tps = ttft > 0.f ? n_prompt * 1000.f / ttft : 0.f;
        stats->status = AK_OK;
    }
    return AK_OK;
}

int host_argmax(const std::vector<float>& values) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(values.size()); ++i) {
        if (values[static_cast<size_t>(i)] > values[static_cast<size_t>(best)]) {
            best = i;
        }
    }
    return best;
}

ak_status copy_logits(ak_model* model, std::vector<float>& dst) {
    dst.resize(static_cast<size_t>(model->cfg.vocab));
    ak_status st = ck(cudaStreamSynchronize(model->stream));
    if (st != AK_OK) {
        return st;
    }
    return ck(cudaMemcpy(dst.data(), model->logits, sizeof(float) * model->cfg.vocab, cudaMemcpyDeviceToHost));
}

ak_status write_logits(ak_model* model, const std::vector<float>& src) {
    return ck(cudaMemcpyAsync(model->logits, src.data(), sizeof(float) * model->cfg.vocab, cudaMemcpyHostToDevice, model->stream));
}

ak_status draft_step(ak_model* model, int token) {
    const int sample = 0;
    ak_status st = ck(cudaMemcpyAsync(model->d_token, &token, sizeof(int), cudaMemcpyHostToDevice, model->stream));
    if (st != AK_OK) {
        return st;
    }
    st = ck(cudaMemcpyAsync(model->d_sample, &sample, sizeof(int), cudaMemcpyHostToDevice, model->stream));
    if (st != AK_OK) {
        return st;
    }
    return run_backend(model, AK_BACKEND_EAGER, nullptr);
}

extern "C" ak_status ak_speculate(ak_model* target, ak_model* draft, const int32_t* prompt, int n_prompt,
                                  int32_t* out_tokens, int n_gen, int gamma, int32_t* out_accepts, int accept_cap,
                                  ak_gen_stats* stats) {
    if (!target || !draft || !prompt || n_prompt < 1 || !out_tokens || n_gen < 1) {
        ak_set_error("speculate: bad arguments");
        return AK_ERR_INVALID;
    }
    if (target->cfg.vocab != draft->cfg.vocab) {
        ak_set_error("speculate: draft and target vocabularies differ");
        return AK_ERR_INVALID;
    }
    if (gamma < 1) {
        gamma = 1;
    }
    if (gamma > 8) {
        gamma = 8;
    }
    if (n_prompt + n_gen > target->cfg.max_seq || n_prompt + n_gen > draft->cfg.max_seq) {
        ak_set_error("speculate: sequence exceeds max_seq");
        return AK_ERR_INVALID;
    }
    if (!target->all_resident || !draft->all_resident) {
        ak_set_error("speculate: both models must be GPU resident");
        return AK_ERR_UNSUPPORTED;
    }
    if (stats) {
        std::memset(stats, 0, sizeof(*stats));
        stats->backend = AK_BACKEND_EAGER;
        stats->prompt_tokens = n_prompt;
    }
    ak_status st = reset_token_state(target);
    if (st == AK_OK) {
        st = reset_token_state(draft);
    }
    if (st != AK_OK) {
        return st;
    }
    st = run_prefill(target, prompt, n_prompt, false);
    if (st == AK_OK) {
        st = run_prefill(draft, prompt, n_prompt, false);
    }
    if (st != AK_OK) {
        return st;
    }
    const int vocab = target->cfg.vocab;
    std::vector<float> tlog, dlog, cursor, steps(static_cast<size_t>(gamma) * vocab);
    st = copy_logits(target, tlog);
    if (st == AK_OK) {
        st = copy_logits(draft, dlog);
    }
    if (st != AK_OK) {
        return st;
    }
    std::vector<int> fed(prompt, prompt + n_prompt);
    int produced = 0;
    int rounds = 0;
    int accepted_tokens = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    while (produced < n_gen) {
        const int span = std::min(gamma, n_gen - produced);
        std::vector<int> prop(static_cast<size_t>(span));
        for (int i = 0; i < span; ++i) {
            prop[static_cast<size_t>(i)] = host_argmax(dlog);
            st = draft_step(draft, prop[static_cast<size_t>(i)]);
            if (st != AK_OK) {
                return st;
            }
            st = copy_logits(draft, dlog);
            if (st != AK_OK) {
                return st;
            }
        }
        const int pos0 = static_cast<int>(fed.size());
        st = ck(cudaMemcpyAsync(target->d_prompt, prop.data(), sizeof(int) * span, cudaMemcpyHostToDevice, target->stream));
        if (st != AK_OK) {
            return st;
        }
        std::vector<PrefillLayer> layers(static_cast<size_t>(target->cfg.n_layers));
        for (int i = 0; i < target->cfg.n_layers; ++i) {
            const Layer& src = target->layers[i];
            layers[static_cast<size_t>(i)] = PrefillLayer{src.attn_norm, src.wq, src.wk, src.wv, src.wo, src.q_norm, src.k_norm,
                                                          src.ffn_norm, src.wgate, src.wup, src.wdown, src.kc, src.vc, src.kv_view};
        }
        PrefillBuffers buf{target->pref_x, target->pref_y, target->pref_q, target->pref_k, target->pref_v, target->pref_attn,
                           target->pref_gate, target->pref_up, target->pref_ffn, target->pref_proj, target->logits, target->pref_logits};
        st = ak_prefill_extend(target->cublas, target->stream, &target->cfg, layers.data(), target->tok_embd, target->out_norm,
                               target->wout, target->d_prompt, span, pos0, buf, true);
        if (st != AK_OK) {
            return st;
        }
        st = ck(cudaStreamSynchronize(target->stream));
        if (st != AK_OK) {
            return st;
        }
        st = ck(cudaMemcpy(steps.data(), target->pref_logits, sizeof(float) * static_cast<size_t>(span) * vocab, cudaMemcpyDeviceToHost));
        if (st != AK_OK) {
            return st;
        }
        cursor = tlog;
        int fed_now = 0;
        while (fed_now < span && produced + fed_now < n_gen && host_argmax(cursor) == prop[static_cast<size_t>(fed_now)]) {
            out_tokens[produced + fed_now] = prop[static_cast<size_t>(fed_now)];
            cursor.assign(steps.begin() + static_cast<size_t>(fed_now) * vocab, steps.begin() + static_cast<size_t>(fed_now + 1) * vocab);
            ++fed_now;
        }
        produced += fed_now;
        accepted_tokens += fed_now;
        fed.insert(fed.end(), prop.begin(), prop.begin() + fed_now);
        if (produced < n_gen) {
            const int bonus = host_argmax(cursor);
            out_tokens[produced++] = bonus;
            if (produced < n_gen) {
                fed.push_back(bonus);
                const int bonus_pos = static_cast<int>(fed.size()) - 1;
                st = ck(cudaMemcpyAsync(target->d_pos, &bonus_pos, sizeof(int), cudaMemcpyHostToDevice, target->stream));
                if (st != AK_OK) {
                    return st;
                }
                st = draft_step(target, bonus);
                if (st != AK_OK) {
                    return st;
                }
                st = copy_logits(target, tlog);
                if (st != AK_OK) {
                    return st;
                }
                if (fed_now < span) {
                    st = reset_token_state(draft);
                    if (st != AK_OK) {
                        return st;
                    }
                    st = run_prefill(draft, fed.data(), static_cast<int>(fed.size()), false);
                } else {
                    st = draft_step(draft, bonus);
                }
                if (st != AK_OK) {
                    return st;
                }
                st = copy_logits(draft, dlog);
                if (st != AK_OK) {
                    return st;
                }
            } else {
                tlog = cursor;
            }
        } else {
            tlog = cursor;
        }
        if (out_accepts && rounds < accept_cap) {
            out_accepts[rounds] = fed_now;
        }
        ++rounds;
        if (rounds > n_gen + 2) {
            ak_set_error("speculate: too many rounds");
            return AK_ERR_CUDA;
        }
    }
    const auto wall1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->generated_tokens = n_gen;
        stats->decode_ms = elapsed_ms(wall0, wall1);
        stats->decode_tps = stats->decode_ms > 0.f ? n_gen * 1000.f / stats->decode_ms : 0.f;
        stats->launches_per_token = rounds;
        stats->persistent_blocks = accepted_tokens;
        stats->status = AK_OK;
    }
    return AK_OK;
}
