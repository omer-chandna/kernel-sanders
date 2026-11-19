// Qwen3 decoder that runs straight from a llama.cpp GGUF file.
//
// Weights stay in their file format (Q8_0, Q4_K, Q6_K) in one device allocation that mirrors the GGUF data
// section. Every projection is "quantize the input vector to int8 blocks, then dp4a GEMV" (qgemv.cuh).
// Norms, RoPE, attention and the KV cache are FP32 and reuse the device ops of the FP32 engine.
//
// The same task list drives three executors:
//   eager       one launch per task, whole matrices as one task
//   cuda-graph  that launch sequence captured once
//   persistent  matrices split into row chunks, run by the device-side DAG scheduler (megakernel.cu)
//   hybrid      persistent, captured in a CUDA graph
// Prompt tokens are processed one at a time through the decode graph. There is no batched prefill here.

#include "error.h"
#include "launch.h"
#include "ops.cuh"

#include <adakernel/api.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kSpecCols = 9;  // must match kMaxCols in the speculative section

enum { kGgufF32 = 0, kGgufQ8 = 8, kGgufQ4K = 12, kGgufQ6K = 14 };

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
};

struct GgufFile {
    std::map<std::string, double> num;
    std::map<std::string, std::string> str;
    std::map<std::string, GgufTensor> tensors;
    uint64_t data_offset = 0;
    uint64_t data_bytes = 0;
};

struct Reader {
    std::ifstream in;
    bool ok() const { return static_cast<bool>(in); }
    uint64_t u64() {
        uint64_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 8);
        return v;
    }
    uint32_t u32() {
        uint32_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 4);
        return v;
    }
    std::string str() {
        const uint64_t n = u64();
        if (n > (1ull << 30)) {
            in.setstate(std::ios::failbit);
            return {};
        }
        std::string s(static_cast<size_t>(n), '\0');
        if (n) in.read(s.data(), static_cast<std::streamsize>(n));
        return s;
    }
    // Scalar value as double; strings returned through *s; arrays skipped.
    bool value(uint32_t t, double* d, std::string* s) {
        switch (t) {
            case 0: { uint8_t v = 0; in.read(reinterpret_cast<char*>(&v), 1); *d = v; return true; }
            case 1: { int8_t v = 0; in.read(reinterpret_cast<char*>(&v), 1); *d = v; return true; }
            case 2: { uint16_t v = 0; in.read(reinterpret_cast<char*>(&v), 2); *d = v; return true; }
            case 3: { int16_t v = 0; in.read(reinterpret_cast<char*>(&v), 2); *d = v; return true; }
            case 4: { uint32_t v = 0; in.read(reinterpret_cast<char*>(&v), 4); *d = v; return true; }
            case 5: { int32_t v = 0; in.read(reinterpret_cast<char*>(&v), 4); *d = v; return true; }
            case 6: { float v = 0; in.read(reinterpret_cast<char*>(&v), 4); *d = v; return true; }
            case 7: { uint8_t v = 0; in.read(reinterpret_cast<char*>(&v), 1); *d = v; return true; }
            case 8: *s = str(); return true;
            case 9: {
                const uint32_t et = u32();
                const uint64_t n = u64();
                double dd;
                std::string ss;
                for (uint64_t i = 0; i < n && ok(); ++i) value(et, &dd, &ss);
                return false;
            }
            case 10: { uint64_t v = 0; in.read(reinterpret_cast<char*>(&v), 8); *d = static_cast<double>(v); return true; }
            case 11: { int64_t v = 0; in.read(reinterpret_cast<char*>(&v), 8); *d = static_cast<double>(v); return true; }
            case 12: { double v = 0; in.read(reinterpret_cast<char*>(&v), 8); *d = v; return true; }
            default: in.setstate(std::ios::failbit); return false;
        }
    }
};

ak_status parse_gguf(const char* path, GgufFile* f) {
    Reader r;
    r.in.open(path, std::ios::binary);
    if (!r.ok()) {
        ak_set_error("gguf: cannot open file");
        return AK_ERR_INVALID;
    }
    if (r.u32() != 0x46554747u) {
        ak_set_error("gguf: bad magic");
        return AK_ERR_INVALID;
    }
    const uint32_t version = r.u32();
    if (version < 2) {
        ak_set_error("gguf: version 1 is not supported");
        return AK_ERR_UNSUPPORTED;
    }
    const uint64_t n_tensors = r.u64();
    const uint64_t n_kv = r.u64();
    for (uint64_t i = 0; i < n_kv && r.ok(); ++i) {
        const std::string key = r.str();
        const uint32_t t = r.u32();
        double d = 0;
        std::string s;
        if (r.value(t, &d, &s)) {
            if (t == 8) f->str[key] = s;
            else f->num[key] = d;
        }
    }
    for (uint64_t i = 0; i < n_tensors && r.ok(); ++i) {
        GgufTensor t;
        t.name = r.str();
        const uint32_t nd = r.u32();
        if (nd > 4) {
            ak_set_error("gguf: tensor with more than 4 dims");
            return AK_ERR_INVALID;
        }
        t.dims.resize(nd);
        for (uint32_t k = 0; k < nd; ++k) t.dims[k] = r.u64();
        t.type = r.u32();
        t.offset = r.u64();
        f->tensors[t.name] = t;
    }
    if (!r.ok()) {
        ak_set_error("gguf: truncated header");
        return AK_ERR_INVALID;
    }
    uint64_t align = 32;
    auto it = f->num.find("general.alignment");
    if (it != f->num.end() && it->second > 0) align = static_cast<uint64_t>(it->second);
    const uint64_t header_end = static_cast<uint64_t>(r.in.tellg());
    f->data_offset = (header_end + align - 1) / align * align;
    r.in.seekg(0, std::ios::end);
    const uint64_t file_end = static_cast<uint64_t>(r.in.tellg());
    if (file_end < f->data_offset) {
        ak_set_error("gguf: no data section");
        return AK_ERR_INVALID;
    }
    f->data_bytes = file_end - f->data_offset;
    return AK_OK;
}

int qtype_code(uint32_t t) {
    if (t == kGgufQ8) return 0;
    if (t == kGgufQ4K) return 1;
    if (t == kGgufQ6K) return 2;
    return -1;
}

size_t qrow_bytes(uint32_t t, int K) {
    if (t == kGgufQ8) return static_cast<size_t>(K / 32) * 34;
    if (t == kGgufQ4K) return static_cast<size_t>(K / 256) * 144;
    if (t == kGgufQ6K) return static_cast<size_t>(K / 256) * 210;
    return 0;
}

struct QMat {
    const uint8_t* w = nullptr;
    int type = -1;  // 0 Q8_0, 1 Q4_K, 2 Q6_K
    int rows = 0;
    int K = 0;
    size_t row_bytes = 0;
};

struct QLayer {
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    QMat wq, wk, wv, wo, wgate, wup, wdown;
    float* kc = nullptr;
    float* vc = nullptr;
};

struct GSchedule {
    std::vector<ak_task> tasks;
    std::vector<int> succ;
    std::vector<int> ready;
};

ak_status ck(cudaError_t err) {
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return err == cudaErrorMemoryAllocation ? AK_ERR_OOM : AK_ERR_CUDA;
    }
    return AK_OK;
}

}  // namespace

struct ak_gguf_model {
    ak_gguf_info info{};
    int n_layers = 0, d = 0, n_heads = 0, n_kv = 0, hd = 0, ff = 0, vocab = 0, max_seq = 0;
    float eps = 1e-6f, theta = 10000.f;
    int chunk_rows = 64;
    uint8_t* blob = nullptr;
    std::vector<QLayer> layers;
    QMat embd;
    QMat out;
    const float* out_norm = nullptr;
    // activations
    float *x = nullptr, *h = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr, *proj = nullptr;
    float *gate = nullptr, *up = nullptr, *ffn = nullptr, *logits = nullptr, *scores = nullptr;
    int8_t *hq = nullptr, *aq = nullptr, *fq = nullptr;
    float *hd_s = nullptr, *ad_s = nullptr, *fd_s = nullptr;
    unsigned long long* d_best = nullptr;
    unsigned long long* d_vbest = nullptr;  // per verify column
    int *d_vtok = nullptr, *d_vout = nullptr;
    std::unordered_map<int, cudaGraphExec_t> verify_graphs;
    int *d_token = nullptr, *d_pos = nullptr, *d_sample = nullptr, *d_out = nullptr, *d_count = nullptr;
    std::vector<void*> mem;
    size_t total_alloc = 0;
    cudaStream_t stream = nullptr;
    GSchedule eager_sched;  // whole-matrix tasks
    GSchedule chunk_sched;  // row-chunk tasks for the persistent scheduler
    ak_persist_buf persist{};
    int persist_ready = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int graph_kind = -1;
    float graph_capture_ms = 0.f;
};

namespace {

ak_status galloc(ak_gguf_model* m, void** p, size_t bytes) {
    *p = nullptr;
    if (bytes == 0) return AK_OK;
    ak_status st = ck(cudaMalloc(p, bytes));
    if (st != AK_OK) return st;
    m->mem.push_back(*p);
    m->total_alloc += bytes;
    return AK_OK;
}

template <class T>
ak_status galloc_n(ak_gguf_model* m, T** p, size_t n) {
    void* raw = nullptr;
    ak_status st = galloc(m, &raw, n * sizeof(T));
    *p = static_cast<T*>(raw);
    return st;
}

struct Rec {
    std::vector<ak_task> tasks;
    std::vector<std::pair<int, int>> edges;
    int add(ak_task t, const std::vector<int>& preds) {
        t.preds_init = 0;
        const int id = static_cast<int>(tasks.size());
        tasks.push_back(t);
        for (int p : preds) edges.emplace_back(p, id);
        return id;
    }
};

ak_status finalize(const Rec& rec, GSchedule* out) {
    const int n = static_cast<int>(rec.tasks.size());
    out->tasks = rec.tasks;
    std::vector<int> counts(n, 0);
    for (const auto& e : rec.edges) {
        if (e.first < 0 || e.second >= n || e.first >= e.second) {
            ak_set_error("gguf model: invalid task edge");
            return AK_ERR_INVALID;
        }
        counts[e.first] += 1;
        out->tasks[e.second].preds_init += 1;
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
    for (const auto& e : rec.edges) out->succ[out->tasks[e.first].succ_begin + fill[e.first]++] = e.second;
    out->ready.clear();
    for (int i = 0; i < n; ++i) {
        if (out->tasks[i].preds_init == 0) out->ready.push_back(i);
    }
    return AK_OK;
}

// Emits one decode step. chunk <= 0 means whole matrices and vectors as single tasks (eager / graph).
//
// Per layer:  add+norm+quant -> q,k,v gemv -> q norm+rope | k norm+rope+kv write -> attention (writes int8)
//             -> o gemv -> add+norm+quant -> gate,up gemv -> silu+quant -> down gemv -> (next layer's add)
void emit_step(ak_gguf_model* m, Rec& rec, int chunk) {
    const bool chunked = chunk > 0;
    const int n_rep = m->n_heads / m->n_kv;
    const int silu_chunk = 1024;
    const int argmax_chunk = 4096;
    int theta_bits = 0;
    std::memcpy(&theta_bits, &m->theta, sizeof(int));

    auto gemv = [&](const QMat& W, const int8_t* xq, const float* xd, float* y, const std::vector<int>& preds) {
        std::vector<int> ids;
        const int op = W.type == 0 ? AK_OP_Q8_DP4A : (W.type == 1 ? AK_OP_Q4K_DP4A : AK_OP_Q6K_DP4A);
        const int step = chunked ? chunk : W.rows;
        for (int r = 0; r < W.rows; r += step) {
            ak_task t{};
            t.op = op;
            t.p0 = reinterpret_cast<uint64_t>(W.w + static_cast<size_t>(r) * W.row_bytes);
            t.p1 = reinterpret_cast<uint64_t>(xq);
            t.p2 = reinterpret_cast<uint64_t>(y + r);
            t.p3 = reinterpret_cast<uint64_t>(xd);
            t.n0 = std::min(step, W.rows - r);
            t.n1 = W.K;
            ids.push_back(rec.add(t, preds));
        }
        return ids;
    };
    // Same type and K, and the first matrix's rows a multiple of the grid kernel's rows per block (8).
    auto pairable = [&](const QMat& a, const QMat& b) { return a.type == b.type && a.K == b.K && (a.rows % 8) == 0; };
    auto gemv_pair = [&](const QMat& A, const QMat& B, const int8_t* xq, const float* xd, float* y,
                         const std::vector<int>& preds) {
        ak_task t{};
        t.op = A.type == 0 ? AK_OP_Q8_DP4A : (A.type == 1 ? AK_OP_Q4K_DP4A : AK_OP_Q6K_DP4A);
        t.p0 = reinterpret_cast<uint64_t>(A.w);
        t.p1 = reinterpret_cast<uint64_t>(xq);
        t.p2 = reinterpret_cast<uint64_t>(y);
        t.p3 = reinterpret_cast<uint64_t>(xd);
        t.p4 = reinterpret_cast<uint64_t>(B.w);
        t.n0 = A.rows + B.rows;
        t.n1 = A.K;
        t.n2 = A.rows;
        return std::vector<int>{rec.add(t, preds)};
    };
    auto add_norm_quant = [&](const float* res, const float* w, const std::vector<int>& preds) {
        ak_task t{};
        t.op = AK_OP_ADD_NORM_QUANT;
        t.p0 = reinterpret_cast<uint64_t>(m->x);
        t.p1 = reinterpret_cast<uint64_t>(res);
        t.p2 = reinterpret_cast<uint64_t>(w);
        t.p3 = reinterpret_cast<uint64_t>(m->hd_s);
        t.p4 = reinterpret_cast<uint64_t>(m->hq);
        t.n0 = m->d;
        t.f0 = m->eps;
        return rec.add(t, preds);
    };

    ak_task embed{};
    embed.op = AK_OP_EMBED_Q;
    embed.p0 = reinterpret_cast<uint64_t>(m->embd.w);
    embed.p1 = reinterpret_cast<uint64_t>(m->d_token);
    embed.p2 = reinterpret_cast<uint64_t>(m->x);
    embed.n0 = m->d;
    embed.i0 = m->embd.type;
    std::vector<int> prev{rec.add(embed, {})};
    const float* pending = nullptr;  // residual still to be added into x

    for (int l = 0; l < m->n_layers; ++l) {
        const QLayer& ly = m->layers[l];
        const int nq = add_norm_quant(pending, ly.attn_norm, prev);
        std::vector<int> q_ids, k_ids;
        if (!chunked && pairable(ly.wq, ly.wk)) {
            // One launch for Q and K: k sits right after q in memory.
            q_ids = gemv_pair(ly.wq, ly.wk, m->hq, m->hd_s, m->q, {nq});
            k_ids = q_ids;
        } else {
            q_ids = gemv(ly.wq, m->hq, m->hd_s, m->q, {nq});
            k_ids = gemv(ly.wk, m->hq, m->hd_s, m->k, {nq});
        }
        const auto v_ids = gemv(ly.wv, m->hq, m->hd_s, m->v, {nq});

        ak_task qr{};
        qr.op = AK_OP_QK_NORM_ROPE;
        qr.p0 = reinterpret_cast<uint64_t>(m->q);
        qr.p1 = reinterpret_cast<uint64_t>(ly.q_norm);
        qr.p3 = reinterpret_cast<uint64_t>(m->d_pos);
        qr.n0 = m->n_heads;
        qr.n1 = m->hd;
        qr.n2 = m->max_seq;
        qr.f0 = m->eps;
        qr.i0 = theta_bits;
        const int qr_id = rec.add(qr, q_ids);
        ak_task kr = qr;
        kr.p0 = reinterpret_cast<uint64_t>(m->k);
        kr.p1 = reinterpret_cast<uint64_t>(ly.k_norm);
        kr.p2 = reinterpret_cast<uint64_t>(ly.kc);
        kr.p4 = reinterpret_cast<uint64_t>(m->v);
        kr.p5 = reinterpret_cast<uint64_t>(ly.vc);
        kr.n0 = m->n_kv;
        std::vector<int> kr_preds = k_ids;
        kr_preds.insert(kr_preds.end(), v_ids.begin(), v_ids.end());
        const int kr_id = rec.add(kr, kr_preds);

        ak_task a{};
        a.op = AK_OP_ATTN_DECODE;
        a.p1 = reinterpret_cast<uint64_t>(ly.kc);
        a.p2 = reinterpret_cast<uint64_t>(ly.vc);
        a.p4 = reinterpret_cast<uint64_t>(m->d_pos);
        a.n1 = m->hd;
        a.n2 = m->max_seq;
        a.n3 = m->n_kv;
        std::vector<int> heads;
        if (chunked) {
            for (int hh = 0; hh < m->n_heads; ++hh) {
                ak_task t = a;
                t.p0 = reinterpret_cast<uint64_t>(m->q + hh * m->hd);
                t.p3 = reinterpret_cast<uint64_t>(m->aq + hh * m->hd);
                t.p5 = reinterpret_cast<uint64_t>(m->scores + static_cast<size_t>(hh) * m->max_seq);
                t.p6 = reinterpret_cast<uint64_t>(m->ad_s + hh * (m->hd / 32));
                t.n0 = hh / n_rep;
                heads.push_back(rec.add(t, {qr_id, kr_id}));
            }
        } else {
            a.p0 = reinterpret_cast<uint64_t>(m->q);
            a.p3 = reinterpret_cast<uint64_t>(m->aq);
            a.p5 = reinterpret_cast<uint64_t>(m->scores);
            a.p6 = reinterpret_cast<uint64_t>(m->ad_s);
            a.i0 = m->n_heads;
            a.i1 = n_rep;
            heads.push_back(rec.add(a, {qr_id, kr_id}));
        }
        const auto o_ids = gemv(ly.wo, m->aq, m->ad_s, m->proj, heads);
        const int n2 = add_norm_quant(m->proj, ly.ffn_norm, o_ids);
        std::vector<int> gu;
        if (!chunked && pairable(ly.wgate, ly.wup)) {
            gu = gemv_pair(ly.wgate, ly.wup, m->hq, m->hd_s, m->gate, {n2});
        } else {
            gu = gemv(ly.wgate, m->hq, m->hd_s, m->gate, {n2});
            const auto u_ids = gemv(ly.wup, m->hq, m->hd_s, m->up, {n2});
            gu.insert(gu.end(), u_ids.begin(), u_ids.end());
        }
        std::vector<int> silu_ids;
        const int sstep = chunked ? silu_chunk : m->ff;
        for (int i = 0; i < m->ff; i += sstep) {
            ak_task s{};
            s.op = AK_OP_SILU_QUANT;
            s.p0 = reinterpret_cast<uint64_t>(m->gate + i);
            s.p1 = reinterpret_cast<uint64_t>(m->up + i);
            s.p2 = reinterpret_cast<uint64_t>(m->fq + i);
            s.p3 = reinterpret_cast<uint64_t>(m->fd_s + i / 32);
            s.n0 = std::min(sstep, m->ff - i);
            silu_ids.push_back(rec.add(s, gu));
        }
        prev = gemv(ly.wdown, m->fq, m->fd_s, m->proj, silu_ids);
        pending = m->proj;
    }
    const int fnorm = add_norm_quant(pending, m->out_norm, prev);
    const auto logit_ids = gemv(m->out, m->hq, m->hd_s, m->logits, {fnorm});
    std::vector<int> parts;
    const int astep = chunked ? argmax_chunk : m->vocab;
    for (int i = 0; i < m->vocab; i += astep) {
        ak_task t{};
        t.op = AK_OP_ARGMAX_PART;
        t.p0 = reinterpret_cast<uint64_t>(m->logits + i);
        t.p1 = reinterpret_cast<uint64_t>(m->d_best);
        t.n0 = std::min(astep, m->vocab - i);
        t.i0 = i;
        parts.push_back(rec.add(t, logit_ids));
    }
    ak_task fin{};
    fin.op = AK_OP_FINALIZE;
    fin.p0 = reinterpret_cast<uint64_t>(m->logits);
    fin.p1 = reinterpret_cast<uint64_t>(m->d_pos);
    fin.p2 = reinterpret_cast<uint64_t>(m->d_token);
    fin.p3 = reinterpret_cast<uint64_t>(m->d_out);
    fin.p4 = reinterpret_cast<uint64_t>(m->d_sample);
    fin.p5 = reinterpret_cast<uint64_t>(m->d_count);
    fin.p6 = reinterpret_cast<uint64_t>(m->d_best);
    fin.n0 = m->vocab;
    rec.add(fin, parts);
}

ak_status get_mat(ak_gguf_model* m, const GgufFile& f, const std::string& name, int rows, int K, QMat* out) {
    auto it = f.tensors.find(name);
    if (it == f.tensors.end()) {
        ak_set_error(("gguf: missing tensor " + name).c_str());
        return AK_ERR_INVALID;
    }
    const GgufTensor& t = it->second;
    if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != K || static_cast<int>(t.dims[1]) != rows) {
        ak_set_error(("gguf: unexpected shape for " + name).c_str());
        return AK_ERR_INVALID;
    }
    const int code = qtype_code(t.type);
    if (code < 0) {
        ak_set_error(("gguf: unsupported type for " + name + " (need Q8_0, Q4_K or Q6_K)").c_str());
        return AK_ERR_UNSUPPORTED;
    }
    out->w = m->blob + t.offset;
    out->type = code;
    out->rows = rows;
    out->K = K;
    out->row_bytes = qrow_bytes(t.type, K);
    if (code == 1 && (reinterpret_cast<uintptr_t>(out->w) & 15) != 0) {
        ak_set_error(("gguf: Q4_K tensor not 16-byte aligned: " + name).c_str());
        return AK_ERR_UNSUPPORTED;
    }
    return AK_OK;
}

ak_status get_vec(ak_gguf_model* m, const GgufFile& f, const std::string& name, int n, const float** out) {
    auto it = f.tensors.find(name);
    if (it == f.tensors.end()) {
        ak_set_error(("gguf: missing tensor " + name).c_str());
        return AK_ERR_INVALID;
    }
    if (it->second.type != kGgufF32 || it->second.dims.size() != 1 || static_cast<int>(it->second.dims[0]) != n) {
        ak_set_error(("gguf: expected f32 vector for " + name).c_str());
        return AK_ERR_UNSUPPORTED;
    }
    *out = reinterpret_cast<const float*>(m->blob + it->second.offset);
    return AK_OK;
}

ak_status upload_blob(const char* path, const GgufFile& f, uint8_t* dst) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(f.data_offset));
    const size_t chunk = 64ull << 20;
    void* pinned = nullptr;
    ak_status st = ck(cudaHostAlloc(&pinned, chunk, cudaHostAllocDefault));
    if (st != AK_OK) return st;
    size_t done = 0;
    while (done < f.data_bytes && st == AK_OK) {
        const size_t n = std::min<size_t>(chunk, f.data_bytes - done);
        in.read(static_cast<char*>(pinned), static_cast<std::streamsize>(n));
        if (!in) {
            ak_set_error("gguf: short read of tensor data");
            st = AK_ERR_INVALID;
            break;
        }
        st = ck(cudaMemcpy(dst + done, pinned, n, cudaMemcpyHostToDevice));
        done += n;
    }
    cudaFreeHost(pinned);
    return st;
}

ak_status ensure_persistent(ak_gguf_model* m) {
    if (m->persist_ready) return AK_OK;
    const GSchedule& s = m->chunk_sched;
    ak_status st = ak_persist_alloc(&m->persist, static_cast<int>(s.tasks.size()), static_cast<int>(s.succ.size()),
                                    static_cast<int>(s.ready.size()), 0);
    if (st != AK_OK) return st;
    st = ak_persist_upload(&m->persist, s.tasks.data(), s.succ.data(), s.ready.data());
    if (st != AK_OK) return st;
    m->persist_ready = 1;
    return AK_OK;
}

ak_status launch_eager(ak_gguf_model* m, int* launches) {
    for (const ak_task& t : m->eager_sched.tasks) {
        ak_status st = ak_launch_task(&t, m->stream);
        if (st != AK_OK) return st;
    }
    if (launches) *launches = static_cast<int>(m->eager_sched.tasks.size());
    return AK_OK;
}

ak_status capture(ak_gguf_model* m, int kind) {
    if (m->graph_exec && m->graph_kind == kind) return AK_OK;
    if (m->graph_exec) cudaGraphExecDestroy(m->graph_exec);
    if (m->graph) cudaGraphDestroy(m->graph);
    m->graph_exec = nullptr;
    m->graph = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    ak_status st = ck(cudaStreamSynchronize(m->stream));
    if (st != AK_OK) return st;
    st = ck(cudaStreamBeginCapture(m->stream, cudaStreamCaptureModeGlobal));
    if (st != AK_OK) return st;
    st = kind == AK_BACKEND_HYBRID ? ak_persist_launch(&m->persist, 1000000000, m->stream) : launch_eager(m, nullptr);
    cudaError_t end = cudaStreamEndCapture(m->stream, &m->graph);
    if (st != AK_OK) {
        cudaGetLastError();
        return st;
    }
    if (end != cudaSuccess) return ck(end);
    st = ck(cudaGraphInstantiate(&m->graph_exec, m->graph, 0));
    if (st != AK_OK) return st;
    m->graph_kind = kind;
    m->graph_capture_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return AK_OK;
}

ak_status step(ak_gguf_model* m, int backend, int* launches) {
    if (backend == AK_BACKEND_PERSISTENT || backend == AK_BACKEND_HYBRID) {
        ak_status st = ensure_persistent(m);
        if (st != AK_OK) return st;
    }
    if (backend == AK_BACKEND_CUDA_GRAPH || backend == AK_BACKEND_HYBRID) {
        ak_status st = capture(m, backend);
        if (st != AK_OK) return st;
        if (launches) *launches = 1;
        return ck(cudaGraphLaunch(m->graph_exec, m->stream));
    }
    if (backend == AK_BACKEND_PERSISTENT) {
        if (launches) *launches = 2;
        return ak_persist_launch(&m->persist, 1000000000, m->stream);
    }
    if (backend == AK_BACKEND_EAGER) return launch_eager(m, launches);
    ak_set_error("gguf model: backend must be eager, cuda-graph, persistent or hybrid");
    return AK_ERR_UNSUPPORTED;
}

ak_status reset_state(ak_gguf_model* m) {
    const size_t kv = static_cast<size_t>(m->max_seq) * m->n_kv * m->hd * sizeof(float);
    for (QLayer& ly : m->layers) {
        ak_status st = ck(cudaMemsetAsync(ly.kc, 0, kv, m->stream));
        if (st == AK_OK) st = ck(cudaMemsetAsync(ly.vc, 0, kv, m->stream));
        if (st != AK_OK) return st;
    }
    ak_status st = ck(cudaMemsetAsync(m->d_pos, 0, sizeof(int), m->stream));
    if (st == AK_OK) st = ck(cudaMemsetAsync(m->d_best, 0, sizeof(unsigned long long), m->stream));
    if (st == AK_OK) st = ck(cudaMemsetAsync(m->d_count, 0, sizeof(int), m->stream));
    return st;
}

ak_status check_persist(ak_gguf_model* m, int backend) {
    if (backend != AK_BACKEND_PERSISTENT && backend != AK_BACKEND_HYBRID) return AK_OK;
    int completed = 0, error = 0;
    cudaMemcpy(&completed, m->persist.d_completed, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&error, m->persist.d_error, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != 0 || completed != m->persist.n_tasks) {
        ak_set_error(error == AK_ERR_TIMEOUT ? "persistent kernel timed out" : "persistent kernel did not finish");
        return error ? static_cast<ak_status>(error) : AK_ERR_TIMEOUT;
    }
    return AK_OK;
}

// Feeds the prompt one token at a time. The last prompt token samples (writes out[0]).
ak_status run_prompt(ak_gguf_model* m, const int32_t* prompt, int n_prompt, int backend, int* launches) {
    for (int i = 0; i < n_prompt; ++i) {
        const int sample = (i + 1 == n_prompt) ? 1 : 0;
        ak_status st = ck(cudaMemcpyAsync(m->d_token, &prompt[i], sizeof(int), cudaMemcpyHostToDevice, m->stream));
        if (st == AK_OK) st = ck(cudaMemcpyAsync(m->d_sample, &sample, sizeof(int), cudaMemcpyHostToDevice, m->stream));
        if (st == AK_OK) st = step(m, backend, launches);
        // The host values above are on this stack frame; finish the copies before they go out of scope.
        if (st == AK_OK) st = ck(cudaStreamSynchronize(m->stream));
        if (st != AK_OK) return st;
    }
    return AK_OK;
}

}  // namespace

extern "C" ak_status ak_gguf_destroy(ak_gguf_model* m) {
    if (!m) return AK_OK;
    if (m->stream) cudaStreamSynchronize(m->stream);
    if (m->graph_exec) cudaGraphExecDestroy(m->graph_exec);
    if (m->graph) cudaGraphDestroy(m->graph);
    for (auto& kv : m->verify_graphs) cudaGraphExecDestroy(kv.second);
    if (m->persist_ready) ak_persist_free(&m->persist);
    for (void* p : m->mem) cudaFree(p);
    if (m->stream) cudaStreamDestroy(m->stream);
    delete m;
    return AK_OK;
}

extern "C" ak_status ak_gguf_load(const char* path, int max_seq, int chunk_rows, ak_gguf_model** out,
                                  ak_gguf_info* info) {
    if (!path || !out) {
        ak_set_error("gguf load: bad arguments");
        return AK_ERR_INVALID;
    }
    *out = nullptr;
    ak_status st = ak_init(0);
    if (st != AK_OK) return st;
    GgufFile f;
    st = parse_gguf(path, &f);
    if (st != AK_OK) return st;
    auto it = f.str.find("general.architecture");
    const std::string arch = it == f.str.end() ? "" : it->second;
    if (arch != "qwen3") {
        ak_set_error(("gguf: architecture '" + arch + "' is not qwen3").c_str());
        return AK_ERR_UNSUPPORTED;
    }
    auto num = [&](const char* key, double fallback) {
        auto n = f.num.find(arch + "." + key);
        return n == f.num.end() ? fallback : n->second;
    };
    ak_gguf_model* m = new ak_gguf_model();
    m->n_layers = static_cast<int>(num("block_count", 0));
    m->d = static_cast<int>(num("embedding_length", 0));
    m->ff = static_cast<int>(num("feed_forward_length", 0));
    m->n_heads = static_cast<int>(num("attention.head_count", 0));
    m->n_kv = static_cast<int>(num("attention.head_count_kv", m->n_heads));
    m->hd = static_cast<int>(num("attention.key_length", m->n_heads ? m->d / m->n_heads : 0));
    m->eps = static_cast<float>(num("attention.layer_norm_rms_epsilon", 1e-6));
    m->theta = static_cast<float>(num("rope.freq_base", 10000.0));
    m->max_seq = max_seq > 0 ? max_seq : 1024;
    m->chunk_rows = chunk_rows > 0 ? chunk_rows : 128;
    auto embd_it = f.tensors.find("token_embd.weight");
    if (embd_it == f.tensors.end() || embd_it->second.dims.size() != 2) {
        delete m;
        ak_set_error("gguf: missing token_embd.weight");
        return AK_ERR_INVALID;
    }
    m->vocab = static_cast<int>(embd_it->second.dims[1]);
    if (m->n_layers <= 0 || m->d <= 0 || m->n_heads <= 0 || m->n_kv <= 0 || m->hd <= 0 || m->ff <= 0 ||
        (m->n_heads % m->n_kv) != 0 || (m->d % 256) != 0 || (m->ff % 256) != 0 || ((m->n_heads * m->hd) % 256) != 0) {
        delete m;
        ak_set_error("gguf: unsupported qwen3 shape");
        return AK_ERR_UNSUPPORTED;
    }

    st = ck(cudaStreamCreateWithFlags(&m->stream, cudaStreamNonBlocking));
    void* blob = nullptr;
    if (st == AK_OK) st = galloc(m, &blob, f.data_bytes);
    m->blob = static_cast<uint8_t*>(blob);
    if (st == AK_OK) st = upload_blob(path, f, m->blob);

    const int q_rows = m->n_heads * m->hd;
    const int kv_rows = m->n_kv * m->hd;
    if (st == AK_OK) st = get_mat(m, f, "token_embd.weight", m->vocab, m->d, &m->embd);
    if (st == AK_OK) st = get_vec(m, f, "output_norm.weight", m->d, &m->out_norm);
    if (st == AK_OK) {
        if (f.tensors.count("output.weight")) {
            st = get_mat(m, f, "output.weight", m->vocab, m->d, &m->out);
        } else {
            m->out = m->embd;
            m->info.tied = 1;
        }
    }
    m->layers.resize(m->n_layers);
    for (int l = 0; l < m->n_layers && st == AK_OK; ++l) {
        QLayer& ly = m->layers[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        if (st == AK_OK) st = get_vec(m, f, p + "attn_norm.weight", m->d, &ly.attn_norm);
        if (st == AK_OK) st = get_vec(m, f, p + "ffn_norm.weight", m->d, &ly.ffn_norm);
        if (st == AK_OK) st = get_vec(m, f, p + "attn_q_norm.weight", m->hd, &ly.q_norm);
        if (st == AK_OK) st = get_vec(m, f, p + "attn_k_norm.weight", m->hd, &ly.k_norm);
        if (st == AK_OK) st = get_mat(m, f, p + "attn_q.weight", q_rows, m->d, &ly.wq);
        if (st == AK_OK) st = get_mat(m, f, p + "attn_k.weight", kv_rows, m->d, &ly.wk);
        if (st == AK_OK) st = get_mat(m, f, p + "attn_v.weight", kv_rows, m->d, &ly.wv);
        if (st == AK_OK) st = get_mat(m, f, p + "attn_output.weight", m->d, q_rows, &ly.wo);
        if (st == AK_OK) st = get_mat(m, f, p + "ffn_gate.weight", m->ff, m->d, &ly.wgate);
        if (st == AK_OK) st = get_mat(m, f, p + "ffn_up.weight", m->ff, m->d, &ly.wup);
        if (st == AK_OK) st = get_mat(m, f, p + "ffn_down.weight", m->d, m->ff, &ly.wdown);
        const size_t kv = static_cast<size_t>(m->max_seq) * kv_rows;
        if (st == AK_OK) st = galloc_n(m, &ly.kc, kv);
        if (st == AK_OK) st = galloc_n(m, &ly.vc, kv);
    }
    const int kmax = std::max(m->d, std::max(m->ff, q_rows));
    // Activations hold kSpecCols columns for the speculative verify pass; one-token decode uses column 0.
    const size_t C = kSpecCols;
    if (st == AK_OK) st = galloc_n(m, &m->x, C * m->d);
    if (st == AK_OK) st = galloc_n(m, &m->h, m->d);
    // q|k and gate|up are contiguous so a paired GEMV writes both with one launch.
    if (st == AK_OK) st = galloc_n(m, &m->q, C * (static_cast<size_t>(q_rows) + kv_rows));
    if (st == AK_OK) m->k = m->q + q_rows;
    if (st == AK_OK) st = galloc_n(m, &m->v, C * kv_rows);
    if (st == AK_OK) st = galloc_n(m, &m->attn, q_rows);
    if (st == AK_OK) st = galloc_n(m, &m->proj, C * m->d);
    if (st == AK_OK) st = galloc_n(m, &m->gate, C * 2 * static_cast<size_t>(m->ff));
    if (st == AK_OK) m->up = m->gate + m->ff;
    if (st == AK_OK) st = galloc_n(m, &m->ffn, m->ff);
    if (st == AK_OK) st = galloc_n(m, &m->logits, C * m->vocab);
    if (st == AK_OK) st = galloc_n(m, &m->scores, C * m->n_heads * m->max_seq);
    if (st == AK_OK) st = galloc_n(m, &m->hq, C * kmax);
    if (st == AK_OK) st = galloc_n(m, &m->aq, C * kmax);
    if (st == AK_OK) st = galloc_n(m, &m->fq, C * kmax);
    if (st == AK_OK) st = galloc_n(m, &m->hd_s, C * kmax / 32);
    if (st == AK_OK) st = galloc_n(m, &m->ad_s, C * kmax / 32);
    if (st == AK_OK) st = galloc_n(m, &m->fd_s, C * kmax / 32);
    if (st == AK_OK) st = galloc_n(m, &m->d_best, 1);
    if (st == AK_OK) st = galloc_n(m, &m->d_vbest, C);
    if (st == AK_OK) st = galloc_n(m, &m->d_vtok, C);
    if (st == AK_OK) st = galloc_n(m, &m->d_vout, C);
    if (st == AK_OK) st = ck(cudaMemset(m->d_vbest, 0, sizeof(unsigned long long) * C));
    if (st == AK_OK) st = galloc_n(m, &m->d_token, 1);
    if (st == AK_OK) st = galloc_n(m, &m->d_pos, 1);
    if (st == AK_OK) st = galloc_n(m, &m->d_sample, 1);
    if (st == AK_OK) st = galloc_n(m, &m->d_count, 1);
    if (st == AK_OK) st = galloc_n(m, &m->d_out, static_cast<size_t>(m->max_seq) + 1);
    if (st != AK_OK) {
        ak_gguf_destroy(m);
        return st;
    }
    Rec eager, chunked;
    emit_step(m, eager, 0);
    emit_step(m, chunked, m->chunk_rows);
    st = finalize(eager, &m->eager_sched);
    if (st == AK_OK) st = finalize(chunked, &m->chunk_sched);
    if (st != AK_OK) {
        ak_gguf_destroy(m);
        return st;
    }

    size_t per_token = m->out.row_bytes * m->out.rows;
    for (const QLayer& ly : m->layers) {
        for (const QMat* q : {&ly.wq, &ly.wk, &ly.wv, &ly.wo, &ly.wgate, &ly.wup, &ly.wdown}) {
            per_token += q->row_bytes * q->rows;
        }
    }
    ak_gguf_info& i = m->info;
    std::snprintf(i.arch, sizeof(i.arch), "%s", arch.c_str());
    i.n_layers = m->n_layers;
    i.d_model = m->d;
    i.n_heads = m->n_heads;
    i.n_kv_heads = m->n_kv;
    i.head_dim = m->hd;
    i.d_ff = m->ff;
    i.vocab = m->vocab;
    i.max_seq = m->max_seq;
    i.rope_theta = m->theta;
    i.rms_eps = m->eps;
    i.weight_gb_per_token = static_cast<float>(per_token / 1.0e9);
    i.vram_mb = static_cast<float>(m->total_alloc / (1024.0 * 1024.0));
    i.eager_tasks = static_cast<int>(m->eager_sched.tasks.size());
    i.persistent_tasks = static_cast<int>(m->chunk_sched.tasks.size());
    if (info) *info = i;
    *out = m;
    return AK_OK;
}

extern "C" ak_status ak_gguf_generate(ak_gguf_model* m, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                                      int n_gen, int backend, ak_gen_stats* stats) {
    if (!m || !prompt || n_prompt < 1 || n_gen < 1 || !out_tokens) {
        ak_set_error("gguf generate: bad arguments");
        return AK_ERR_INVALID;
    }
    if (n_prompt + n_gen - 1 > m->max_seq) {
        ak_set_error("gguf generate: prompt plus generation exceeds max_seq");
        return AK_ERR_INVALID;
    }
    for (int i = 0; i < n_prompt; ++i) {
        if (prompt[i] < 0 || prompt[i] >= m->vocab) {
            ak_set_error("gguf generate: prompt token out of range");
            return AK_ERR_INVALID;
        }
    }
    if (stats) std::memset(stats, 0, sizeof(*stats));
    ak_status st = reset_state(m);
    int launches = 0;
    const auto t0 = std::chrono::steady_clock::now();
    if (st == AK_OK) st = run_prompt(m, prompt, n_prompt, backend, &launches);
    if (st == AK_OK) st = check_persist(m, backend);
    const auto t1 = std::chrono::steady_clock::now();
    for (int g = 1; g < n_gen && st == AK_OK; ++g) st = step(m, backend, &launches);
    if (st == AK_OK) st = ck(cudaStreamSynchronize(m->stream));
    const auto t2 = std::chrono::steady_clock::now();
    if (st == AK_OK) st = check_persist(m, backend);
    int count = 0;
    if (st == AK_OK) st = ck(cudaMemcpy(&count, m->d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (st == AK_OK && count != n_gen) {
        ak_set_error("gguf generate: device token count does not match the request");
        st = AK_ERR_CUDA;
    }
    if (st == AK_OK) st = ck(cudaMemcpy(out_tokens, m->d_out, sizeof(int) * n_gen, cudaMemcpyDeviceToHost));
    if (stats) {
        const float pm = std::chrono::duration<float, std::milli>(t1 - t0).count();
        const float dm = std::chrono::duration<float, std::milli>(t2 - t1).count();
        stats->status = st;
        stats->backend = backend;
        stats->prompt_tokens = n_prompt;
        stats->generated_tokens = n_gen;
        stats->prefill_ms = pm;
        stats->ttft_ms = pm;
        stats->decode_ms = dm;
        stats->prefill_tps = pm > 0 ? n_prompt * 1000.f / pm : 0.f;
        stats->decode_tps = dm > 0 && n_gen > 1 ? (n_gen - 1) * 1000.f / dm : 0.f;
        stats->launches_per_token = launches;
        stats->graph_capture_ms = m->graph_capture_ms;
        stats->persistent_blocks = m->persist.n_blocks;
        stats->n_gpu_layers = m->n_layers;
        stats->vram_peak_mb = static_cast<float>(m->total_alloc / (1024.0 * 1024.0));
        stats->weight_vram_mb = static_cast<float>(m->info.weight_gb_per_token * 1.0e9 / (1024.0 * 1024.0));
    }
    return st;
}

// GPU time per task op over one decode step, eager schedule, an event pair around each launch.
// ms_by_op must hold 32 floats (indexed by AK_OP_*); count_by_op 32 ints. Assumes a prior generate/logits call
// left a valid position; it advances that position by one.
extern "C" ak_status ak_gguf_profile(ak_gguf_model* m, float* ms_by_op, int32_t* count_by_op) {
    if (!m || !ms_by_op || !count_by_op) {
        ak_set_error("gguf profile: bad arguments");
        return AK_ERR_INVALID;
    }
    for (int i = 0; i < 32; ++i) {
        ms_by_op[i] = 0.f;
        count_by_op[i] = 0;
    }
    const auto& tasks = m->eager_sched.tasks;
    std::vector<cudaEvent_t> ev(tasks.size() + 1);
    for (auto& e : ev) cudaEventCreate(&e);
    ak_status st = ck(cudaEventRecord(ev[0], m->stream));
    for (size_t i = 0; i < tasks.size() && st == AK_OK; ++i) {
        st = ak_launch_task(&tasks[i], m->stream);
        if (st == AK_OK) st = ck(cudaEventRecord(ev[i + 1], m->stream));
    }
    if (st == AK_OK) st = ck(cudaStreamSynchronize(m->stream));
    for (size_t i = 0; i < tasks.size() && st == AK_OK; ++i) {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
        const int op = tasks[i].op & 31;
        ms_by_op[op] += ms;
        count_by_op[op] += 1;
    }
    for (auto& e : ev) cudaEventDestroy(e);
    return st;
}

extern "C" ak_status ak_gguf_logits(ak_gguf_model* m, const int32_t* prompt, int n_prompt, int backend, float* logits) {
    if (!m || !prompt || n_prompt < 1 || !logits || n_prompt > m->max_seq) {
        ak_set_error("gguf logits: bad arguments");
        return AK_ERR_INVALID;
    }
    ak_status st = reset_state(m);
    if (st == AK_OK) st = run_prompt(m, prompt, n_prompt, backend, nullptr);
    if (st == AK_OK) st = check_persist(m, backend);
    if (st == AK_OK) st = ck(cudaMemcpy(logits, m->logits, sizeof(float) * m->vocab, cudaMemcpyDeviceToHost));
    return st;
}

// =====================================================================================================
// Speculative decoding.
//
// verify: T tokens at positions pos .. pos + T - 1 (pos = *d_pos, not advanced) go through every layer together.
// Each weight matrix is read once for all T tokens (qgemv_dp4a_rows_mc). Token c attends to positions
// <= pos + c, so the pass is causal, and it writes the KV cache for all T positions. Output: greedy argmax for
// each of the T positions in d_vout.
//
// Per column the math is the single-token path's (same kernels for norms, RoPE, attention; the multi-column
// GEMV keeps each column's summation order), so verify's argmax for a column equals what one-token decode
// would have produced at that position. Speculative greedy output is therefore identical to plain greedy.
// =====================================================================================================

namespace {

constexpr int kMaxCols = 9;  // verify width: up to 8 drafted tokens plus the current one

__global__ void spec_embed_cols(const uint8_t* table, int type, const int* tokens, float* x, int d) {
    akd::embed_q_row(table, type, tokens + blockIdx.x, x + static_cast<size_t>(blockIdx.x) * d, d);
}

__global__ void spec_add_norm_quant_cols(float* x, const float* res, const float* w, int8_t* xq, float* xd, int d,
                                         float eps) {
    const size_t c = blockIdx.x;
    akd::add_rmsnorm_quant(x + c * d, res ? res + c * d : nullptr, w, xq + c * d, xd + c * (d / 32), d, eps);
}

// grid (ceil(n_heads / 8), T). Column c sits at position *pos + c.
__global__ void spec_qk_norm_rope_cols(float* q, int qstride, const float* w, int n_heads, int hd, float eps,
                                       const int* pos, float theta, float* kc, const float* v, int vstride, float* vc,
                                       int max_seq) {
    const int c = blockIdx.y;
    const int h0 = blockIdx.x * (blockDim.x >> 5);
    if (h0 >= n_heads) return;
    const int nh = min(n_heads - h0, static_cast<int>(blockDim.x >> 5));
    akd::qk_norm_rope_range(q + static_cast<size_t>(c) * qstride, w, h0, nh, n_heads, hd, eps, pos, theta, kc,
                            v ? v + static_cast<size_t>(c) * vstride : nullptr, vc, max_seq, c);
}

// grid (n_heads, T)
__global__ void spec_attn_cols(const float* q, int qstride, const float* kc, const float* vc, int8_t* aq, float* ad,
                               float* scores, int n_rep, int hd, int max_seq, int n_kv, const int* pos) {
    const int head = blockIdx.x;
    const int c = blockIdx.y;
    const int n_heads = gridDim.x;
    int seq = *pos + c + 1;
    if (seq < 1) seq = 1;
    if (seq > max_seq) seq = max_seq;
    const size_t arow = static_cast<size_t>(c) * n_heads * hd;
    akd::attn_decode_head(q + static_cast<size_t>(c) * qstride + head * hd, kc, vc, aq + arow + head * hd,
                          ad + arow / 32 + head * (hd / 32),
                          scores + (static_cast<size_t>(c) * n_heads + head) * max_seq, head / n_rep, hd, seq, n_kv);
}

// grid (ceil(ff / 256), T); gate|up rows are [gate(ff) | up(ff)] per column.
__global__ void spec_silu_quant_cols(const float* gu, int ff, int8_t* fq, float* fd) {
    const int c = blockIdx.y;
    const int lo = blockIdx.x * 256;
    if (lo >= ff) return;
    const float* g = gu + static_cast<size_t>(c) * 2 * ff;
    const size_t o = static_cast<size_t>(c) * ff + lo;
    akd::silu_quant(g + lo, g + ff + lo, fq + o, fd + o / 32, min(256, ff - lo));
}

// grid (ceil(vocab / 4096), T)
__global__ void spec_argmax_cols(const float* logits, int vocab, unsigned long long* best) {
    const int c = blockIdx.y;
    const int lo = blockIdx.x * 4096;
    if (lo >= vocab) return;
    akd::argmax_part(logits + static_cast<size_t>(c) * vocab + lo, min(4096, vocab - lo), lo, best + c);
}

__global__ void spec_finish_cols(unsigned long long* best, int T, int* out) {
    const int c = threadIdx.x;
    if (c < T) {
        out[c] = akd::argmax_key_index(best[c]);
        best[c] = 0ull;
    }
}

template <int type, int NC>
__global__ void __launch_bounds__(128, 8) spec_gemv_mc(const uint8_t* W, const uint8_t* W2, int rows1, const int8_t* xq, const float* xd,
                             float* y, int ystride, int rows, int K) {
    constexpr int bb = type == 0 ? 34 : (type == 1 ? 144 : 210);
    const int nb = type == 0 ? (K >> 5) : (K >> 8);
    const int per_block = (blockDim.x >> 5) * akd::qgemv_rows_per_warp(type);
    const int row0 = blockIdx.x * per_block;
    if (row0 >= rows) return;
    const int nrows = min(rows - row0, per_block);
    const uint8_t* base = W;
    int wrow = row0;
    if (W2 && row0 >= rows1) {
        base = W2;
        wrow = row0 - rows1;
    }
    akd::qgemv_dp4a_rows_mc<type, NC>(base + static_cast<size_t>(wrow) * nb * bb, xq, xd, y + row0, ystride, nrows, K);
}

template <int type>
void launch_gemv_mc_t(int nc, const uint8_t* W, const uint8_t* W2, int rows1, const int8_t* xq, const float* xd,
                      float* y, int ystride, int rows, int K, cudaStream_t s) {
    const int threads = 128;
    const int per_block = (threads >> 5) * akd::qgemv_rows_per_warp(type);
    const int blocks = (rows + per_block - 1) / per_block;
#define AK_MC(N) \
    case N: spec_gemv_mc<type, N><<<blocks, threads, 0, s>>>(W, W2, rows1, xq, xd, y, ystride, rows, K); break;
    switch (nc) {
        AK_MC(1) AK_MC(2) AK_MC(3) AK_MC(4) AK_MC(5) AK_MC(6) AK_MC(7) AK_MC(8) AK_MC(9)
        default: break;
    }
#undef AK_MC
}

// y[c * ystride + r] = (A | B) row r . x_c ; B optional (same type and K), rows of B follow A's.
void launch_gemv_mc(int nc, const QMat& A, const QMat* B, const int8_t* xq, const float* xd, float* y, int ystride,
                    cudaStream_t s) {
    const int rows = A.rows + (B ? B->rows : 0);
    const uint8_t* W2 = B ? B->w : nullptr;
    if (A.type == 0) launch_gemv_mc_t<0>(nc, A.w, W2, A.rows, xq, xd, y, ystride, rows, A.K, s);
    else if (A.type == 1) launch_gemv_mc_t<1>(nc, A.w, W2, A.rows, xq, xd, y, ystride, rows, A.K, s);
    else launch_gemv_mc_t<2>(nc, A.w, W2, A.rows, xq, xd, y, ystride, rows, A.K, s);
}

bool pair_ok(const QMat& a, const QMat& b) { return a.type == b.type && a.K == b.K && (a.rows % 8) == 0; }

ak_status launch_verify(ak_gguf_model* m, int T, cudaStream_t s) {
    const int d = m->d;
    const int q_rows = m->n_heads * m->hd;
    const int kv_rows = m->n_kv * m->hd;
    const int qk_stride = q_rows + kv_rows;
    const int n_rep = m->n_heads / m->n_kv;
    float theta = m->theta;
    spec_embed_cols<<<T, akd::kThreads, 0, s>>>(m->embd.w, m->embd.type, m->d_vtok, m->x, d);
    const float* pending = nullptr;
    for (int l = 0; l < m->n_layers; ++l) {
        const QLayer& ly = m->layers[l];
        spec_add_norm_quant_cols<<<T, akd::kThreads, 0, s>>>(m->x, pending, ly.attn_norm, m->hq, m->hd_s, d, m->eps);
        if (pair_ok(ly.wq, ly.wk)) {
            launch_gemv_mc(T, ly.wq, &ly.wk, m->hq, m->hd_s, m->q, qk_stride, s);
        } else {
            launch_gemv_mc(T, ly.wq, nullptr, m->hq, m->hd_s, m->q, qk_stride, s);
            launch_gemv_mc(T, ly.wk, nullptr, m->hq, m->hd_s, m->q + q_rows, qk_stride, s);
        }
        launch_gemv_mc(T, ly.wv, nullptr, m->hq, m->hd_s, m->v, kv_rows, s);
        spec_qk_norm_rope_cols<<<dim3((m->n_heads + 7) / 8, T), akd::kThreads, 0, s>>>(
            m->q, qk_stride, ly.q_norm, m->n_heads, m->hd, m->eps, m->d_pos, theta, nullptr, nullptr, 0, nullptr,
            m->max_seq);
        spec_qk_norm_rope_cols<<<dim3((m->n_kv + 7) / 8, T), akd::kThreads, 0, s>>>(
            m->q + q_rows, qk_stride, ly.k_norm, m->n_kv, m->hd, m->eps, m->d_pos, theta, ly.kc, m->v, kv_rows, ly.vc,
            m->max_seq);
        spec_attn_cols<<<dim3(m->n_heads, T), akd::kThreads, 0, s>>>(m->q, qk_stride, ly.kc, ly.vc, m->aq, m->ad_s,
                                                                     m->scores, n_rep, m->hd, m->max_seq, m->n_kv,
                                                                     m->d_pos);
        launch_gemv_mc(T, ly.wo, nullptr, m->aq, m->ad_s, m->proj, d, s);
        spec_add_norm_quant_cols<<<T, akd::kThreads, 0, s>>>(m->x, m->proj, ly.ffn_norm, m->hq, m->hd_s, d, m->eps);
        if (pair_ok(ly.wgate, ly.wup)) {
            launch_gemv_mc(T, ly.wgate, &ly.wup, m->hq, m->hd_s, m->gate, 2 * m->ff, s);
        } else {
            launch_gemv_mc(T, ly.wgate, nullptr, m->hq, m->hd_s, m->gate, 2 * m->ff, s);
            launch_gemv_mc(T, ly.wup, nullptr, m->hq, m->hd_s, m->gate + m->ff, 2 * m->ff, s);
        }
        spec_silu_quant_cols<<<dim3((m->ff + 255) / 256, T), akd::kThreads, 0, s>>>(m->gate, m->ff, m->fq, m->fd_s);
        launch_gemv_mc(T, ly.wdown, nullptr, m->fq, m->fd_s, m->proj, d, s);
        pending = m->proj;
    }
    spec_add_norm_quant_cols<<<T, akd::kThreads, 0, s>>>(m->x, pending, m->out_norm, m->hq, m->hd_s, d, m->eps);
    launch_gemv_mc(T, m->out, nullptr, m->hq, m->hd_s, m->logits, m->vocab, s);
    spec_argmax_cols<<<dim3((m->vocab + 4095) / 4096, T), akd::kThreads, 0, s>>>(m->logits, m->vocab, m->d_vbest);
    spec_finish_cols<<<1, 32, 0, s>>>(m->d_vbest, T, m->d_vout);
    return ck(cudaGetLastError());
}

ak_status verify_graph(ak_gguf_model* m, int T, cudaGraphExec_t* out) {
    auto it = m->verify_graphs.find(T);
    if (it != m->verify_graphs.end()) {
        *out = it->second;
        return AK_OK;
    }
    cudaGraph_t g = nullptr;
    ak_status st = ck(cudaStreamSynchronize(m->stream));
    if (st == AK_OK) st = ck(cudaStreamBeginCapture(m->stream, cudaStreamCaptureModeGlobal));
    if (st != AK_OK) return st;
    st = launch_verify(m, T, m->stream);
    cudaError_t end = cudaStreamEndCapture(m->stream, &g);
    if (st != AK_OK) {
        cudaGetLastError();
        if (g) cudaGraphDestroy(g);
        return st;
    }
    if (end != cudaSuccess) return ck(end);
    cudaGraphExec_t exec = nullptr;
    st = ck(cudaGraphInstantiate(&exec, g, 0));
    cudaGraphDestroy(g);
    if (st != AK_OK) return st;
    m->verify_graphs[T] = exec;
    *out = exec;
    return AK_OK;
}

ak_status set_pos(ak_gguf_model* m, int pos) {
    return ck(cudaMemcpyAsync(m->d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, m->stream));
}

// Greedy-verify T tokens at positions pos..pos+T-1; out[c] = argmax after token c. KV for those positions is
// written; d_pos is left at pos.
ak_status verify(ak_gguf_model* m, int pos, const int32_t* tokens, int T, int32_t* out) {
    if (T < 1 || T > kMaxCols || pos + T > m->max_seq) {
        ak_set_error("verify: T out of range or past max_seq");
        return AK_ERR_INVALID;
    }
    cudaGraphExec_t g = nullptr;
    ak_status st = verify_graph(m, T, &g);
    if (st == AK_OK) st = set_pos(m, pos);
    if (st == AK_OK) st = ck(cudaMemcpyAsync(m->d_vtok, tokens, sizeof(int) * T, cudaMemcpyHostToDevice, m->stream));
    if (st == AK_OK) st = ck(cudaGraphLaunch(g, m->stream));
    if (st == AK_OK) st = ck(cudaMemcpyAsync(out, m->d_vout, sizeof(int) * T, cudaMemcpyDeviceToHost, m->stream));
    if (st == AK_OK) st = ck(cudaStreamSynchronize(m->stream));
    return st;
}

// Single-token decode chain on the cuda-graph backend: feed `first` at `pos`, then keep feeding the model's own
// greedy token. out[i] is the token predicted after i + 1 fed tokens. Leaves d_pos at pos + n.
ak_status decode_chain(ak_gguf_model* m, int pos, int first, int n, int32_t* out) {
    const int one = 1;
    const int zero = 0;
    ak_status st = set_pos(m, pos);
    if (st == AK_OK) st = ck(cudaMemcpyAsync(m->d_token, &first, sizeof(int), cudaMemcpyHostToDevice, m->stream));
    if (st == AK_OK) st = ck(cudaMemcpyAsync(m->d_sample, &one, sizeof(int), cudaMemcpyHostToDevice, m->stream));
    if (st == AK_OK) st = ck(cudaMemcpyAsync(m->d_count, &zero, sizeof(int), cudaMemcpyHostToDevice, m->stream));
    for (int i = 0; i < n && st == AK_OK; ++i) st = step(m, AK_BACKEND_CUDA_GRAPH, nullptr);
    if (st == AK_OK && n > 0) st = ck(cudaMemcpyAsync(out, m->d_out, sizeof(int) * n, cudaMemcpyDeviceToHost, m->stream));
    if (st == AK_OK) st = ck(cudaStreamSynchronize(m->stream));
    return st;
}

}  // namespace

extern "C" ak_status ak_gguf_verify(ak_gguf_model* m, int pos, const int32_t* tokens, int n_tokens, int32_t* out) {
    if (!m || !tokens || !out) {
        ak_set_error("gguf verify: bad arguments");
        return AK_ERR_INVALID;
    }
    return verify(m, pos, tokens, n_tokens, out);
}

extern "C" ak_status ak_gguf_speculate(ak_gguf_model* target, ak_gguf_model* draft, const int32_t* prompt, int n_prompt,
                                       int32_t* out_tokens, int n_gen, int n_draft, ak_spec_stats* stats) {
    if (!target || !draft || !prompt || n_prompt < 1 || !out_tokens || n_gen < 1 || n_draft < 1 ||
        n_draft + 1 > kMaxCols) {
        ak_set_error("gguf speculate: bad arguments (n_draft must be 1..8)");
        return AK_ERR_INVALID;
    }
    if (target->vocab != draft->vocab) {
        ak_set_error("gguf speculate: target and draft vocabularies differ");
        return AK_ERR_INVALID;
    }
    const int need = n_prompt + n_gen + n_draft + 1;
    if (need > target->max_seq || need > draft->max_seq) {
        ak_set_error("gguf speculate: prompt + generation + draft exceeds max_seq");
        return AK_ERR_INVALID;
    }
    if (stats) std::memset(stats, 0, sizeof(*stats));
    const auto t0 = std::chrono::steady_clock::now();
    // Prompt through both models, one token at a time (graph backend). The target's greedy token after the
    // prompt is the first generated token.
    ak_status st = reset_state(target);
    if (st == AK_OK) st = reset_state(draft);
    if (st == AK_OK) st = run_prompt(target, prompt, n_prompt, AK_BACKEND_CUDA_GRAPH, nullptr);
    if (st == AK_OK) st = run_prompt(draft, prompt, n_prompt, AK_BACKEND_CUDA_GRAPH, nullptr);
    int cur = 0;
    if (st == AK_OK) st = ck(cudaMemcpy(&cur, target->d_out, sizeof(int), cudaMemcpyDeviceToHost));
    if (st != AK_OK) return st;
    const auto t1 = std::chrono::steady_clock::now();

    int emitted = 0;
    out_tokens[emitted++] = cur;
    int P = n_prompt;  // both models have the KV for positions < P; cur is not yet fed
    int rounds = 0, drafted = 0, accepted = 0;
    int32_t dtok[kMaxCols] = {0};
    int32_t vin[kMaxCols] = {0};
    int32_t vout[kMaxCols] = {0};
    while (emitted < n_gen && st == AK_OK) {
        const int k = n_draft;
        st = decode_chain(draft, P, cur, k, dtok);  // dtok[i] = draft token i+1
        if (st != AK_OK) break;
        vin[0] = cur;
        for (int i = 0; i < k; ++i) vin[i + 1] = dtok[i];
        st = verify(target, P, vin, k + 1, vout);  // vout[j] = target's token after vin[0..j]
        if (st != AK_OK) break;
        int n = 0;
        while (n < k && dtok[n] == vout[n]) ++n;
        for (int i = 0; i < n && emitted < n_gen; ++i) out_tokens[emitted++] = dtok[i];
        if (emitted < n_gen) out_tokens[emitted++] = vout[n];
        if (n == k) {
            // The draft never fed its own last token; feed it so its KV covers position P + k.
            int32_t ignore = 0;
            st = decode_chain(draft, P + k, dtok[k - 1], 1, &ignore);
        }
        ++rounds;
        drafted += k;
        accepted += n;
        P += n + 1;
        cur = vout[n];
    }
    const auto t2 = std::chrono::steady_clock::now();
    if (stats) {
        stats->status = st;
        stats->rounds = rounds;
        stats->drafted = drafted;
        stats->accepted = accepted;
        stats->generated = emitted;
        stats->prompt_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        stats->decode_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
        // The first token comes from the prompt pass; rate over the rest, like ak_gguf_generate.
        stats->decode_tps = stats->decode_ms > 0 && emitted > 1 ? (emitted - 1) * 1000.f / stats->decode_ms : 0.f;
        stats->acceptance = drafted > 0 ? static_cast<float>(accepted) / drafted : 0.f;
        stats->tokens_per_round = rounds > 0 ? static_cast<float>(emitted - 1) / rounds : 0.f;
    }
    return st;
}
