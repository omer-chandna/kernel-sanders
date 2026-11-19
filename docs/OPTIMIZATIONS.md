# Optimizations

Each change was measured on the RTX 4060. "Keep" means the final tree contains it.

## Kept

**Device-side DAG instead of one launch per task.** Hypothesis: independent Q, K, V, gate, and up chunks are waiting on each other only because they share a stream. Change: persistent kernel, tickets, predecessor counts, occupancy-capped grid. Medium decode went from 137 tok/s eager and 159 tok/s CUDA Graph to about 1150 tok/s. Tokens matched. Keep.

**CUDA Graph around that persistent kernel (hybrid).** Hypothesis: the reset kernel plus the persistent kernel still pay two launches. Tiny decode moved from 11502 to 11669 tok/s. Medium did not move outside the repetition range. Keep as a backend. It is the best tiny configuration and tied for the best medium one.

**cuBLAS SGEMV for the serialized baseline.** Hypothesis: the hand-written matvec is bandwidth-starved inside one block. 1024x1024 matvec: 91 us versus cuBLAS 10 us. Medium decode: 895 tok/s versus eager 137. It still loses to the persistent scheduler (1150) because the GEMVs remain serial. Keep as a backend, not as a replacement for the scheduler.

**64-row chunks.** Sweep on the medium persistent model: 16 rows 663 tok/s, 64 rows 1157, 1024 rows 315. Keep 64.

**Occupancy cap.** The grid is `min(requested, resident blocks)`. Eight workers: 834 tok/s. Twenty-four through ninety-six: 1154–1164, overlapping. Keep the cap. Do not hard-code 24.

**Offload scratch.** Layers that do not fit are pinned and copied before use. Required for the 0-GPU-layer run. Correct, slower. Keep.

**int8-activation dp4a GEMV for Q4_K, Q6_K, Q8_0 (`native/cuda/qgemv.cuh`).** Hypothesis: the old quantized kernels decode every weight to float, re-derive 6-bit scales per element, and issue one-byte loads, so they are instruction-bound, not bandwidth-bound. Change: quantize the activation once to 32-value int8 blocks, then dot 16-byte (Q4_K) or 16-bit-aligned (Q6_K, Q8_0) weight loads against it with `__dp4a`, one warp per row, 8 lanes per super-block. Measured with `ak_bench_qwen35_projections` over every projection in the real files (embedding excluded), 10 iterations:

| File | Bytes / token | Old kernels | dp4a | Q4_K | Q6_K |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B Q4_K_M (216 Q4_K, 37 Q6_K) | 4.67 GB | 72.0 ms, 65 GB/s | 20.0 ms, 234 GB/s | 234 GB/s | 230 GB/s |
| Qwen3-0.6B Q8_0 (196 tensors) | 0.47 GB | 8.3 ms, 56 GB/s | 2.5 ms, 186 GB/s | — | — |

234 GB/s is 86% of the 272 GB/s theoretical and 95% of the 246 GB/s copy kernel. 20 ms of projections per token is the same budget llama.cpp's 50.9 tok/s implies, so decode on 8B is now at the bandwidth wall and further gains have to come from elsewhere (fewer bytes, batching, speculation). The 0.6B number is lower because its matrices are 1–3 MB each and launch and tail effects show up: the case a persistent scheduler can still help. Kernel arithmetic matches the dequantized-int8 reference to 4e-7 relative; total error against the float activation is 1.2% on a test vector with 20x outliers, which is the int8 activation rounding llama.cpp also accepts. New task ops `AK_OP_QUANT_Q8_1` and `AK_OP_Q{8,4K,6K}_DP4A`. Adding them to the persistent dispatch switch pushed `persist_kernel` from 78 to 114 registers (72 to 48 resident blocks); `__launch_bounds__(256, 3)` holds it at 80 registers and 72 blocks with a 24-byte spill. Medium persistent decode 1098 vs 1122 tok/s, hybrid 1156 vs 1165, one run each. Keep. The engine does not call these ops yet: it has no GGUF loader.

**Fused decode ops for the GGUF runtime (`native/cuda/decode_ops.cuh`).** Hypothesis: on real models the
small ops are launch- and dependency-bound, and attention with one thread per key is uncoalesced. Change: residual +
RMSNorm + activation quantize as one task; QK-norm + RoPE + KV append as one; attention as warp-per-key with int8
output (no separate quantize) and all heads in one grid launch for the graph backends; SiLU + quantize; argmax as
parallel parts plus a packed atomicMax. 8B cuda-graph went from 36.3 to 48.1 tok/s, 0.6B from 162.6 to 291.3 (one
run each, 32 tokens). Tokens unchanged. Keep.

**Persistent chunk size 128 for the GGUF runtime.** Sweep in `docs/RESULTS.md` Table E. Keep as default.

**Speculative decoding with a multi-column verify pass (`ak_gguf_speculate`).** Hypothesis: at batch 1 the target
is bandwidth-bound, so checking several drafted tokens per weight read is nearly free. Change: `qgemv_dp4a_rows_mc`
dots up to 9 activation columns per weight load, sharing the per-column arithmetic with the one-token kernel so
results are bit-identical; a T-column forward pass (`ak_gguf_verify`) with causal attention per column; the 0.6B
model drafts with CUDA graph replays. 8B 1.76x, 4B 1.36x over greedy at k=3, exact greedy output. Keep.

## Reverted

**Four-way float4 accumulator in the matvec.** Hypothesis: extra independent accumulators would hide load latency and raise the 27 GB/s single-block matvec. Result: registers rose from 56 to 66, resident blocks fell from 96 to 72, 1024x1024 matvec went from 91.1 us to 96.3 us, 4096 stayed at about 2390 us. No spill, but no win. Reverted. End-to-end was not re-measured after the microbenchmark loss.

## Not a win, left in as a separate backend

**Fused residual + RMSNorm.** Tiny +6%, small +4%, medium inside noise. The kernel stays available as `--backend fused`. It is not the default schedule for the persistent path, so the persistent-versus-graph comparison is the same math.

## Not competitive

**Q8_0 and Q4_K matvecs.** They implement the llama.cpp block layouts and match a dense matvec of the dequantized values (absolute error under 1e-8 on the random trials). At 1024 columns they take 466 us and 599 us against a 91 us FP32 matvec and a 10 us cuBLAS GEMV. They are correctness vehicles. They are not on the decode path.
