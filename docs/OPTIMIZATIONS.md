# Optimizations

Each change was measured on the RTX 4060. "Keep" means the final tree contains it.

## Kept

**Device-side DAG instead of one launch per task.** Hypothesis: independent Q, K, V, gate, and up chunks are waiting on each other only because they share a stream. Change: persistent kernel, tickets, predecessor counts, occupancy-capped grid. Medium decode went from 137 tok/s eager and 159 tok/s CUDA Graph to about 1150 tok/s. Tokens matched. Keep.

**CUDA Graph around that persistent kernel (hybrid).** Hypothesis: the reset kernel plus the persistent kernel still pay two launches. Tiny decode moved from 11502 to 11669 tok/s. Medium did not move outside the repetition range. Keep as a backend. It is the best tiny configuration and tied for the best medium one.

**cuBLAS SGEMV for the serialized baseline.** Hypothesis: the hand-written matvec is bandwidth-starved inside one block. 1024x1024 matvec: 91 us versus cuBLAS 10 us. Medium decode: 895 tok/s versus eager 137. It still loses to the persistent scheduler (1150) because the GEMVs remain serial. Keep as a backend, not as a replacement for the scheduler.

**64-row chunks.** Sweep on the medium persistent model: 16 rows 663 tok/s, 64 rows 1157, 1024 rows 315. Keep 64.

**Occupancy cap.** The grid is `min(requested, resident blocks)`. Eight workers: 834 tok/s. Twenty-four through ninety-six: 1154–1164, overlapping. Keep the cap. Do not hard-code 24.

**Offload scratch.** Layers that do not fit are pinned and copied before use. Required for the 0-GPU-layer run. Correct, slower. Keep.

## Reverted

**Four-way float4 accumulator in the matvec.** Hypothesis: extra independent accumulators would hide load latency and raise the 27 GB/s single-block matvec. Result: registers rose from 56 to 66, resident blocks fell from 96 to 72, 1024x1024 matvec went from 91.1 us to 96.3 us, 4096 stayed at about 2390 us. No spill, but no win. Reverted. End-to-end was not re-measured after the microbenchmark loss.

## Not a win, left in as a separate backend

**Fused residual + RMSNorm.** Tiny +6%, small +4%, medium inside noise. The kernel stays available as `--backend fused`. It is not the default schedule for the persistent path, so the persistent-versus-graph comparison is the same math.

## Not competitive

**Q8_0 and Q4_K matvecs.** They implement the llama.cpp block layouts and match a dense matvec of the dequantized values (absolute error under 1e-8 on the random trials). At 1024 columns they take 466 us and 599 us against a 91 us FP32 matvec and a 10 us cuBLAS GEMV. They are correctness vehicles. They are not on the decode path.
