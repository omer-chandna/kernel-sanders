# Next steps

Ordered by what the measurements say will move decode on this 4060. Status as of the GGUF runtime
(`native/cuda/gguf_model.cu`, Table E in `docs/RESULTS.md`).

Done: bandwidth-level Q4_K / Q6_K / Q8_0 GEMV (`qgemv.cuh`), Qwen3 GGUF loader and decode, token check against an
FP32 reference and llama.cpp, fused decode ops (`decode_ops.cuh`).

1. **Batched prefill.** Prompts are fed one token at a time through the decode graph, so time to first token is
   about prompt length times 21 ms on 8B. llama.cpp does 2420 prompt tok/s at 512. A prefill path needs a
   quantized GEMM (int8 activations times Q4_K, many columns), not the GEMV.
2. **Last 3% on 8B decode.** GEMV is at about 230 GB/s. Candidates: fewer, larger graph nodes (QKV as one GEMV over
   the concatenated rows, gate+up as one), which also cuts launch gaps; two rows per warp for x reuse. Measure in
   `ak_bench_qwen35_projections` first, then end to end.
3. **Persistent scheduler overhead.** It loses to graph replay on real models because each task costs about
   0.5 us of atomics. If it is kept: batch successor publication (one atomic per chunk group instead of per
   edge), avoid fan-in of hundreds of chunks into one task, and give GEMVs a work-stealing tile counter like the
   FP32 fat GEMV instead of hundreds of separately scheduled chunks.
4. **Beat the bandwidth wall, not the kernels.** At batch 1 the 8B ceiling is about 50 tok/s on this card. Bigger
   gains need fewer bytes per token: speculative decoding with Qwen3-0.6B as the draft (both now load), or
   batching several streams per weight read.
5. **Long context.** KV cache is FP32 and attention reads it with one block per head. Past a few thousand tokens
   attention will dominate; an f16 or int8 KV cache and a split-K attention are the next steps there.
6. **Profiling.** Nsight is still not installed. `GgufModel.profile()` gives per-op GPU time for one eager step.
