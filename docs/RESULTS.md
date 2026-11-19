# Results

All numbers below were measured on the RTX 4060 described in `docs/EXPERIMENTS.md`. Ranges are the minimum and maximum of the recorded repetitions. They are not confidence intervals. llama.cpp token rates are **NOT RUN** until `experiments/results/llamacpp_baseline.json` says otherwise.

## Table A. Model feasibility

Synthetic FP32 models. "File size" is the weight footprint, not a downloaded checkpoint.

| Model | Quantization | Weight size | GPU alloc | Host weights | Context | Placement | Decode tok/s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny, 2x64 | FP32 | under 1 MB | resident | 0 | 64 | all GPU | 3818 eager, 11669 hybrid | tokens match NumPy |
| small, 4x256 | FP32 | a few MB | resident | 0 | 128 | all GPU | 798 eager, 4364 hybrid | tokens match eager |
| medium, 4x1024 | FP32 | tens of MB | resident | 0 | 256 | all GPU | 137 eager, 1153 hybrid | tokens match eager |
| offload, 4x512 | FP32 | 36.5 MB | 46 MB with all layers on GPU | 0 to 36 MB | 64 | 4, 2, 1, or 0 GPU layers | 380, 175, 143, 116 | tokens match the resident run |
| synthetic 24x2048, ff=5504 | FP32 | 4057 MB weights | 4233 MB | 0 | 64 | all 24 layers on GPU | 5.20 eager | ran |
| synthetic 40x2048, ff=5504 | FP32 | 6762 MB weights | 5758 MB | 1183 MB | 64 | 33 GPU layers, 7 host | 2.15 eager | ran, larger than the resident budget |
| synthetic 48x2048, ff=5504 | FP32 | 8114 MB weights | 5760 MB | 2535 MB | 64 | 33 GPU layers, 15 host | 1.36 eager | ran |
| Qwen3-0.6B Instruct Q8_0 | Q8_0 GGUF | 639,446,688 bytes, 596M params | llama.cpp, all layers | — | bench default | llama.cpp `-ngl 99`, flash attention on, KV f16 | 284.1 decode, 18361 prefill at 512 | llama.cpp only. SHA256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| Qwen3-8B Q4_K_M | Q4_K_M | 5,027,783,488 bytes | llama.cpp, all layers, short bench context | — | bench default | llama.cpp `-ngl 99 -fa on` | 50.94 decode, 2420 prefill at 512 | ran in llama.cpp, not in AdaKernel |
| Qwen3-14B Q4_K_M | Q4_K_M | not downloaded | — | — | — | — | — | **NOT RUN** |

The capacity probe in `experiments/results/capacity_probe.json`, when present, is the larger synthetic attempt. It is not a Qwen checkpoint.

## Table B. Kernel performance

1024x1024, 30 iterations after 5 warmups, except the 32-launch rows. The 4096x4096 matvec row is from a 20-iteration run. Absolute error versus a serial FP32 reference is about 1e-8 or smaller. One 4096 run initially flagged a relative error of 0.002 because the reference values themselves are ~1e-5; that is summation noise, and the later run with the same kernel marks those rows correct under a 1e-4 absolute or 5e-3 relative rule.

| Kernel | Reference | This kernel | Speedup vs that reference | Correct | Notes |
| --- | --- | --- | --- | --- | --- |
| RMSNorm, n=1024 | — | 6.14 us | — | yes | max abs 8e-7 |
| Residual | — | 4.99 us | — | exact | |
| SiLU * mul | — | 5.12 us | — | yes | |
| Matvec 1024 | cuBLAS 10.24 us | 91.1 us | 0.11x | yes | one block, warp per row, float4 |
| Matvec 4096 | cuBLAS 266 us | 2388 us | 0.11x | yes, abs 2e-8 | cuBLAS uses the whole GPU |
| Q8_0 matvec 1024 | dequant + FP32 matvec | 466 us | slower than FP32 matvec | yes | |
| Q4_K matvec 1024 | dequant + FP32 matvec | 599 us | slower than FP32 matvec | yes | real 6-bit scales, not flat int4 |
| Q4_K matvec 4096 | dequant + FP32 matvec | 8356 us | slower | yes | |
| 32 empty launches | eager 171 us | graph 18.4 us | 9.3x | n/a | 5.3 us vs 0.58 us per launch |
| 32 serial empty tasks | graph 18.4 us | persistent 56.3 us | 0.33x | n/a | persistent is slower than a graph here |
| Device copy | 272 GB/s theoretical | 246 GB/s | 0.90 of peak | n/a | 32 MB float4 |

## Table C. Execution backends

Five repetitions, one warmup, 16 prompt tokens, 32 generated tokens, batch 1, greedy. Speedup is median decode tok/s divided by eager. Token identity is against eager. Eager matches the NumPy reference on the tiny model.

| Backend | Model | Decode tok/s | TTFT is prefill | Median tok/s range | Peak alloc | Launches / token | vs eager |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eager | tiny | 3818 | included in prefill | 3769–3868 | resident | 50 | 1 |
| cuda-graph | tiny | 10015 | | 9900–10022 | | 1 replay, 50 nodes | 2.62x |
| fused | tiny | 4054 | | 4023–4212 | | 46 | 1.06x |
| cublas | tiny | 3982 | | 3895–4125 | | 46 | 1.04x |
| persistent | tiny | 11502 | | 11300–11519 | | 2 | 3.01x |
| hybrid | tiny | 11669 | | 11651–11679 | | 1 | 3.06x |
| eager | small | 798 | | 760–817 | | 205 | 1 |
| cuda-graph | small | 1580 | | 1579–1582 | | 1 | 1.98x |
| fused | small | 831 | | 813–835 | | 197 | 1.04x |
| cublas | small | 1722 | | 1700–1740 | | 104 | 2.16x |
| persistent | small | 4327 | | 4304–4329 | | 2 | 5.42x |
| hybrid | small | 4364 | | 4329–4367 | | 1 | 5.47x |
| eager | medium | 137 | | 136–138 | | 591 | 1 |
| cuda-graph | medium | 159 | | 159–160 | | 1 | 1.16x |
| fused | medium | 136 | | 136–137 | | 583 | 1.00x, inside noise |
| cublas | medium | 895 | | 890–901 | | 104 | 6.54x |
| persistent | medium | 1150 | | 1148–1152 | | 2 | 8.40x |
| hybrid | medium | 1153 | | 1150–1154 | | 1 | 8.43x |

Hybrid and persistent overlap on the medium model. Hybrid is clearly ahead of persistent on tiny, where two host launches still show up. cuBLAS beats the CUDA Graph once the matvecs are large, and the persistent scheduler beats cuBLAS on small and medium because Q, K, V, gate, and up are independent tasks and run together. The graph and the cuBLAS path run those kernels one after another on one stream.

## Table D. Memory

Single run, eager backend, 4-layer d=512 model, 16 generated tokens. Every placement produced the same token ids.

| GPU layers | Host weights | GPU alloc | Decode tok/s | tok/s vs 4 layers | Correct |
| --- | --- | --- | --- | --- | --- |
| 4 | 0 MB | 46 MB | 380 | 1 | yes |
| 2 | 18 MB | 28 MB | 175 | 0.46x | yes |
| 1 | 27 MB | 19 MB | 143 | 0.37x | yes |
| 0 | 36 MB | 10 MB | 116 | 0.30x | yes |

Streaming every layer across PCIe 3.0 x8 costs about 3.3x decode throughput on this small model. The model still fits in RAM. The 40- and 48-layer synthetic models do not: the loader keeps 33 layers on the GPU (about 5.6 GB of weights) and leaves the rest in pinned memory. Decode falls from 5.20 tok/s at 24 resident layers to 2.15 tok/s at 40 layers and 1.36 tok/s at 48. Part of that drop is simply more layers.

llama.cpp on the real 0.6B model is a different mechanism: layers that are not offloaded to the GPU are computed on the CPU. Five repetitions, flash attention on, KV cache f16:

| GPU layers | Prompt 512 tok/s | Prompt 128 tok/s | Decode 128 tok/s |
| --- | --- | --- | --- |
| 99 (all 28) | 18361 (sd 469) | 13685 (sd 1333; first sample 11302, the other four 14196–14341) | 284.1 (sd 0.7) |
| 14 | — | 1945 (sd 157, 3 reps) | 34.2 (sd 0.3) |
| 0 | — | 1008 (sd 41, 3 reps) | 15.5 (sd 0.0) |

Full GPU decode is 284 tok/s. Half the layers on the CPU is 34 tok/s, about 8x slower. CPU-only decode is 15.5 tok/s, about 18x slower than full GPU. Commit `5d4c8ee`, CUDA backend, `ggml-cuda` built with `-DNDEBUG -O2`. The bench executable itself printed "asserts enabled" because that one translation unit was not built with `NDEBUG`. The JSON is `experiments/results/llamacpp_0.6b.json` and `llamacpp_0.6b_offload.json`.

## Scheduler sweep

Medium model, persistent backend, three repetitions, 16 generated tokens. Chunk size is rows per matvec task. Worker count is the persistent grid, clamped to the 96 resident blocks.

| Chunk rows | Workers | Median tok/s |
| --- | --- | --- |
| 16 | auto (96) | 663 |
| 32 | auto | 1077 |
| 64 | auto | 1157 |
| 128 | auto | 1088 |
| 256 | auto | 862 |
| 512 | auto | 564 |
| 1024 | auto | 315 |
| 64 | 8 | 834 |
| 64 | 16 | 1122 |
| 64 | 24 | 1164 |
| 64 | 48 | 1161 |
| 64 | 72 | 1154 |
| 64 | 96 | 1159 |

Chunk size changes throughput by more than 3x. Worker counts from 24 through 96 sit on top of each other. Eight workers are slower. The default remains 64-row chunks and the occupancy cap.

## Table E. Real Qwen3 GGUF decode (AdaKernel vs llama.cpp)

`python -m benchmarks.gguf_bench --model <file>`, run back to back with llama-bench in the same session so both
see the same clocks. Prompt "The capital of France is" (5 tokens, fed one at a time), 128 greedy decode tokens,
5 repetitions, median. llama.cpp is `llama-bench -p 0 -n 128 -ngl 99 -fa 1`. JSON in
`experiments/results/gguf_<model>.json`.

| Model | llama.cpp tg128 | AdaKernel cuda-graph | eager | persistent | hybrid |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q8_0 | 253.4 (sd 2.4) | **269.0** (252.0-271.4) | 223.6 | 173.5 | 173.1 |
| Qwen3-8B Q4_K_M | **48.9** (sd 0.08) | 47.4 (47.3-47.4) | 45.8 | 41.1 | 41.4 |

All four backends produce identical tokens. The llama.cpp numbers are lower than the 284 / 50.9 recorded earlier
in this file; those were a different session, which is why this table is measured back to back.

Second session, after adding Qwen3-4B Q4_K_M (official `Qwen/Qwen3-4B-GGUF` at commit `bc64014`, 2,497,280,256
bytes, SHA256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`, tied Q6_K embedding/output) and
merging Q+K and gate+up into single GEMV launches on the graph path:

| Model | llama.cpp tg128 | AdaKernel cuda-graph | eager | hybrid | ratio |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q8_0 | 276.8 (sd 0.3) | 274.5 (270.9-275.9) | 243.5 | 171.0 | 0.99 |
| Qwen3-4B Q4_K_M | 84.4 (sd 0.3) | **86.3** (84.9-86.3) | 82.1 | 66.8 | 1.02 |
| Qwen3-8B Q4_K_M | **49.9** (sd 0.35) | 48.3 (48.0-48.4) | 47.1 | 42.3 | 0.97 |

llama.cpp itself moved 9-14% between the two sessions on 0.6B and 4B (GPU clock state), so only ratios measured in
the same session are comparable. Across both sessions AdaKernel's CUDA graph path is within about 3-6% of llama.cpp
on all three models, ahead on 4B both times and behind on 8B both times. Treat 0.6B as a tie.

4B greedy text matches llama.cpp token for token over 32 tokens. Per-op GPU time for one eager 4B step at about 100
positions: Q4_K GEMV 8.1 ms, Q6_K GEMV 3.0 ms, attention 0.84 ms, residual+norm+quantize 0.70 ms (single block,
9.6 us each), QK-norm+RoPE 0.47 ms, everything else under 0.2 ms.

Null results this session:

- Half-warp rows for the K-quant GEMV (two super-blocks per pass, so K = 2560 has no idle lanes). Interleaved A/B in
  one process: 4B 225.2 / 224.1 and 228.1 / 227.8 GB/s (whole warp / half warp), 8B 236.6 / 236.7 and
  236.5 / 236.9. No difference. An earlier single run showing 200 -> 220 GB/s on 4B, and another showing 8B Q4_K at
  171 GB/s, were GPU clock drift. Kept (harmless); `ak_bench_qwen35_projections` now reports both layouts
  interleaved (`lpr32_gbps`, `lpr16_gbps`) so this stays checkable.
- Q+K and gate+up as one GEMV launch each, QK-norm+RoPE as a multi-block grid: no gain against llama.cpp that rises
  above the session-to-session noise. Kept: fewer launches, same tokens.

Correctness: on 0.6B, logits after the prompt are within 1.8% relative L2 of a NumPy FP32 forward pass over the
same dequantized weights, with the same argmax (`tests/test_gguf.py`). The generated text matches llama.cpp for six
tokens, then splits at a near-tie (" France" 16.64 vs " Italy" 16.51 in our logits; 16.61 vs 16.55 in the FP32
reference). The FP32 reference picks " France", as AdaKernel does; llama.cpp picks " Italy".

What this says:

- On 8B, decode is memory-bound and both runtimes sit on the same wall: 4.67 GB of weights per token. The dp4a
  GEMVs reach about 230 GB/s (85% of 272 GB/s); llama.cpp's whole token implies at least 228 GB/s. The remaining
  3% gap is GEMV efficiency and launch gaps between about 470 graph nodes.
- On 0.6B, AdaKernel's CUDA graph beats llama.cpp by 6%: matrices are small, so the fused small ops (residual +
  RMSNorm + quantize, QK-norm + RoPE + KV write, attention writing int8, SiLU + quantize, split argmax) matter.
- The persistent device-side scheduler (the megakernel) is the slowest backend on real models. Best chunk size is
  128 rows: 192 tok/s on 0.6B, 42 on 8B. Each extra task costs about 0.5 us of wall time (atomic ticket, atomic
  predecessor decrements, atomic completion count), and that bookkeeping outweighs the overlap it buys once the
  kernels themselves are fast. Its earlier wins in Table C were against slow FP32 kernels on synthetic models.

Chunk sweep, hybrid backend, 64 decode tokens:

| Chunk rows | 0.6B tasks | 0.6B tok/s | 8B tasks | 8B tok/s |
| --- | --- | --- | --- | --- |
| 32 | 16185 | 93 | 50293 | 28.1 |
| 64 | 8435 | 148 | 26031 | 41.1 |
| 128 | 4560 | 192 | 13900 | 42.0 |
| 256 | 2623 | 174 | 7835 | 36.5 |
| 512 | 1654 | 124 | 4802 | 29.0 |

Per-op GPU time for one eager 8B step before the fused ops (about 100 cached positions): attention 22.9 ms over
1152 single-head launches, Q4_K GEMV 15.4 ms, Q6_K GEMV 5.5 ms, every other op under 1 ms. The rewritten attention
(warp per key, coalesced K loads, int8 output) and one grid launch for all heads removed most of the first item.

## Table F. Speculative decoding (Qwen3-0.6B Q8_0 drafting)

`python -m benchmarks.spec_bench --target <file>`: four prompts (story, explanation, code, capital-cities facts),
128 greedy tokens, median of 3. Greedy speculative decoding: the draft proposes k tokens, the target checks all
k + 1 positions in one verify pass (`ak_gguf_verify`: every weight matrix is read once for all columns), the longest
agreeing prefix is kept plus the target's own next token. Every run reproduced the target's plain greedy tokens
exactly (checked per run, and in `tests/test_gguf.py`). JSON: `experiments/results/spec_<model>.json`.

Speedup over plain greedy decoding of the same target, CUDA graph backend:

| Target | greedy tok/s | k=2 | k=3 | k=4 | k=5 | k=6 | k=8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B Q4_K_M | 48.9 | 1.62x | **1.76x** | 1.72x | 1.65x | 1.69x | 1.46x |
| Qwen3-4B Q4_K_M | 87.5 | 1.28x | **1.36x** | 1.31x | 1.20x | 1.27x | 1.11x |

Mean over the four prompts. Per prompt at k=3: 8B 1.56x (explain) to 2.05x (facts), 77-100 tok/s; 4B 1.11x
(story) to 1.58x (facts), 98-139 tok/s. 8B gains more because a 0.6B draft step (about 3.5 ms) is a smaller share of
its 20 ms step than of 4B's 11.5 ms one. k=3 is the default.

Against llama.cpp's `llama-speculative` (same target, same 0.6B draft, `--spec-draft-n-max 3`, `-ngl 99 -ngld 99
-fa on --temp 0`), same prompts, tok/s:

| Target / prompt | llama.cpp speculative | AdaKernel k=3 | ratio |
| --- | --- | --- | --- |
| 8B story | 65.5 (accept 59%) | 77.4 | 1.18 |
| 8B explain | 59.5 (50%) | 76.0 | 1.28 |
| 8B code | 73.9 (72%) | 89.8 | 1.22 |
| 8B facts | 80.4 (82%) | 100.3 | 1.25 |
| 4B story | 77.1 (54%) | 97.5 | 1.26 |
| 4B explain | 94.5 (77%) | 125.8 | 1.33 |
| 4B code | 82.7 (61%) | 114.8 | 1.39 |
| 4B facts | 96.5 (79%) | 138.8 | 1.44 |

`llama-speculative` is llama.cpp's example program; llama-server's speculative path was not measured. JSON:
`experiments/results/llamacpp_speculative.json`.

Verify-pass cost against width T (tokens checked at once), ms, after capping the wide kernels at 64 registers:

| Model | T=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4B | 11.33 | 11.44 | 11.71 | 12.12 | 13.77 | 15.01 | 16.12 | 17.89 | 20.50 |
| 8B | 20.34 | 20.47 | 20.89 | 21.56 | 23.90 | 26.37 | 28.61 | 31.96 | 36.36 |

Checking 4 tokens costs 7% more than checking 1. Before the register cap (`__launch_bounds__(128, 8)`, no unroll in
the wide variants), Q4_K at T >= 6 used 183-191 registers and T=6 cost 18.97 ms on 4B.

## Answers

1. **Is the persistent scheduler faster than a CUDA Graph for single-token decode on this 4060?** Yes for the three synthetic models. The medium gap is 1153 vs 159 tok/s. It is not faster for a serial chain of empty kernels: the graph replay costs 0.58 us per empty kernel and the persistent chain costs 1.76 us.

2. **How much time is launches versus math?** An empty eager launch is about 5.3 us. Fifty of those are already 0.27 ms, which is visible on the tiny model (eager 3818 tok/s vs graph 10015). On the medium model the graph only gains 16% over eager, so most of the remaining ~6 ms/token is serialized GPU work. The persistent scheduler cuts that to about 0.87 ms/token by running independent matvec chunks at once, not by making an empty launch cheaper than a graph.

3. **Which ops benefit from fusion?** Fusing residual add with the following RMSNorm is a 6% decode gain on tiny and 4% on small. On medium the ranges overlap (136–138 vs 136–137). Q4_K and Q8_0 fused dequant-matvec are correct and several times slower than an FP32 matvec, which is itself about 9x slower than cuBLAS GEMV.

4. **Do occupancy or registers limit the megakernel?** The persistent kernel uses 56 registers and no spill, which allows 4 of the 6 thread-slots per SM (1536 threads, 256-thread blocks). A four-accumulator unroll went to 66 registers, 72 resident blocks, and a slightly slower 1024 matvec (96 us vs 91 us). It was reverted. Using fewer than 24 workers hurts. Using more than 24, up to 96, does not.

5. **Fastest verified configuration for the primary benchmark model?** For Qwen3-8B Q4_K_M the only working configuration is llama.cpp: `-ngl 99 -fa on`, 50.94 decode tok/s and 2420 prompt tok/s at 512 tokens. AdaKernel does not run this file. For the synthetic medium model the fastest AdaKernel configuration is the hybrid persistent graph with 64-row chunks, about 1150 tok/s.

6. **Largest model that ran?** llama.cpp runs Qwen3-8B Q4_K_M (file 5,027,783,488 bytes, weights reported as 5,021,827,072) entirely on the GPU at the bench context: 50.94 decode tok/s (sd 0.01, 3 reps, 128 tokens) and 2420 prompt tok/s at 512 tokens (sd 7). That context is the bench default, not 32k; KV cache at 512 tokens is small. In AdaKernel, a 48-layer FP32 model (d=2048, MLP 5504, about 8.1 GB of weights) generates with 33 layers on the GPU and 15 in pinned RAM, at 1.36 tok/s eager. The largest fully resident synthetic model in that sweep is 24 layers, 4.06 GB of weights, 5.20 tok/s. Qwen3-14B was not downloaded.

7. **What does offload cost?** On the 36 MB AdaKernel model, moving all four layers to pinned RAM and still computing them on the GPU drops decode from 380 to 116 tok/s. Tokens stay identical. On Qwen3-0.6B in llama.cpp, computing every layer on the CPU instead of the GPU drops decode from 284 tok/s to 15.5 tok/s. Leaving 14 of 28 layers on the CPU drops it to 34 tok/s. Those two mechanisms are not the same: one copies weights onto the GPU, the other runs the missed layers on the CPU. The link in this machine is PCIe 3.0 x8.

8. **What helped end-to-end, not just a microbenchmark?** The persistent DAG, and cuBLAS GEMV versus the hand-written matvec. Chunk size 64 versus 16 or 1024. Fusion and the matvec unroll did not help the medium decode.

9. **Where does this beat or lose to llama.cpp?** llama.cpp runs Qwen3-0.6B Q8_0 at 284 decode tok/s and about 18k prompt tok/s at 512 tokens, with flash attention, on this GPU. AdaKernel does not load that file. Its fastest measured decode is about 1150 tok/s on a 4-layer d=1024 FP32 toy, and 5.2 tok/s eager on a 24-layer d=2048 FP32 model. Those are different shapes, precisions, and kernels. The toy can look faster because it does much less work per token. On a real Qwen checkpoint, llama.cpp is the system that runs, and this runtime does not. The hand-written matvec is also about 9x slower than cuBLAS on a 1024x1024 GEMV, which is the class of kernel llama.cpp already uses.

10. **What should the next iteration change?** Put a bandwidth-saturating GEMV, likely cuBLAS or a multi-block kernel, behind the same task interface so the persistent scheduler stops using a 27 GB/s single-block matvec. Then load real Q4_K_M weights, with a Q4 kernel that is actually faster than dequant-plus-GEMV. Overlap layer copies on the copy engine only after a Qwen-sized model forces offload. The PCIe link on this board is generation 3 x8, so that overlap has a hard ceiling.
