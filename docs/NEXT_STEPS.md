# Next steps

Ordered by what the measurements say will move decode on this 4060.

1. **Replace the single-block matvec inside the persistent scheduler.** cuBLAS already shows a 9x matvec gap and reaches 241 GB/s on a 4096 GEMV, against about 27 GB/s for one resident block. The scheduler's advantage is concurrency. A multi-block GEMV task, or a cuBLAS call issued for a whole projection while the scheduler handles the DAG around it, is the largest open speedup. Do not unroll the current kernel again without a register budget: 66 registers already cut occupancy and slowed the microbenchmark.

2. **Load the GGUF files AdaKernel still cannot run.** llama.cpp already measured them on this GPU: Qwen3-0.6B Q8_0 at 284 decode tok/s, Qwen3-8B Q4_K_M at 50.9 decode tok/s and 2420 prompt tok/s (512 tokens, `-ngl 99 -fa on`). The next AdaKernel milestone is to load `models/Qwen3-8B-Q4_K_M.gguf` and produce greedy tokens that match llama.cpp on a short prompt. Qwen3-14B was not downloaded; it is the offload target after 8B loads.

3. **Put a fast Q4_K GEMV on the decode path only after it beats dequant-plus-cuBLAS.** The current kernel matches the super-block layout and loses badly. A win has to show up in tokens per second on a real matrix shape (4096 by 4096 and 4096 by 12288), not only in a smaller error.

4. **GGUF loader for Qwen3.** Tensor names are the llama.cpp ones (`blk.N.attn_q`, `attn_q_norm`, `ffn_gate`, `ffn_up`, `ffn_down`). Q4_K_M mixes Q6_K on some tensors. CPU dequant of Q6_K is implemented; it is not wired to the model. Tokenizer should be `llama-tokenize`, not a new BPE.

5. **Copy-engine overlap for layers that miss.** Same-stream copies are correct and 3.3x slower when every layer misses, on a 36 MB model and a PCIe 3.0 x8 link. Overlap is worth doing when the working set exceeds VRAM. It will not make host-resident weights as fast as on-GPU weights on this board.

6. **Profiling.** Nsight was not installed. After the GEMV change, capture one medium decode with `nsys profile` and one matvec with Nsight Compute, and check whether the persistent kernel is bandwidth-bound or waiting on atomics. The current evidence says the serial graph is launch- and dependency-bound, and the single-block matvec is not at the 246 GB/s the copy kernel reaches.

7. **Do not pursue** a larger fused megakernel that holds a whole layer in one block, a new quantization format, or a cooperative `grid.sync()` decode. The residency math on 24 SMs already forbids a cooperative grid big enough for a 4096-row GEMV, and the worker sweep says extra resident blocks past one per SM are not the bottleneck.
