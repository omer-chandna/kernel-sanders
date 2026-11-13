# Architecture

AdaKernel runs one token at a time through a Qwen3-style decoder:

1. Token embedding.
2. For each layer: RMSNorm, Q/K/V matvecs, optional per-head QK RMSNorm, NeoX RoPE, KV append, grouped-query attention, output projection, residual, SwiGLU, residual.
3. Final RMSNorm, vocabulary matvec, greedy argmax.

The same device functions implement every backend. Eager, CUDA Graph, and persistent runs are not three different math implementations. A difference in tokens is treated as a bug. cuBLAS GEMV is the exception: it replaces only the matvec and is allowed a small numeric difference.

## Task

```text
op, predecessor count, successor range, sizes, one float, five device pointers
```

The host builds a DAG whose edges always point from an earlier task id to a later one, then checks that a serial walk visits every task. Matvecs are split into row chunks. Q, K, and V chunks share the RMSNorm as their only predecessor, so they have no edge between them.

## Backends

| Backend | What one token does |
| --- | --- |
| eager | One CUDA launch per task, in task-id order, on one stream |
| cuda-graph | That same launch sequence, captured once and replayed |
| fused | Eager schedule with residual+RMSNorm fused into one task |
| cublas | One cuBLAS SGEMV per projection instead of row chunks |
| persistent | One reset kernel plus one persistent kernel |
| hybrid | The reset kernel and the persistent kernel captured as a CUDA Graph |

The persistent kernel launches only as many blocks as `cudaOccupancyMaxActiveBlocksPerMultiprocessor` says can stay resident. On this 4060 that is 96 blocks of 256 threads (56 registers, about 520 bytes of shared memory, 4 blocks per SM). A block that has finished its tasks returns. It does not spin for the rest of the launch, and it does not take part in a grid-wide barrier.

Each block claims a ticket with `atomicAdd`. The ticket waits until that slot's ready flag is published or until about a billion SM cycles pass, then the block exits with a timeout error. Publishing a successor is an atomic decrement of its remaining predecessors. The last predecessor writes the task id and sets the flag, with a device threadfence so the consumer sees the outputs. There is no `grid.sync()`. A cooperative launch is used only for a one-kernel residency check in the microbenchmark, not for decode.

KV length and the sampled token live in device memory. The captured graph does not bake in a sequence length. The host copies the next prompt token in, or leaves the device argmax in place during decode. Each launch is one token, so a kernel cannot sit on the GPU long enough to trip Windows timeout detection under the sizes tested here.

## Weights that do not fit

Norms, embeddings, and the KV cache stay on the device. Projection matrices for layers past `n_gpu_layers` sit in pinned host memory. Before those layers, the eager and CUDA Graph paths copy one layer into a device scratch arena on the same stream, then run the kernels. The copy is ordered with the kernels, so the scratch can be reused. The persistent scheduler rejects this path: its tasks hold device pointers for the whole token, and a host copy cannot be interleaved inside that single launch.

`n_gpu_layers = -1` asks the loader to keep every layer it can fit in 82% of free VRAM after activations, KV, and one scratch layer.

## Quantized kernels

Q8_0 is the llama.cpp block: FP16 scale plus 32 int8 values, 34 bytes. Q4_K follows the llama.cpp super-block: FP16 `d` and `dmin`, 12 bytes of 6-bit scales and mins, 128 bytes of nibbles, 144 bytes covering 256 weights. Dequantization uses the same nibble pairing and `get_scale_min_k4` packing as `ggml-quants.c`. The quantizer itself is a direct min/max fit into that layout, not llama.cpp's iterative search, so a Q4_K_M GGUF file is not bit-identical to weights produced here. The model runtime does not call these kernels yet. Decode is FP32.

## Reference

`adakernel/reference.py` is the NumPy decoder used by the tests. It is the correctness target for logits. Backend-to-backend token identity is checked separately, because those paths share device code and should not drift even when both drift slightly from NumPy's summation order.
