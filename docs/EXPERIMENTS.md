# Experiments

## Hardware actually used

Recorded by `python -m adakernel doctor` on 12 Nov 2026. This is not the Ryzen 5 5600X named in the project brief. The detected CPU is a Ryzen 5 7600X.

| Item | Measured value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060, sm_89, 24 SMs |
| VRAM | 8188 MB, about 7106 MB free at doctor time |
| SMs / warp / threads | 24 SMs, warp 32, 1536 threads per SM, 1024 per block |
| Registers | 65536 32-bit registers per SM |
| Shared memory | 102400 bytes per SM, 49152 default per block, 101376 opt-in |
| L2 | 24 MB |
| Memory clock / bus | 8501 MHz, 128-bit |
| Theoretical bandwidth | 272 GB/s (`2 * clock * bus width`) |
| Copy bandwidth | 246 GB/s on a 32 MB float4 copy |
| PCIe | generation 3, width 8, as reported by nvidia-smi |
| SM clock during doctor | 2505 MHz |
| Temperature during doctor | 50 C |
| Power | nvidia-smi returned N/A |
| Driver | 621.14, CUDA UMD 13.5 |
| Compiler | CUDA nvcc 12.8.93 from the user-local conda prefix, MSVC 19.44 |
| Python | 3.12.2 in `.venv` |

WSL has only `docker-desktop`. Ubuntu 24.04 is not installed, so the build is native Windows. The CUDA 13.5 installer was downloaded and then cancelled at the administrator prompt. No driver, firmware, or power-plan changes were made.

llama.cpp is pinned at `5d4c8eeb24db049acf74bd6f354bed6eaeffc7dc` (`CUDA: skip empty MoE tiles in MMQ stream-k fixup`).

## Controls

- Batch size 1, greedy argmax, FP32 weights and activations.
- One warmup generation, then the reported repetitions.
- The same weights, prompt, and generated length for every backend.
- Decode tokens/s uses the wall clock after the prompt has been consumed, divided into `n_gen - 1` following tokens. The first generated token is inside time-to-first-token.
- Kernel times are CUDA event medians around the launches. Empty-kernel times use the same method.
- A result is not called a speedup when the repetition ranges overlap.

## Models

Synthetic configs, seed 1, Gaussian weights scaled by 0.02, unless noted.

| Name | Layers | d | heads / KV | ff | vocab | prompt | generated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | 2 | 64 | 4 / 2 | 128 | 48 | 16 | 32 |
| small | 4 | 256 | 8 / 4 | 512 | 128 | 16 | 32 |
| medium | 4 | 1024 | 8 / 4 | 2048 | 256 | 16 | 32 |
| offload | 4 | 512 | 8 / 4 | 1024 | 128 | 4 | 16 |

Kernel microbenchmarks use 1024x1024 and 4096x4096 matvecs. Quantized kernels are compared with a dense matvec of the weights after dequantization, not with the original FP32 values.

## What is deliberately not claimed

- AdaKernel has not executed a Qwen checkpoint. llama.cpp has. `models/Qwen3-0.6B-Q8_0.gguf` is from `Qwen/Qwen3-0.6B-GGUF` (639,446,688 bytes, SHA256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`). Header: 28 layers, hidden 1024, 16 query heads, 8 KV heads, head dim 128, MLP 3072, RoPE base 1,000,000, RMS epsilon 1e-6, context 40960. 197 tensors are Q8_0 and 113 are FP32. `models/Qwen3-8B-Q4_K_M.gguf` is from `Qwen/Qwen3-8B-GGUF` and is 5,027,783,488 bytes, matching the Hub content length. Qwen3-14B was not downloaded.
- llama.cpp's CUDA build was started and is recorded separately. Until `llama-bench` prints a number, the llama.cpp column is NOT RUN.
- Nsight Systems and Nsight Compute were not run. Launch counts, CUDA events, occupancy, and ptxas register counts are the profiling evidence.
