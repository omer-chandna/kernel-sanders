# Experiment log

## Environment

- Windows 11, Ryzen 5 7600X, 32 GB, RTX 4060 8 GB, driver 621.14.
- WSL Ubuntu is not installed. Development stayed on Windows.
- CUDA 13.5 installer was cancelled at the UAC prompt. nvcc 12.8.93 was installed with conda into `.conda-cuda` without changing the driver.
- Tests: 13 passed (`pytest tests`).

## Runs

- Kernel bench 1024 and 4096, inference bench tiny/small/medium (5 reps), memory bench, scheduler sweep. JSON under `experiments/results/`.
- Four-accumulator matvec: 56 to 66 registers, 96 to 72 blocks, 1024 matvec 91.1 us to 96.3 us. Reverted.
- Chunk sweep and worker sweep kept the defaults (64 rows, occupancy cap).

## llama.cpp

- Second configure found MSVC 19.44. The first failed because a `set PATH=...%PATH%` wiped the vcvars environment.
- Built `llama-bench.exe` and `llama-cli.exe` with CUDA graphs, NCCL off, sm_89.
- Qwen3-0.6B Q8_0, `-ngl 99 -fa on`, 5 reps: prefill 512 = 18361 tok/s, decode 128 = 284.1 tok/s.
- Same model, 3 reps: 14 GPU layers decode 34.2 tok/s, CPU-only decode 15.5 tok/s.
- Qwen3-8B Q4_K_M, file size 5027783488, `-ngl 99 -fa on`, 3 reps: prefill 512 = 2420 tok/s, decode 128 = 50.94 tok/s.
- AdaKernel still cannot load these files.

## Not run

- Qwen3-14B.
- Nsight traces.
- A long-context 8B run. The 8B bench used llama-bench's default context.
