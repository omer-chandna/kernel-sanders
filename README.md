# AdaKernel

AdaKernel is a local inference experiment for a single consumer GPU. It compares eager CUDA launches, CUDA Graphs, fused elementwise kernels, and a persistent device-side task scheduler on the same FP32 decoder. The machine used for the measurements in `docs/RESULTS.md` is an NVIDIA GeForce RTX 4060 8 GB (Ada, sm_89), Ryzen 5 7600X, 32 GB RAM, Windows 11.

The custom runtime generates tokens for a synthetic Qwen-style decoder (RMSNorm, QK-norm, NeoX RoPE, grouped-query attention, SwiGLU). It is not a loader for Qwen GGUF files. llama.cpp is the production baseline and is built separately.

## Build

The CUDA 13.5 system installer needs administrator approval. This tree instead uses a user-local CUDA 12.8 toolkit under `.conda-cuda` (created by the conda command in `scripts/setup_windows.ps1`) and the Python 3.12 virtualenv in `.venv`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
.\.venv\Scripts\python.exe -m pytest tests -q
```

`nvcc` from CUDA 11.2 cannot target sm_89. The build script refuses a toolkit that does not contain `bin\nvcc.exe`.

## Run

```powershell
.\.venv\Scripts\python.exe -m adakernel doctor
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m benchmarks.kernel_bench
.\.venv\Scripts\python.exe -m benchmarks.inference_bench
.\.venv\Scripts\python.exe -m benchmarks.run_all --smoke
.\.venv\Scripts\python.exe -m adakernel init-model --output models\tiny.npz
.\.venv\Scripts\python.exe -m adakernel generate --model models\tiny.npz --backend persistent --prompt-tokens 1,5,9,3 -n 16
```

Backends: `eager`, `cuda-graph`, `fused`, `persistent`, `hybrid`, `cublas`, `cublas-graph`.

`--prompt` on a synthetic `.npz` model is mapped to token ids with a non-linguistic hash. It does not run a Qwen tokenizer. Pass `--prompt-tokens` when the ids matter.

## Layout

```text
adakernel/          Python package, ctypes bindings, NumPy reference
native/cuda/        kernels, persistent scheduler, model runtime
include/adakernel/  C API
benchmarks/         measurement scripts
tests/              correctness tests
experiments/        raw JSON, plots, sweep scripts
docs/               architecture, method, measured results
third_party/llama.cpp   pinned upstream checkout, not vendored into git
```

## What the measurements show

On this 4060, a persistent DAG scheduler is faster than replaying the same kernels through a CUDA Graph for single-token decode, because independent projections run at the same time. A CUDA Graph is faster than the persistent scheduler for a serial chain of empty kernels. Elementwise fusion does not move medium-model decode. The Q4_K and Q8_0 matvec kernels match their dequantized reference and are slower than FP32 cuBLAS GEMV. Full write-up: `docs/RESULTS.md`.
