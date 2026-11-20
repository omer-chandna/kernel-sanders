# README plan — kernel-sanders (AdaKernel)

**One-liner:** A from-scratch CUDA runtime that runs Qwen3 GGUF models on one 8 GB RTX 4060, matches llama.cpp on plain decode, and beats llama.cpp's speculative decoding by 18–44%.

**Style:** `clean`, with research depth (tables folded into `<details>`). No emoji: the project's voice is dry and exact.

**Video:** skipped. There is no UI to film and the runtime needs Windows + an Ada GPU, which this Mac doesn't have; a terminal recording would have to be faked. The demo is an animated chart of real measured numbers instead.

## Section order
1. Hero (animated SVG: speculative decoding round — draft proposes 3, one verify pass, keep prefix + 1), one-liner, badges, nav
2. Demo: animated bar chart, AdaKernel k=3 vs `llama-speculative`, 8 target/prompt pairs (RESULTS.md Table F)
3. Why: three sentences + Python snippet using the real `GgufModel.speculate` API
4. Feature cards (4): GGUF quant formats / parity with llama.cpp / verify-pass cost curve / every backend same tokens
5. Results: Table E (session 2) and Table F speedups, link to full RESULTS.md
6. Quickstart (Windows, from old README, verified against scripts/)
7. Usage: old Run commands + gguf_bench / spec_bench
8. How it works: Mermaid of the speculative loop + backends table (from ARCHITECTURE.md)
9. Kept from old README: hardware line, build notes (CUDA 12.8 user-local, nvcc 11.2 can't target sm_89), backends list, --prompt hashing caveat, layout tree, "What the measurements show" (in details, labeled synthetic FP32)
10. Docs links, footer strip

## Badges (evidence)
- Python 3.10+ — pyproject `requires-python = ">=3.10"`
- CUDA 12.8 — scripts/setup_windows.ps1 pins cuda-nvcc 12.8.93; build.ps1 rejects older
- GPU sm_89 — CMakeLists `CMAKE_CUDA_ARCHITECTURES 89`
- Windows — build scripts are PowerShell + MSVC (/EHsc)
- No license badge (no LICENSE file), no CI badge (no workflows), no version badge (not published)

## Visuals (.github/readme/)
hero.svg, bench.svg, card-gguf.svg, card-parity.svg, card-verify.svg, card-backends.svg, divider.svg, footer.svg, social-preview.png
Font: DejaVu Sans / Sans Mono (free license, from local matplotlib), subset and embedded as WOFF.
Palette: ink #0b0f14, panel #121821, lime #a3e635 (AdaKernel), amber #f5a524 (llama.cpp / draft), red #f87171 (rejected), text #e6edf3, muted #8b98a9.
