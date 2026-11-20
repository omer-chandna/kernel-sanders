# Brag Plan: kernel-sanders (AdaKernel)

## What is this?
A from-scratch CUDA runtime for Qwen3 GGUF models on one 8 GB RTX 4060. Level with llama.cpp on plain decode, 18 to 44% faster than llama-speculative.

## The angle
No UI to film, so the product is the numbers. Every figure in the video is a real measurement from docs/RESULTS.md. Deadpan and exact, like the project's own voice.

## Hook (0 to 2.5s)
"Up to +44% faster speculative decoding, with a draft model, on one 8 GB card." Lead with the result; the 4.67 GB bandwidth wall comes right after as the reason.

## Key moments
- One speculative round: the draft proposes 3 tokens, the target checks 4 in one pass, keeps the match plus one.
- Parity with llama.cpp: 276.8 vs 274.5, 84.4 vs 86.3, 49.9 vs 48.3 tok/s.
- Speculative k=3 vs llama-speculative, eight target and prompt pairs, rows arriving one by one, then "+18% to +44%".

## Outro
"kernel-sanders. AdaKernel: a hand-written CUDA runtime for Qwen3 GGUF."

## Tone
Preset: polished. Direction: dry research film. Hard facts, no jokes.

## Format: landscape 1920x1080, 30fps. Duration 22.5s (36 beats at 96 BPM)

## Visual identity (from the README assets)
Ink #0b0f14, panel #121821, lime #a3e635 (AdaKernel), amber #f5a524 (llama.cpp and the draft), red #f87171 (rejected), text #e6edf3, muted #8b98a9. DejaVu Sans Bold and Sans Mono, embedded from the README hero.

## Storyboard (cuts on beats, 0.625s per beat)
1. Hook, beats 0-4 (0-2.5s): "up to +44%" then "faster speculative decoding, with a draft model, on one 8 GB card."
2. Speculative round, beats 4-13 (2.5-8.1s): draft chips arrive, verify pass, accept 2, reject 1, bonus token. Three captions build up.
3. Parity, beats 13-19 (8.1-11.9s): three model rows, llama.cpp vs AdaKernel, then "Never more than 3% behind."
4. Speculative results, beats 19-31 (11.9-19.4s): eight rows one by one, then "+18% to +44%".
5. Outro, beats 31-36 (19.4-22.5s): wordmark and one line.

Nothing invented: the rows are Table F and the parity table from the README. No UI or recording is faked.

## Share copy (draft)
A hand-written CUDA runtime for Qwen3 on one 8 GB RTX 4060: level with llama.cpp on plain decode and 18 to 44% faster on speculative decoding. Every number measured on the one card.
