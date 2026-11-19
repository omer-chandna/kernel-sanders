"""Speculative decoding (Qwen3-0.6B drafting) against plain greedy decoding of the same target.

    python -m benchmarks.spec_bench --target models/Qwen3-4B-Q4_K_M.gguf

For every prompt and draft length k, the speculative tokens are checked against the target's own greedy tokens;
greedy speculative decoding must reproduce them exactly. tok/s counts generated tokens after the first.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from adakernel.gguf import GgufModel, tokenize

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = {
    "story": "Write a short story about a robot who learns to paint.",
    "explain": "Explain how a hash table works and when you would use one instead of a sorted array.",
    "code": "Here is a Python function that parses a CSV file and returns the average of each column:\n\n```python\n",
    "facts": "The capital of France is",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(ROOT / "models" / "Qwen3-4B-Q4_K_M.gguf"))
    ap.add_argument("--draft", default=str(ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"))
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--ks", default="2,3,4,5,6,8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    result = {"target": args.target, "draft": args.draft, "n_gen": args.n_gen, "prompts": {}}
    with GgufModel(args.target, max_seq=512) as tgt, GgufModel(args.draft, max_seq=512) as drf:
        for name, text in PROMPTS.items():
            ids = tokenize(args.target, text)
            tgt.generate(ids, 8, "cuda-graph")
            base = [tgt.generate(ids, args.n_gen, "cuda-graph") for _ in range(args.reps)]
            greedy_tps = statistics.median(r["decode_tps"] for r in base)
            ref = base[0]["tokens"].tolist()
            row = {"greedy_tps": greedy_tps, "spec": {}}
            line = f"{name:8s} greedy {greedy_tps:6.1f}"
            for k in ks:
                tgt.speculate(drf, ids, 16, k)  # warm graphs for this width
                runs = [tgt.speculate(drf, ids, args.n_gen, k) for _ in range(args.reps)]
                tps = statistics.median(r["decode_tps"] for r in runs)
                exact = all(r["tokens"].tolist() == ref for r in runs)
                row["spec"][k] = {
                    "tps": tps,
                    "speedup": tps / greedy_tps,
                    "acceptance": runs[0]["acceptance"],
                    "tokens_per_round": runs[0]["tokens_per_round"],
                    "exact": exact,
                }
                line += f" | k={k} {tps:6.1f} ({tps / greedy_tps:.2f}x, acc {runs[0]['acceptance']:.2f}{'' if exact else ', MISMATCH'})"
            print(line, flush=True)
            result["prompts"][name] = row
    best = {}
    for k in ks:
        best[k] = statistics.mean(result["prompts"][p]["spec"][k]["speedup"] for p in PROMPTS)
    result["mean_speedup_by_k"] = best
    print("mean speedup by k:", {k: round(v, 2) for k, v in best.items()})
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
