"""Decode throughput of the AdaKernel GGUF runtime against llama.cpp on the same file, back to back.

    python -m benchmarks.gguf_bench --model models/Qwen3-8B-Q4_K_M.gguf

AdaKernel: greedy, prompt fed one token at a time, then n_gen decoded tokens; tok/s is over the decoded tokens.
llama.cpp: llama-bench -p 0 -n n_gen -ngl 99 -fa on (decode only, same token count).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from adakernel.gguf import Detokenizer, GgufModel, llama_env, llama_exe, tokenize

ROOT = Path(__file__).resolve().parents[1]


def llama_decode_tps(model: str, n_gen: int, reps: int) -> dict | None:
    exe = llama_exe("llama-bench")
    if not exe.exists():
        return None
    proc = subprocess.run(
        [str(exe), "-m", model, "-p", "0", "-n", str(n_gen), "-ngl", "99", "-fa", "1", "-r", str(reps), "-o", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=llama_env(),
        stdin=subprocess.DEVNULL,
        timeout=900,
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        return None
    rows = json.loads(proc.stdout)
    row = rows[-1]
    return {"avg_tps": row["avg_ts"], "stddev_tps": row["stddev_ts"], "build_commit": row.get("build_commit")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3-8B-Q4_K_M.gguf"))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--backends", default="eager,cuda-graph,persistent,hybrid")
    ap.add_argument("--no-llama", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ids = tokenize(args.model, args.prompt)
    det = Detokenizer(args.model)
    result = {"model": args.model, "prompt_ids": ids, "n_gen": args.n_gen, "reps": args.reps, "adakernel": {}}
    with GgufModel(args.model, max_seq=len(ids) + args.n_gen + 8) as m:
        result["info"] = m.info
        reference = None
        for be in args.backends.split(","):
            m.generate(ids, 8, be)  # warmup, graph capture, persistent upload
            runs = [m.generate(ids, args.n_gen, be) for _ in range(args.reps)]
            tps = [r["decode_tps"] for r in runs]
            toks = runs[0]["tokens"].tolist()
            reference = reference or toks
            entry = {
                "median_tps": statistics.median(tps),
                "min_tps": min(tps),
                "max_tps": max(tps),
                "tokens_match_first_backend": toks == reference,
                "text": det(toks),
            }
            result["adakernel"][be] = entry
            print(f"adakernel {be:11s} {entry['median_tps']:7.1f} tok/s  (range {min(tps):.1f}-{max(tps):.1f})  "
                  f"match={entry['tokens_match_first_backend']}")
    if not args.no_llama:
        ll = llama_decode_tps(args.model, args.n_gen, args.reps)
        result["llama_cpp"] = ll
        if ll:
            print(f"llama.cpp   tg{args.n_gen}     {ll['avg_tps']:7.1f} tok/s  (sd {ll['stddev_tps']:.2f})")
    print("text:", ascii(result["adakernel"][args.backends.split(",")[0]]["text"][:160]))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
