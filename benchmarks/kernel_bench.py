"""Microbenchmark custom kernels against cuBLAS and launch methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adakernel.native import Native

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "experiments" / "results" / "kernel_bench.json"))
    args = parser.parse_args()
    report = Native().microbench(args.rows, args.k, args.seq, args.heads, args.iters, args.warmup)
    report["shape"] = {"rows": args.rows, "k": args.k, "seq": args.seq, "heads": args.heads}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'kernel':<24} {'median us':>12} {'correct':>8} {'max abs':>12}")
    for row in report["rows"]:
        print(f"{row['name']:<24} {row['median_us']:12.3f} {str(row['correct']):>8} {row['max_abs_err']:12.3e}")
    print(
        f"bandwidth {report['achieved_bandwidth_gbps']:.1f} GB/s achieved, "
        f"{report['theoretical_bandwidth_gbps']:.1f} GB/s theoretical, "
        f"persistent regs {report['regs_persistent']}, blocks {report['persistent_blocks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
