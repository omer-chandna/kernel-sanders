"""Compare eager, CUDA Graph, fused, and persistent decode on the same model."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from adakernel.native import Model, Native
from adakernel.reference import make_weights

ROOT = Path(__file__).resolve().parents[1]

SHAPES = {
    "tiny": {"n_layers": 2, "d_model": 64, "n_heads": 4, "n_kv_heads": 2, "head_dim": 16, "d_ff": 128, "vocab": 48, "max_seq": 64},
    "small": {"n_layers": 4, "d_model": 256, "n_heads": 8, "n_kv_heads": 4, "head_dim": 32, "d_ff": 512, "vocab": 128, "max_seq": 128},
    "medium": {"n_layers": 4, "d_model": 1024, "n_heads": 8, "n_kv_heads": 4, "head_dim": 128, "d_ff": 2048, "vocab": 256, "max_seq": 256},
}


def _config(name: str) -> dict:
    cfg = {
        "rms_eps": 1.0e-6,
        "rope_theta": 10000.0,
        "qk_norm": 1,
        "n_gpu_layers": -1,
        "chunk_rows": 64,
        "offload_overlap": 0,
        "tied_embeddings": 0,
        "init_scale": 0.02,
        "seed": 1,
    }
    cfg.update(SHAPES[name])
    return cfg


def _summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "n": len(ordered),
    }


def run_shape(native: Native, shape: str, backends: list[str], n_gen: int, reps: int, warmup: int) -> dict:
    cfg = _config(shape)
    weights = make_weights(cfg, seed=cfg["seed"])
    prompt = np.arange(1, 17, dtype=np.int32) % cfg["vocab"]
    rows = []
    for backend in backends:
        model = Model(native, cfg, weights)
        try:
            for _ in range(warmup):
                model.generate(prompt, n_gen, backend, profile=False)
            samples = []
            launches = None
            blocks = None
            capture = None
            tokens = None
            for _ in range(reps):
                result = model.generate(prompt, n_gen, backend, profile=True)
                samples.append(result["decode_tps"])
                launches = result["launches_per_token"]
                blocks = result["persistent_blocks"]
                capture = result["graph_capture_ms"]
                tokens = result["tokens"].tolist()
        except RuntimeError as exc:
            rows.append({"backend": backend, "status": "FAILED", "error": str(exc)})
            continue
        finally:
            model.close()
        summary = _summarize(samples)
        rows.append(
            {
                "backend": backend,
                "status": "OK",
                "decode_tps_median": summary["median"],
                "decode_tps_min": summary["min"],
                "decode_tps_max": summary["max"],
                "launches_per_token": launches,
                "persistent_blocks": blocks,
                "graph_capture_ms": capture,
                "tokens": tokens,
                "reps": reps,
            }
        )
    if rows and rows[0].get("status") == "OK":
        base = rows[0]["decode_tps_median"]
        for row in rows:
            if row.get("status") == "OK" and base:
                row["speedup_vs_eager"] = row["decode_tps_median"] / base
            if row.get("status") == "OK":
                row["tokens_match_eager"] = row["tokens"] == rows[0]["tokens"]
    return {"shape": shape, "config": cfg, "n_gen": n_gen, "results": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="tiny,small")
    parser.add_argument("--backends", default="eager,cuda-graph,fused,persistent,hybrid,cublas")
    parser.add_argument("--n-gen", type=int, default=32)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default=str(ROOT / "experiments" / "results" / "inference_bench.json"))
    args = parser.parse_args()
    native = Native()
    report = {"device": native.device_query(), "shapes": []}
    for shape in args.shapes.split(","):
        print(f"running {shape}")
        report["shapes"].append(
            run_shape(native, shape, args.backends.split(","), args.n_gen, args.reps, args.warmup)
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for shape in report["shapes"]:
        print(shape["shape"])
        for row in shape["results"]:
            if row["status"] != "OK":
                print(f"  {row['backend']:<16} FAILED {row['error']}")
            else:
                print(
                    f"  {row['backend']:<16} {row['decode_tps_median']:8.2f} tok/s  "
                    f"match={row.get('tokens_match_eager')} launches={row['launches_per_token']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
