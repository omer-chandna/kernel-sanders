"""Measure decode speed when transformer layers stay in pinned system RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adakernel.native import Model, Native
from adakernel.reference import make_weights

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "experiments" / "results" / "memory_bench.json"))
    args = parser.parse_args()
    cfg = {
        "n_layers": 4,
        "d_model": 512,
        "n_heads": 8,
        "n_kv_heads": 4,
        "head_dim": 64,
        "d_ff": 1024,
        "vocab": 128,
        "max_seq": 64,
        "rms_eps": 1e-6,
        "rope_theta": 10000.0,
        "qk_norm": 1,
        "chunk_rows": 64,
        "offload_overlap": 0,
        "tied_embeddings": 0,
        "init_scale": 0.02,
        "seed": 2,
    }
    weights = make_weights(cfg, seed=2)
    prompt = np.array([2, 4, 8, 16], dtype=np.int32)
    native = Native()
    rows = []
    reference_tokens = None
    for gpu_layers in (4, 2, 1, 0):
        local = dict(cfg)
        local["n_gpu_layers"] = gpu_layers
        model = Model(native, local, weights)
        try:
            result = model.generate(prompt, 16, "eager", profile=True)
        except RuntimeError as exc:
            rows.append({"n_gpu_layers": gpu_layers, "status": "FAILED", "error": str(exc)})
            continue
        finally:
            model.close()
        if reference_tokens is None:
            reference_tokens = result["tokens"].tolist()
        rows.append(
            {
                "n_gpu_layers": result["n_gpu_layers"],
                "status": "OK",
                "decode_tps": result["decode_tps"],
                "prefill_ms": result["prefill_ms"],
                "decode_ms": result["decode_ms"],
                "weight_vram_mb": result["weight_vram_mb"],
                "host_weight_mb": result["host_weight_mb"],
                "allocated_mb": result["vram_peak_mb"],
                "tokens_match_full_gpu": result["tokens"].tolist() == reference_tokens,
            }
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"device": native.device_query(), "rows": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
