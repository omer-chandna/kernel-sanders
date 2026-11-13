"""Sweep persistent chunk size and resident worker count on the medium model."""

import json
from pathlib import Path

import numpy as np

from adakernel.native import Model, Native
from adakernel.reference import make_weights

CFG = {
    "n_layers": 4,
    "d_model": 1024,
    "n_heads": 8,
    "n_kv_heads": 4,
    "head_dim": 128,
    "d_ff": 2048,
    "vocab": 256,
    "max_seq": 128,
    "rms_eps": 1e-6,
    "rope_theta": 10000.0,
    "qk_norm": 1,
    "n_gpu_layers": -1,
    "offload_overlap": 0,
    "tied_embeddings": 0,
    "init_scale": 0.02,
    "seed": 1,
}


def measure(native, weights, prompt, **overrides):
    cfg = dict(CFG)
    cfg.update(overrides)
    model = Model(native, cfg, weights)
    try:
        model.generate(prompt, 16, "persistent", profile=False)
        samples = [model.generate(prompt, 16, "persistent", profile=True)["decode_tps"] for _ in range(3)]
        blocks = model.generate(prompt, 8, "persistent", profile=True)["persistent_blocks"]
    finally:
        model.close()
    samples.sort()
    return {"decode_tps_median": samples[1], "samples": samples, "persistent_blocks": blocks, **overrides}


def main():
    native = Native()
    weights = make_weights(CFG, seed=1)
    prompt = np.arange(1, 9, dtype=np.int32)
    rows = []
    for chunk in (16, 32, 64, 128, 256, 512, 1024):
        row = measure(native, weights, prompt, chunk_rows=chunk, persistent_blocks=0)
        print("chunk", row)
        rows.append(row)
    best_chunk = max(rows, key=lambda item: item["decode_tps_median"])["chunk_rows"]
    for blocks in (8, 16, 24, 48, 72, 96):
        row = measure(native, weights, prompt, chunk_rows=best_chunk, persistent_blocks=blocks)
        print("blocks", row)
        rows.append(row)
    path = Path("experiments/results/scheduler_sweep.json")
    path.write_text(json.dumps({"best_chunk_candidate": best_chunk, "rows": rows}, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
