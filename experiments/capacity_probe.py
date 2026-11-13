"""Find the largest synthetic FP32 model that still generates tokens."""

import json
from pathlib import Path

import numpy as np

from adakernel.native import Model, Native

BASE = {
    "d_model": 2048,
    "n_heads": 16,
    "n_kv_heads": 4,
    "head_dim": 128,
    "d_ff": 5504,
    "vocab": 128,
    "max_seq": 64,
    "rms_eps": 1e-6,
    "rope_theta": 10000.0,
    "qk_norm": 1,
    "n_gpu_layers": -1,
    "chunk_rows": 64,
    "offload_overlap": 0,
    "tied_embeddings": 1,
    "init_scale": 0.01,
}


def weights_for(cfg):
    layers, d, hd = cfg["n_layers"], cfg["d_model"], cfg["head_dim"]
    n_heads, n_kv, ff, vocab = cfg["n_heads"], cfg["n_kv_heads"], cfg["d_ff"], cfg["vocab"]
    scale = np.float32(cfg["init_scale"])
    return {
        "tok_embd": np.full((vocab, d), scale, dtype=np.float32),
        "attn_norm": np.ones((layers, d), dtype=np.float32),
        "wq": np.full((layers, n_heads * hd, d), scale, dtype=np.float32),
        "wk": np.full((layers, n_kv * hd, d), scale, dtype=np.float32),
        "wv": np.full((layers, n_kv * hd, d), scale, dtype=np.float32),
        "wo": np.full((layers, d, n_heads * hd), scale, dtype=np.float32),
        "q_norm": np.ones((layers, hd), dtype=np.float32),
        "k_norm": np.ones((layers, hd), dtype=np.float32),
        "ffn_norm": np.ones((layers, d), dtype=np.float32),
        "wgate": np.full((layers, ff, d), scale, dtype=np.float32),
        "wup": np.full((layers, ff, d), scale, dtype=np.float32),
        "wdown": np.full((layers, d, ff), scale, dtype=np.float32),
        "out_norm": np.ones(d, dtype=np.float32),
        "wout": None,
    }


def main():
    native = Native()
    rows = []
    for layers in (40, 48):
        cfg = dict(BASE)
        cfg["n_layers"] = layers
        print("trying layers", layers, flush=True)
        try:
            model = Model(native, cfg, weights_for(cfg))
            result = model.generate(np.array([1, 2, 3, 4], dtype=np.int32), 4, "eager", profile=True)
            model.close()
            rows.append(
                {
                    "n_layers": layers,
                    "status": "OK",
                    "n_gpu_layers": result["n_gpu_layers"],
                    "decode_tps": result["decode_tps"],
                    "weight_vram_mb": result["weight_vram_mb"],
                    "host_weight_mb": result["host_weight_mb"],
                    "allocated_mb": result["vram_peak_mb"],
                }
            )
            print(rows[-1], flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"n_layers": layers, "status": "FAILED", "error": str(exc)})
            print(rows[-1], flush=True)
            break
    path = Path("experiments/results/capacity_probe.json")
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8")) + rows
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
