import traceback

import numpy as np

from adakernel.native import Model, Native
from adakernel.reference import make_weights

CFG = {
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "d_ff": 128,
    "vocab": 48,
    "max_seq": 32,
    "rms_eps": 1.0e-6,
    "rope_theta": 10000.0,
    "qk_norm": 1,
    "n_gpu_layers": -1,
    "chunk_rows": 16,
    "offload_overlap": 0,
    "tied_embeddings": 0,
    "init_scale": 0.02,
    "seed": 0,
}
PROMPT = np.array([1, 5, 9, 3], dtype=np.int32)
weights = make_weights(CFG, seed=0)
native = Native()
for kv_type in (0, 1, 2):
    cfg = dict(CFG)
    cfg["kv_type"] = kv_type
    print("create", kv_type, flush=True)
    try:
        model = Model(native, cfg, weights)
    except Exception:
        traceback.print_exc()
        continue
    try:
        print("gen", kv_type, model.generate(PROMPT, 2, "eager")["tokens"].tolist(), flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        model.close()
