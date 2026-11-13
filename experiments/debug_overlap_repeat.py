import numpy as np

from adakernel.native import Model, Native
from adakernel.reference import make_weights

cfg = {
    "n_layers": 4,
    "d_model": 256,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 64,
    "d_ff": 512,
    "vocab": 64,
    "max_seq": 64,
    "rms_eps": 1e-6,
    "rope_theta": 10000.0,
    "qk_norm": 1,
    "n_gpu_layers": 0,
    "chunk_rows": 64,
    "tied_embeddings": 0,
    "init_scale": 0.02,
    "seed": 1,
}
weights = make_weights(cfg, seed=1)
prompt = np.arange(1, 9, dtype=np.int32)
native = Native()
for flag in (0, 1):
    local = dict(cfg)
    local["offload_overlap"] = flag
    model = Model(native, local, weights)
    try:
        samples = [round(model.generate(prompt, 12, "eager")["decode_tps"], 1) for _ in range(4)]
    finally:
        model.close()
    print("overlap" if flag else "serial", samples)
