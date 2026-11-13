import numpy as np

from adakernel.native import Model, Native
from adakernel.reference import make_weights

cfg = {
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
    "chunk_rows": 64,
    "offload_overlap": 0,
    "tied_embeddings": 0,
    "init_scale": 0.02,
    "seed": 1,
}
weights = make_weights(cfg, seed=1)
prompt = np.arange(1, 17, dtype=np.int32)
native = Native()
for backend in ("eager", "persistent", "hybrid", "cublas"):
    model = Model(native, cfg, weights)
    try:
        model.generate(prompt, 8, backend, profile=False)
        samples = [model.generate(prompt, 16, backend, profile=True)["decode_tps"] for _ in range(3)]
        print(backend, [round(v, 1) for v in samples])
    except Exception as exc:
        print(backend, "ERR", exc)
    finally:
        model.close()
