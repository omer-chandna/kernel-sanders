import numpy as np

from adakernel.native import Model, Native
from adakernel.reference import generate, make_weights

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
ref = generate(CFG, weights, PROMPT.tolist(), 6)
print("ref", ref.tolist())
native = Native()
for backend in ("eager", "cuda-graph", "persistent", "hybrid", "fused"):
    model = Model(native, CFG, weights)
    try:
        result = model.generate(PROMPT, 6, backend)
        print(backend, result["tokens"].tolist(), "launches", result["launches_per_token"])
    except Exception as exc:
        print(backend, "ERR", exc)
    finally:
        model.close()
