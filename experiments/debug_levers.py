import numpy as np

from adakernel.native import Model, Native
from adakernel.reference import make_weights

CFG = {
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
    "n_gpu_layers": -1,
    "chunk_rows": 64,
    "offload_overlap": 0,
    "tied_embeddings": 0,
    "init_scale": 0.02,
    "seed": 1,
}
weights = make_weights(CFG, seed=1)
prompt = np.arange(1, 9, dtype=np.int32)
native = Native()
target = Model(native, CFG, weights)
draft = Model(native, CFG, weights)
try:
    alone = [target.generate(prompt, 16, "eager", profile=True)["decode_tps"] for _ in range(2)]
    spec = []
    for _ in range(2):
        row = target.speculate(draft, prompt, 16, gamma=4)
        spec.append((round(row["decode_tps"], 1), int(row["accepted_tokens"]), int(row["rounds"]), row["tokens"].tolist()))
    print("target eager tps", [round(v, 1) for v in alone])
    print("speculate", spec)
finally:
    target.close()
    draft.close()

off = dict(CFG)
off["n_gpu_layers"] = 0
serial = dict(off)
serial["offload_overlap"] = 0
piped = dict(off)
piped["offload_overlap"] = 1
a = Model(native, serial, weights)
b = Model(native, piped, weights)
try:
    print("offload serial", [round(a.generate(prompt, 12, "eager")["decode_tps"], 1) for _ in range(2)])
    print("offload overlap", [round(b.generate(prompt, 12, "eager")["decode_tps"], 1) for _ in range(2)])
finally:
    a.close()
    b.close()
