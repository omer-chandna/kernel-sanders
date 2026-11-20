"""AdaKernel command line.

Examples:
  python -m adakernel doctor
  python -m adakernel init-model --output models/tiny.npz
  python -m adakernel generate --model models/tiny.npz --backend persistent --prompt-tokens 1,2,3
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from adakernel.native import Native, Model
from adakernel.reference import generate, make_weights

ROOT = Path(__file__).resolve().parents[1]


TINY = {
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "d_ff": 128,
    "vocab": 48,
    "max_seq": 64,
    "rms_eps": 1.0e-6,
    "rope_theta": 10000.0,
    "qk_norm": 1,
    "n_gpu_layers": -1,
    "chunk_rows": 32,
    "offload_overlap": 0,
    "tied_embeddings": 0,
    "init_scale": 0.02,
    "seed": 0,
}


def _nvidia_smi() -> dict:
    query = "name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw,pcie.link.gen.current,pcie.link.width.current"
    try:
        text = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}
    keys = ["name", "driver_version", "memory_total_mb", "memory_used_mb", "utilization_gpu", "temperature_c", "sm_clock_mhz", "mem_clock_mhz", "power_w", "pcie_gen", "pcie_width"]
    values = [part.strip() for part in text.split(",")]
    return dict(zip(keys, values))


def _repo_path(path: str | Path) -> str:
    """Path relative to the repo root, or just the file name if it lives elsewhere.

    configs/hardware.yaml is committed, so it must not record the user's home directory.
    """
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return p.name


def cmd_doctor(_: argparse.Namespace) -> int:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        native = Native()
        info["cuda_device"] = native.device_query()
        info["dll"] = _repo_path(native.lib._name)
    except Exception as exc:  # noqa: BLE001 - doctor reports the failure instead of crashing.
        info["cuda_device"] = {"error": str(exc)}
    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardware.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs" / "hardware.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0 if "error" not in info.get("cuda_device", {}) else 1


def _load_npz(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data["config_json"]))
    weights = {key: data[key] for key in data.files if key != "config_json"}
    if config.get("tied_embeddings", 0):
        weights["wout"] = None
    return config, weights


def cmd_init_model(args: argparse.Namespace) -> int:
    config = dict(TINY)
    if args.config:
        config.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    weights = make_weights(config, seed=config.get("seed", 0))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in weights.items() if value is not None}
    payload["config_json"] = np.array(json.dumps(config))
    np.savez(path, **payload)
    print(f"wrote {path}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    if args.model:
        config, weights = _load_npz(Path(args.model))
    else:
        config, weights = dict(TINY), make_weights(TINY, seed=TINY["seed"])
    if args.n_gpu_layers is not None:
        config["n_gpu_layers"] = args.n_gpu_layers
    if args.prompt_tokens:
        prompt = [int(part) for part in args.prompt_tokens.split(",") if part != ""]
    else:
        prompt = _prompt_to_tokens(args.prompt, config["vocab"])
    native = Native()
    model = Model(native, config, weights)
    try:
        result = model.generate(np.asarray(prompt, dtype=np.int32), args.n, args.backend, profile=True)
    finally:
        model.close()
    print(json.dumps({key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in result.items()}, indent=2))
    return 0


def _prompt_to_tokens(prompt: str, vocab: int) -> list[int]:
    """Deterministic stand-in tokenizer for synthetic models.

    Real GGUF prompts are tokenized by llama.cpp's tokenizer. This mapping only
    gives the synthetic demo a stable token sequence.
    """
    tokens = [1]
    for piece in prompt.split():
        tokens.append(1 + (sum(piece.encode("utf-8")) % (vocab - 1)))
    return tokens[:32]


def main() -> int:
    parser = argparse.ArgumentParser(prog="adakernel")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    init = sub.add_parser("init-model")
    init.add_argument("--output", required=True)
    init.add_argument("--config")
    gen = sub.add_parser("generate")
    gen.add_argument("--model")
    gen.add_argument("--backend", default="eager", choices=["eager", "cuda-graph", "persistent", "hybrid", "fused", "cublas", "cublas-graph"])
    gen.add_argument("--prompt", default="Explain how CUDA Graphs work.")
    gen.add_argument("--prompt-tokens")
    gen.add_argument("-n", type=int, default=16)
    gen.add_argument("--n-gpu-layers", type=int)
    args = parser.parse_args()
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "init-model":
        return cmd_init_model(args)
    if args.cmd == "generate":
        return cmd_generate(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
