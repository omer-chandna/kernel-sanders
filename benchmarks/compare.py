"""Plot benchmark JSON that already exists. Missing inputs are labeled, not invented."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PLOTS = ROOT / "experiments" / "plots"


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    written = []
    kernel = RESULTS / "kernel_bench.json"
    if kernel.exists():
        data = json.loads(kernel.read_text(encoding="utf-8"))
        names = [row["name"] for row in data["rows"]]
        values = [row["median_us"] for row in data["rows"]]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(names, values)
        ax.set_ylabel("median microseconds")
        ax.set_title("Kernel latency")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        path = PLOTS / "kernel_latency.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))
    inference = RESULTS / "inference_bench.json"
    if inference.exists():
        data = json.loads(inference.read_text(encoding="utf-8"))
        for shape in data["shapes"]:
            ok = [row for row in shape["results"] if row.get("status") == "OK"]
            if not ok:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar([row["backend"] for row in ok], [row["decode_tps_median"] for row in ok])
            ax.set_ylabel("decode tokens/sec")
            ax.set_title(f"Decode throughput ({shape['shape']})")
            fig.tight_layout()
            path = PLOTS / f"decode_tps_{shape['shape']}.png"
            fig.savefig(path)
            plt.close(fig)
            written.append(str(path))
    memory = RESULTS / "memory_bench.json"
    if memory.exists():
        data = json.loads(memory.read_text(encoding="utf-8"))
        ok = [row for row in data["rows"] if row.get("status") == "OK"]
        if ok:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([row["n_gpu_layers"] for row in ok], [row["decode_tps"] for row in ok], marker="o")
            ax.set_xlabel("GPU-resident layers")
            ax.set_ylabel("decode tokens/sec")
            ax.set_title("Offload versus resident decode")
            fig.tight_layout()
            path = PLOTS / "offload_tps.png"
            fig.savefig(path)
            plt.close(fig)
            written.append(str(path))
    print("\n".join(written) if written else "no benchmark JSON found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
