"""Run the experiment suite. A failed independent experiment is recorded and does not stop the rest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run(name: str, args: list[str], output: Path) -> dict:
    log = output / f"{name}.log"
    print(f"== {name} ==")
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, text=True, capture_output=True)
    log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    status = "OK" if proc.returncode == 0 else "FAILED"
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
    return {"name": name, "status": status, "returncode": proc.returncode, "log": str(log)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiments.yaml"))
    parser.add_argument("--output", default=str(ROOT / "experiments" / "results"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    smoke = args.smoke or bool(config.get("smoke"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shapes = "tiny" if smoke else config.get("shapes", "tiny,small")
    iters = 5 if smoke else int(config.get("kernel_iters", 20))
    reps = 2 if smoke else int(config.get("inference_reps", 5))
    n_gen = 8 if smoke else int(config.get("n_gen", 32))
    steps = [
        ("doctor", ["-m", "adakernel", "doctor"]),
        ("pytest", ["-m", "pytest", "tests", "-q"]),
        ("kernel_bench", ["-m", "benchmarks.kernel_bench", "--iters", str(iters), "--warmup", "2", "--output", str(output / "kernel_bench.json")]),
        ("inference_bench", ["-m", "benchmarks.inference_bench", "--shapes", shapes, "--reps", str(reps), "--n-gen", str(n_gen), "--output", str(output / "inference_bench.json")]),
        ("memory_bench", ["-m", "benchmarks.memory_bench", "--output", str(output / "memory_bench.json")]),
        ("baseline", ["-m", "benchmarks.baseline"]),
        ("plots", ["-m", "benchmarks.compare"]),
    ]
    summary = []
    for name, cmd in steps:
        try:
            summary.append(_run(name, cmd, output))
        except Exception as exc:  # noqa: BLE001
            summary.append({"name": name, "status": "FAILED", "error": str(exc), "trace": traceback.format_exc()})
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# AdaKernel experiment summary", ""]
    for row in summary:
        lines.append(f"- {row['name']}: {row['status']}")
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all(row["status"] == "OK" for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
