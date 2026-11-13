"""Record llama.cpp availability. GPU numbers are measured only when the binary exists."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    binary = None
    for candidate in (
        ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "Release" / "llama-bench.exe",
        ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-bench.exe",
        ROOT / "third_party" / "llama.cpp" / "build" / "llama-bench.exe",
    ):
        if candidate.exists():
            binary = candidate
            break
    if binary is None:
        binary = shutil.which("llama-bench")
    payload = {
        "status": "NOT RUN" if binary is None else "BINARY FOUND",
        "binary": None if binary is None else str(binary),
        "note": "llama.cpp CUDA baseline has not been executed yet." if binary is None else "Binary is built. Run llama-bench separately and store JSON beside this file.",
    }
    path = ROOT / "experiments" / "results" / "llamacpp_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
