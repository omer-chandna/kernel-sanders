"""Print GGUF metadata and tensor types. Does not load weights onto the GPU."""

import json
import struct
import sys
from pathlib import Path

TYPES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_K",
    11: "q3_K",
    12: "q4_K",
    13: "q5_K",
    14: "q6_K",
    15: "q8_K",
}


def read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def read_value(f, vtype):
    if vtype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:
        return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == 8:
        return read_string(f)
    if vtype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:
        return struct.unpack("<d", f.read(8))[0]
    if vtype == 9:
        et = struct.unpack("<I", f.read(4))[0]
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, et) for _ in range(n)]
    raise ValueError(f"unsupported gguf value type {vtype}")


def main():
    path = Path(sys.argv[1])
    with path.open("rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747:
            raise SystemExit(f"bad magic {magic:#x}")
        meta = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            meta[key] = read_value(f, vtype)
        tensors = []
        type_counts = {}
        for _ in range(n_tensors):
            name = read_string(f)
            (ndims,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack("<" + "Q" * ndims, f.read(8 * ndims)))
            (gtype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            label = TYPES.get(gtype, str(gtype))
            type_counts[label] = type_counts.get(label, 0) + 1
            if len(tensors) < 8 or name in ("output.weight", "token_embd.weight", "output_norm.weight"):
                tensors.append({"name": name, "dims": dims, "type": label, "offset": offset})
        info = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "version": version,
            "n_tensors": n_tensors,
            "n_kv": n_kv,
            "architecture": meta.get("general.architecture"),
            "name": meta.get("general.name"),
            "alignment": meta.get("general.alignment"),
            "type_counts": type_counts,
            "sample_tensors": tensors,
        }
        arch = info["architecture"]
        if arch:
            for key in (
                "block_count",
                "embedding_length",
                "feed_forward_length",
                "attention.head_count",
                "attention.head_count_kv",
                "attention.key_length",
                "attention.value_length",
                "context_length",
                "rope.freq_base",
                "attention.layer_norm_rms_epsilon",
            ):
                full = f"{arch}.{key}"
                if full in meta:
                    info[key] = meta[full]
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
