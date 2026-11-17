"""Qwen3 GGUF runtime: ctypes binding for native/cuda/gguf_model.cu, plus tokenizer helpers.

Tokenizing uses llama.cpp's own `llama-tokenize` so the ids match llama.cpp exactly. Detokenizing reads the
vocabulary straight from the GGUF file and undoes the GPT-2 byte-to-unicode mapping.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import struct
import subprocess
from pathlib import Path

import numpy as np

from adakernel.native import BACKENDS, GenStats, Native

ROOT = Path(__file__).resolve().parents[1]
LLAMA_BIN = ROOT / "third_party" / "llama.cpp" / "build" / "bin"


class GgufInfo(ctypes.Structure):
    _fields_ = [
        ("arch", ctypes.c_char * 32),
        ("n_layers", ctypes.c_int32),
        ("d_model", ctypes.c_int32),
        ("n_heads", ctypes.c_int32),
        ("n_kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("d_ff", ctypes.c_int32),
        ("vocab", ctypes.c_int32),
        ("max_seq", ctypes.c_int32),
        ("tied", ctypes.c_int32),
        ("eager_tasks", ctypes.c_int32),
        ("persistent_tasks", ctypes.c_int32),
        ("rope_theta", ctypes.c_float),
        ("rms_eps", ctypes.c_float),
        ("weight_gb_per_token", ctypes.c_float),
        ("vram_mb", ctypes.c_float),
    ]


def _bind(lib) -> None:
    if getattr(lib, "_gguf_bound", False):
        return
    vp = ctypes.c_void_p
    lib.ak_gguf_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(vp), ctypes.POINTER(GgufInfo)]
    lib.ak_gguf_load.restype = ctypes.c_int
    lib.ak_gguf_destroy.argtypes = [vp]
    lib.ak_gguf_destroy.restype = ctypes.c_int
    lib.ak_gguf_generate.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(GenStats)]
    lib.ak_gguf_generate.restype = ctypes.c_int
    lib.ak_gguf_logits.argtypes = [vp, vp, ctypes.c_int, ctypes.c_int, vp]
    lib.ak_gguf_logits.restype = ctypes.c_int
    lib.ak_gguf_profile.argtypes = [vp, vp, vp]
    lib.ak_gguf_profile.restype = ctypes.c_int
    lib._gguf_bound = True


class GgufModel:
    def __init__(self, path: str | Path, max_seq: int = 1024, chunk_rows: int = 128, native: Native | None = None):
        self.native = native or Native()
        _bind(self.native.lib)
        self.path = Path(path)
        handle = ctypes.c_void_p()
        info = GgufInfo()
        self.native.check(
            self.native.lib.ak_gguf_load(str(self.path).encode("utf-8"), max_seq, chunk_rows, ctypes.byref(handle), ctypes.byref(info))
        )
        self.handle = handle
        self.info = {name: getattr(info, name) for name, _ in GgufInfo._fields_}
        self.info["arch"] = info.arch.decode()

    def close(self) -> None:
        if self.handle:
            self.native.lib.ak_gguf_destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def generate(self, prompt, n_gen: int, backend: str = "hybrid") -> dict:
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(n_gen, dtype=np.int32)
        stats = GenStats()
        self.native.check(
            self.native.lib.ak_gguf_generate(
                self.handle, Native._ptr(tokens), int(tokens.size), Native._ptr(out), int(n_gen), BACKENDS[backend], ctypes.byref(stats)
            )
        )
        return {
            "tokens": out,
            "backend": backend,
            "prompt_ms": stats.prefill_ms,
            "decode_ms": stats.decode_ms,
            "decode_tps": stats.decode_tps,
            "launches_per_token": stats.launches_per_token,
            "graph_capture_ms": stats.graph_capture_ms,
            "persistent_blocks": stats.persistent_blocks,
            "vram_mb": stats.vram_peak_mb,
        }

    OP_NAMES = {
        1: "rmsnorm", 2: "residual", 4: "rope", 6: "silu_mul", 7: "attn_head", 10: "embed", 12: "qk_norm",
        13: "kv_write", 14: "finalize/argmax", 16: "quantize_x", 17: "gemv_q8_0", 18: "gemv_q4_k", 19: "gemv_q6_k",
        20: "embed_q",
        21: "add_norm_quant", 22: "silu_quant", 23: "qk_norm_rope", 24: "attention", 25: "argmax_part",
    }

    def profile(self) -> dict:
        """GPU ms per op for one eager decode step. Call after generate()."""
        ms = np.zeros(32, dtype=np.float32)
        cnt = np.zeros(32, dtype=np.int32)
        self.native.check(self.native.lib.ak_gguf_profile(self.handle, Native._ptr(ms), Native._ptr(cnt)))
        return {self.OP_NAMES.get(i, str(i)): (float(ms[i]), int(cnt[i])) for i in range(32) if cnt[i]}

    def logits(self, prompt, backend: str = "eager") -> np.ndarray:
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(self.info["vocab"], dtype=np.float32)
        self.native.check(
            self.native.lib.ak_gguf_logits(self.handle, Native._ptr(tokens), int(tokens.size), BACKENDS[backend], Native._ptr(out))
        )
        return out


# ---------------------------------------------------------------- tokenizer helpers


def llama_env() -> dict:
    """Environment in which llama.cpp's CUDA binaries find cudart / cublas from the local toolkit."""
    env = dict(os.environ)
    extra = [ROOT / ".conda-cuda" / "Library" / "bin", ROOT / ".conda-cuda" / "Library" / "bin" / "x64"]
    env["PATH"] = os.pathsep.join([str(p) for p in extra if p.exists()] + [env.get("PATH", "")])
    return env


def llama_exe(name: str) -> Path:
    exe = LLAMA_BIN / f"{name}.exe"
    return exe if exe.exists() else LLAMA_BIN / name


def tokenize(model_path: str | Path, text: str) -> list[int]:
    """Token ids from llama.cpp's tokenizer, with whatever BOS policy the GGUF declares."""
    proc = subprocess.run(
        [str(llama_exe("llama-tokenize")), "-m", str(model_path), "-p", text, "--ids", "--log-disable"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        env=llama_env(),
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("[")]
    if not lines:
        raise RuntimeError(f"llama-tokenize printed no id list:\n{proc.stdout}\n{proc.stderr}")
    return [int(v) for v in json.loads(lines[-1])]


def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def read_vocab(model_path: str | Path) -> list[str]:
    """tokenizer.ggml.tokens from the GGUF header, without touching tensor data."""
    with open(model_path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747:
            raise ValueError("not a GGUF file")

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t in sizes:
                f.seek(sizes[t], 1)
            elif t == 8:
                rstr()
            elif t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et in sizes:
                    f.seek(sizes[et] * n, 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                raise ValueError(f"unknown gguf value type {t}")

        for _ in range(n_kv):
            key = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            if key == "tokenizer.ggml.tokens" and t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                return [rstr() for _ in range(n)]
            skip(t)
    raise ValueError("tokenizer.ggml.tokens not found")


class Detokenizer:
    def __init__(self, model_path: str | Path):
        self.vocab = read_vocab(model_path)
        self.bd = _byte_decoder()

    def __call__(self, ids) -> str:
        out = bytearray()
        for i in ids:
            tok = self.vocab[int(i)]
            if re.fullmatch(r"<\|[^|]*\|>", tok):
                out += tok.encode("utf-8")
                continue
            for ch in tok:
                b = self.bd.get(ch)
                if b is None:
                    out += ch.encode("utf-8")
                else:
                    out.append(b)
        return out.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- NumPy reference from the same file


def read_header(model_path: str | Path):
    """(scalar metadata, {name: (dims, type, offset)}, data_offset). Arrays in metadata are skipped."""
    sizes = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
    meta: dict = {}
    tensors: dict = {}
    with open(model_path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")

        def val(t):
            if t in sizes:
                fmt = sizes[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if t == 8:
                return rstr()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et in sizes:
                    f.seek(struct.calcsize(sizes[et]) * n, 1)
                else:
                    for _ in range(n):
                        val(et)
                return None
            raise ValueError(f"unknown gguf value type {t}")

        for _ in range(n_kv):
            key = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            v = val(t)
            if v is not None:
                meta[key] = v
        for _ in range(n_tensors):
            name = rstr()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            gtype, offset = struct.unpack("<IQ", f.read(12))
            tensors[name] = (dims, gtype, offset)
        align = int(meta.get("general.alignment", 32))
        data_offset = (f.tell() + align - 1) // align * align
    return meta, tensors, data_offset


def reference_weights(model_path: str | Path, native: Native | None = None):
    """Dequantize a Qwen3 GGUF to FP32 in the layout adakernel.reference.forward_logits expects."""
    native = native or Native()
    meta, tensors, base = read_header(model_path)
    a = meta["general.architecture"]
    cfg = {
        "n_layers": int(meta[f"{a}.block_count"]),
        "d_model": int(meta[f"{a}.embedding_length"]),
        "n_heads": int(meta[f"{a}.attention.head_count"]),
        "n_kv_heads": int(meta[f"{a}.attention.head_count_kv"]),
        "head_dim": int(meta[f"{a}.attention.key_length"]),
        "d_ff": int(meta[f"{a}.feed_forward_length"]),
        "rms_eps": float(meta[f"{a}.attention.layer_norm_rms_epsilon"]),
        "rope_theta": float(meta[f"{a}.rope.freq_base"]),
        "qk_norm": 1,
        "tied_embeddings": 0 if "output.weight" in tensors else 1,
    }
    cfg["vocab"] = int(tensors["token_embd.weight"][0][1])
    block = {8: (32, 34, "ak_q8_dequantize"), 12: (256, 144, "ak_q4k_dequantize"), 14: (256, 210, "ak_q6k_dequantize")}

    with open(model_path, "rb") as f:

        def load(name):
            dims, gtype, off = tensors[name]
            n = int(np.prod(dims))
            f.seek(base + off)
            if gtype == 0:
                arr = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
            else:
                be, bb, fn = block[gtype]
                raw = np.frombuffer(f.read(n // be * bb), dtype=np.uint8).copy()
                arr = np.empty(n, dtype=np.float32)
                native.check(getattr(native.lib, fn)(Native._ptr(raw), Native._ptr(arr), n))
            return arr.reshape(tuple(reversed(dims))) if len(dims) > 1 else arr

        L = cfg["n_layers"]

        def stack(suffix):
            return np.stack([load(f"blk.{i}.{suffix}") for i in range(L)])

        w = {
            "tok_embd": load("token_embd.weight"),
            "out_norm": load("output_norm.weight"),
            "attn_norm": stack("attn_norm.weight"),
            "ffn_norm": stack("ffn_norm.weight"),
            "q_norm": stack("attn_q_norm.weight"),
            "k_norm": stack("attn_k_norm.weight"),
            "wq": stack("attn_q.weight"),
            "wk": stack("attn_k.weight"),
            "wv": stack("attn_v.weight"),
            "wo": stack("attn_output.weight"),
            "wgate": stack("ffn_gate.weight"),
            "wup": stack("ffn_up.weight"),
            "wdown": stack("ffn_down.weight"),
            "wout": None if cfg["tied_embeddings"] else load("output.weight"),
        }
    return cfg, w
