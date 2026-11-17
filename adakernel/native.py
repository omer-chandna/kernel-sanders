"""ctypes bindings for the AdaKernel CUDA runtime."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BACKENDS = {
    "eager": 0,
    "cuda-graph": 1,
    "persistent": 2,
    "hybrid": 3,
    "fused": 4,
    "cublas": 5,
    "cublas-graph": 6,
}


class DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("pci_bus_id", ctypes.c_char * 32),
        ("cc_major", ctypes.c_int32),
        ("cc_minor", ctypes.c_int32),
        ("sm_count", ctypes.c_int32),
        ("warp_size", ctypes.c_int32),
        ("max_threads_per_block", ctypes.c_int32),
        ("max_threads_per_sm", ctypes.c_int32),
        ("regs_per_sm", ctypes.c_int32),
        ("shared_mem_per_sm", ctypes.c_int32),
        ("shared_mem_per_block", ctypes.c_int32),
        ("shared_mem_per_block_optin", ctypes.c_int32),
        ("l2_cache_bytes", ctypes.c_int32),
        ("memory_clock_khz", ctypes.c_int32),
        ("memory_bus_width_bits", ctypes.c_int32),
        ("cooperative_launch", ctypes.c_int32),
        ("persistent_max_blocks", ctypes.c_int32),
        ("pci_bus", ctypes.c_int32),
        ("pci_device", ctypes.c_int32),
        ("clock_sm_khz", ctypes.c_int32),
        ("total_vram_mb", ctypes.c_float),
        ("free_vram_mb", ctypes.c_float),
        ("theoretical_bandwidth_gbps", ctypes.c_float),
    ]


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("n_layers", ctypes.c_int32),
        ("d_model", ctypes.c_int32),
        ("n_heads", ctypes.c_int32),
        ("n_kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("d_ff", ctypes.c_int32),
        ("vocab", ctypes.c_int32),
        ("max_seq", ctypes.c_int32),
        ("rms_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("qk_norm", ctypes.c_int32),
        ("n_gpu_layers", ctypes.c_int32),
        ("chunk_rows", ctypes.c_int32),
        ("offload_overlap", ctypes.c_int32),
        ("persistent_blocks", ctypes.c_int32),
        ("kv_type", ctypes.c_int32),
        ("page_tokens", ctypes.c_int32),
    ]


class Fp32Weights(ctypes.Structure):
    _fields_ = [
        ("tok_embd", ctypes.c_void_p),
        ("attn_norm", ctypes.c_void_p),
        ("wq", ctypes.c_void_p),
        ("wk", ctypes.c_void_p),
        ("wv", ctypes.c_void_p),
        ("wo", ctypes.c_void_p),
        ("q_norm", ctypes.c_void_p),
        ("k_norm", ctypes.c_void_p),
        ("ffn_norm", ctypes.c_void_p),
        ("wgate", ctypes.c_void_p),
        ("wup", ctypes.c_void_p),
        ("wdown", ctypes.c_void_p),
        ("out_norm", ctypes.c_void_p),
        ("wout", ctypes.c_void_p),
        ("tied_embeddings", ctypes.c_int32),
    ]


class GenStats(ctypes.Structure):
    _fields_ = [
        ("prefill_ms", ctypes.c_float),
        ("decode_ms", ctypes.c_float),
        ("ttft_ms", ctypes.c_float),
        ("graph_capture_ms", ctypes.c_float),
        ("prompt_tokens", ctypes.c_int32),
        ("generated_tokens", ctypes.c_int32),
        ("backend", ctypes.c_int32),
        ("launches_per_token", ctypes.c_int32),
        ("decode_tps", ctypes.c_float),
        ("prefill_tps", ctypes.c_float),
        ("vram_peak_mb", ctypes.c_float),
        ("weight_vram_mb", ctypes.c_float),
        ("host_weight_mb", ctypes.c_float),
        ("n_gpu_layers", ctypes.c_int32),
        ("persistent_blocks", ctypes.c_int32),
        ("status", ctypes.c_int32),
        ("graph_nodes", ctypes.c_int32),
    ]


class BenchRow(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 32),
        ("median_us", ctypes.c_float),
        ("min_us", ctypes.c_float),
        ("p95_us", ctypes.c_float),
        ("max_abs_err", ctypes.c_float),
        ("rel_err", ctypes.c_float),
        ("correct", ctypes.c_int32),
        ("iters", ctypes.c_int32),
    ]


class MicroReport(ctypes.Structure):
    _fields_ = [
        ("rows", BenchRow * 32),
        ("n_rows", ctypes.c_int32),
        ("sm_count", ctypes.c_int32),
        ("regs_persistent", ctypes.c_int32),
        ("persistent_blocks", ctypes.c_int32),
        ("achieved_bandwidth_gbps", ctypes.c_float),
        ("theoretical_bandwidth_gbps", ctypes.c_float),
        ("launch_eager_us", ctypes.c_float),
        ("launch_graph_us", ctypes.c_float),
        ("launch_persistent_us", ctypes.c_float),
        ("coop_supported", ctypes.c_float),
    ]


def _cuda_bins() -> list[Path]:
    bins: list[Path] = []
    env = os.environ.get("CUDA_PATH")
    if env:
        bins.append(Path(env) / "bin")
    conda = ROOT / ".conda-cuda"
    bins.extend(
        [
            conda / "Library" / "bin",
            conda / "bin",
            conda / "Library" / "lib",
            conda / "lib" / "x64",
        ]
    )
    for version in ("v13.5", "v13.4", "v13.0", "v12.8", "v12.6"):
        bins.append(Path(rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}\bin"))
    return [path for path in bins if path.exists()]


def dll_path() -> Path:
    override = os.environ.get("ADAKERNEL_DLL")
    if override:
        return Path(override)
    candidates = [
        ROOT / "build" / "adakernel.dll",
        ROOT / "build" / "Release" / "adakernel.dll",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class Native:
    def __init__(self, path: Path | None = None):
        for folder in _cuda_bins():
            os.add_dll_directory(str(folder))
        location = path or dll_path()
        if not location.exists():
            raise FileNotFoundError(
                f"adakernel.dll was not found at {location}. Build it with scripts/build.ps1."
            )
        os.add_dll_directory(str(location.parent))
        self.lib = ctypes.WinDLL(str(location))
        self._bind()

    def _bind(self) -> None:
        lib = self.lib
        lib.ak_last_error.restype = ctypes.c_char_p
        lib.ak_init.argtypes = [ctypes.c_int]
        lib.ak_device_query.argtypes = [ctypes.POINTER(DeviceInfo)]
        lib.ak_q8_quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.ak_q8_dequantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.ak_q4k_quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.ak_q4k_dequantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.ak_q6k_dequantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        for name in ("ak_q8_block_size", "ak_q4k_block_size", "ak_q6k_block_size"):
            getattr(lib, name).restype = ctypes.c_int
        f32 = ctypes.c_void_p
        lib.ak_test_rmsnorm.argtypes = [f32, f32, f32, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.ak_test_residual.argtypes = [f32, f32, f32, ctypes.c_int]
        lib.ak_test_fused_add_rmsnorm.argtypes = [f32, f32, f32, f32, f32, ctypes.c_int, ctypes.c_float]
        lib.ak_test_rope.argtypes = [f32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.ak_test_qk_norm.argtypes = [f32, f32, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.ak_test_matvec.argtypes = [f32, f32, f32, ctypes.c_int, ctypes.c_int]
        lib.ak_test_cublas_matvec.argtypes = [f32, f32, f32, ctypes.c_int, ctypes.c_int]
        lib.ak_test_silu_mul.argtypes = [f32, f32, f32, ctypes.c_int]
        lib.ak_test_attn.argtypes = [f32, f32, f32, f32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.ak_test_q8_matvec.argtypes = [f32, f32, f32, ctypes.c_int, ctypes.c_int]
        lib.ak_test_q4k_matvec.argtypes = [f32, f32, f32, ctypes.c_int, ctypes.c_int]
        lib.ak_test_qgemv_dp4a.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
        ]
        lib.ak_model_create.argtypes = [ctypes.POINTER(ModelConfig), ctypes.POINTER(Fp32Weights), ctypes.POINTER(ctypes.c_void_p)]
        lib.ak_model_destroy.argtypes = [ctypes.c_void_p]
        lib.ak_model_logits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, f32]
        lib.ak_speculate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(GenStats),
        ]
        lib.ak_speculate.restype = ctypes.c_int
        lib.ak_generate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(GenStats),
        ]
        class QwenProjReport(ctypes.Structure):
            _fields_ = [
                ("n_proj", ctypes.c_int32),
                ("n_q4", ctypes.c_int32),
                ("n_q6", ctypes.c_int32),
                ("n_q8", ctypes.c_int32),
                ("weight_gb", ctypes.c_float),
                ("old_ms", ctypes.c_float),
                ("fast_ms", ctypes.c_float),
                ("old_gbps", ctypes.c_float),
                ("fast_gbps", ctypes.c_float),
                ("q4_gbps", ctypes.c_float),
                ("q6_gbps", ctypes.c_float),
                ("q8_gbps", ctypes.c_float),
                ("quant_us", ctypes.c_float),
                ("max_rel_err", ctypes.c_float),
                ("kernel_rel_err", ctypes.c_float),
            ]

        self.QwenProjReport = QwenProjReport
        lib.ak_bench_qwen35_projections.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(QwenProjReport)]
        lib.ak_bench_qwen35_projections.restype = ctypes.c_int
        lib.ak_microbench.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(MicroReport),
        ]

    def check(self, status: int) -> None:
        if status != 0:
            raw = self.lib.ak_last_error()
            message = raw.decode("utf-8", errors="replace") if raw else f"status {status}"
            raise RuntimeError(message)

    @staticmethod
    def _ptr(array: np.ndarray | None):
        if array is None:
            return None
        if not array.flags["C_CONTIGUOUS"]:
            raise ValueError("array must be contiguous")
        return array.ctypes.data_as(ctypes.c_void_p)

    def device_query(self) -> dict:
        info = DeviceInfo()
        self.check(self.lib.ak_device_query(ctypes.byref(info)))
        return {
            "name": info.name.split(b"\0", 1)[0].decode(),
            "pci_bus_id": info.pci_bus_id.split(b"\0", 1)[0].decode(),
            "compute_capability": f"{info.cc_major}.{info.cc_minor}",
            "sm_count": info.sm_count,
            "warp_size": info.warp_size,
            "max_threads_per_block": info.max_threads_per_block,
            "max_threads_per_sm": info.max_threads_per_sm,
            "regs_per_sm": info.regs_per_sm,
            "shared_mem_per_sm": info.shared_mem_per_sm,
            "shared_mem_per_block": info.shared_mem_per_block,
            "shared_mem_per_block_optin": info.shared_mem_per_block_optin,
            "l2_cache_bytes": info.l2_cache_bytes,
            "memory_clock_khz": info.memory_clock_khz,
            "memory_bus_width_bits": info.memory_bus_width_bits,
            "cooperative_launch": info.cooperative_launch,
            "persistent_max_blocks": info.persistent_max_blocks,
            "pci_bus": info.pci_bus,
            "pci_device": info.pci_device,
            "clock_sm_khz": info.clock_sm_khz,
            "total_vram_mb": info.total_vram_mb,
            "free_vram_mb": info.free_vram_mb,
            "theoretical_bandwidth_gbps": info.theoretical_bandwidth_gbps,
        }

    def q8_quantize(self, values: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        blocks = flat.size // 32
        out = np.empty(blocks * 34, dtype=np.uint8)
        self.check(self.lib.ak_q8_quantize(self._ptr(flat), self._ptr(out), flat.size))
        return out

    def q8_dequantize(self, blocks: np.ndarray, n: int) -> np.ndarray:
        raw = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1)
        out = np.empty(n, dtype=np.float32)
        self.check(self.lib.ak_q8_dequantize(self._ptr(raw), self._ptr(out), n))
        return out

    def q4k_quantize(self, values: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        out = np.empty((flat.size // 256) * 144, dtype=np.uint8)
        self.check(self.lib.ak_q4k_quantize(self._ptr(flat), self._ptr(out), flat.size))
        return out

    def q4k_dequantize(self, blocks: np.ndarray, n: int) -> np.ndarray:
        raw = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1)
        out = np.empty(n, dtype=np.float32)
        self.check(self.lib.ak_q4k_dequantize(self._ptr(raw), self._ptr(out), n))
        return out

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        src = np.ascontiguousarray(x, dtype=np.float32)
        w = np.ascontiguousarray(weight, dtype=np.float32)
        y = np.empty_like(src)
        rows = 1 if src.ndim == 1 else src.shape[0]
        cols = src.shape[-1]
        self.check(self.lib.ak_test_rmsnorm(self._ptr(src), self._ptr(w), self._ptr(y), rows, cols, eps))
        return y

    def residual(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
        bb = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
        y = np.empty_like(aa)
        self.check(self.lib.ak_test_residual(self._ptr(aa), self._ptr(bb), self._ptr(y), aa.size))
        return y

    def fused_add_rmsnorm(self, x: np.ndarray, residual: np.ndarray, weight: np.ndarray, eps: float):
        xx = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        rr = np.ascontiguousarray(residual, dtype=np.float32).reshape(-1)
        ww = np.ascontiguousarray(weight, dtype=np.float32).reshape(-1)
        added = np.empty_like(xx)
        normed = np.empty_like(xx)
        self.check(
            self.lib.ak_test_fused_add_rmsnorm(
                self._ptr(xx), self._ptr(rr), self._ptr(ww), self._ptr(added), self._ptr(normed), xx.size, eps
            )
        )
        return added, normed

    def rope(self, q: np.ndarray, pos: int, theta: float) -> np.ndarray:
        out = np.ascontiguousarray(q, dtype=np.float32).copy()
        n_heads, head_dim = out.shape
        self.check(self.lib.ak_test_rope(self._ptr(out), n_heads, head_dim, pos, theta))
        return out

    def qk_norm(self, q: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        out = np.ascontiguousarray(q, dtype=np.float32).copy()
        w = np.ascontiguousarray(weight, dtype=np.float32)
        self.check(self.lib.ak_test_qk_norm(self._ptr(out), self._ptr(w), out.shape[0], out.shape[1], eps))
        return out

    def matvec(self, weight: np.ndarray, vector: np.ndarray) -> np.ndarray:
        w = np.ascontiguousarray(weight, dtype=np.float32)
        x = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        y = np.empty(w.shape[0], dtype=np.float32)
        self.check(self.lib.ak_test_matvec(self._ptr(w), self._ptr(x), self._ptr(y), w.shape[0], w.shape[1]))
        return y

    def cublas_matvec(self, weight: np.ndarray, vector: np.ndarray) -> np.ndarray:
        w = np.ascontiguousarray(weight, dtype=np.float32)
        x = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        y = np.empty(w.shape[0], dtype=np.float32)
        self.check(self.lib.ak_test_cublas_matvec(self._ptr(w), self._ptr(x), self._ptr(y), w.shape[0], w.shape[1]))
        return y

    def silu_mul(self, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        g = np.ascontiguousarray(gate, dtype=np.float32).reshape(-1)
        u = np.ascontiguousarray(up, dtype=np.float32).reshape(-1)
        y = np.empty_like(g)
        self.check(self.lib.ak_test_silu_mul(self._ptr(g), self._ptr(u), self._ptr(y), g.size))
        return y

    def attn(self, q, k, v) -> np.ndarray:
        qq = np.ascontiguousarray(q, dtype=np.float32)
        kk = np.ascontiguousarray(k, dtype=np.float32)
        vv = np.ascontiguousarray(v, dtype=np.float32)
        out = np.empty_like(qq)
        n_heads, head_dim = qq.shape
        seq, n_kv, _ = kk.shape
        self.check(
            self.lib.ak_test_attn(
                self._ptr(qq), self._ptr(kk), self._ptr(vv), self._ptr(out), n_heads, n_kv, head_dim, seq, seq
            )
        )
        return out

    def q8_matvec(self, packed: np.ndarray, vector: np.ndarray, rows: int, k: int) -> np.ndarray:
        raw = np.ascontiguousarray(packed, dtype=np.uint8)
        x = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        y = np.empty(rows, dtype=np.float32)
        self.check(self.lib.ak_test_q8_matvec(self._ptr(raw), self._ptr(x), self._ptr(y), rows, k))
        return y

    def qgemv_dp4a(self, qtype: str, packed: np.ndarray, vector: np.ndarray, rows: int, k: int):
        """int8-activation dp4a GEMV. Returns (y, xq, xd): the output and the device-quantized activation."""
        code = {"q8_0": 0, "q4_k": 1, "q6_k": 2}[qtype]
        raw = np.ascontiguousarray(packed, dtype=np.uint8)
        x = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        y = np.empty(rows, dtype=np.float32)
        xq = np.empty(k, dtype=np.int8)
        xd = np.empty(k // 32, dtype=np.float32)
        self.check(
            self.lib.ak_test_qgemv_dp4a(
                code, self._ptr(raw), self._ptr(x), self._ptr(y), self._ptr(xq), self._ptr(xd), rows, k
            )
        )
        return y, xq, xd

    def q4k_matvec(self, packed: np.ndarray, vector: np.ndarray, rows: int, k: int) -> np.ndarray:
        raw = np.ascontiguousarray(packed, dtype=np.uint8)
        x = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        y = np.empty(rows, dtype=np.float32)
        self.check(self.lib.ak_test_q4k_matvec(self._ptr(raw), self._ptr(x), self._ptr(y), rows, k))
        return y

    def microbench(self, rows: int, k: int, seq: int, heads: int, iters: int, warmup: int) -> dict:
        report = MicroReport()
        self.check(self.lib.ak_microbench(rows, k, seq, heads, iters, warmup, ctypes.byref(report)))
        rows_out = []
        for i in range(report.n_rows):
            row = report.rows[i]
            rows_out.append(
                {
                    "name": row.name.split(b"\0", 1)[0].decode(),
                    "median_us": row.median_us,
                    "min_us": row.min_us,
                    "p95_us": row.p95_us,
                    "max_abs_err": row.max_abs_err,
                    "rel_err": row.rel_err,
                    "correct": bool(row.correct),
                    "iters": row.iters,
                }
            )
        return {
            "rows": rows_out,
            "sm_count": report.sm_count,
            "regs_persistent": report.regs_persistent,
            "persistent_blocks": report.persistent_blocks,
            "achieved_bandwidth_gbps": report.achieved_bandwidth_gbps,
            "theoretical_bandwidth_gbps": report.theoretical_bandwidth_gbps,
            "launch_eager_us": report.launch_eager_us,
            "launch_graph_us": report.launch_graph_us,
            "launch_persistent_us": report.launch_persistent_us,
            "coop_supported": report.coop_supported,
        }

    def qwen35_projections(self, path: str, iters: int = 3) -> dict:
        report = self.QwenProjReport()
        self.check(self.lib.ak_bench_qwen35_projections(path.encode("utf-8"), int(iters), ctypes.byref(report)))
        return {name: getattr(report, name) for name, _ in report._fields_}


class Model:
    def __init__(self, native: Native, config: dict, weights: dict):
        self.native = native
        self.weights = {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in weights.items() if value is not None}
        self.handle = ctypes.c_void_p()
        cfg = ModelConfig(
            n_layers=config["n_layers"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            head_dim=config["head_dim"],
            d_ff=config["d_ff"],
            vocab=config["vocab"],
            max_seq=config["max_seq"],
            rms_eps=config["rms_eps"],
            rope_theta=config["rope_theta"],
            qk_norm=int(config.get("qk_norm", 1)),
            n_gpu_layers=int(config.get("n_gpu_layers", -1)),
            chunk_rows=int(config.get("chunk_rows", 64)),
            offload_overlap=int(config.get("offload_overlap", 0)),
            persistent_blocks=int(config.get("persistent_blocks", 0)),
            kv_type=int(config.get("kv_type", 0)),
            page_tokens=int(config.get("page_tokens", 16)),
        )
        tied = int(config.get("tied_embeddings", 0))
        packed = Fp32Weights(
            tok_embd=native._ptr(self.weights["tok_embd"]),
            attn_norm=native._ptr(self.weights["attn_norm"]),
            wq=native._ptr(self.weights["wq"]),
            wk=native._ptr(self.weights["wk"]),
            wv=native._ptr(self.weights["wv"]),
            wo=native._ptr(self.weights["wo"]),
            q_norm=native._ptr(self.weights["q_norm"]) if "q_norm" in self.weights else None,
            k_norm=native._ptr(self.weights["k_norm"]) if "k_norm" in self.weights else None,
            ffn_norm=native._ptr(self.weights["ffn_norm"]),
            wgate=native._ptr(self.weights["wgate"]),
            wup=native._ptr(self.weights["wup"]),
            wdown=native._ptr(self.weights["wdown"]),
            out_norm=native._ptr(self.weights["out_norm"]),
            wout=None if tied else native._ptr(self.weights["wout"]),
            tied_embeddings=tied,
        )
        native.check(native.lib.ak_model_create(ctypes.byref(cfg), ctypes.byref(packed), ctypes.byref(self.handle)))
        self.config = config

    def close(self) -> None:
        if self.handle:
            self.native.lib.ak_model_destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def logits(self, prompt: np.ndarray) -> np.ndarray:
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(self.config["vocab"], dtype=np.float32)
        self.native.check(self.native.lib.ak_model_logits(self.handle, Native._ptr(tokens), tokens.size, Native._ptr(out)))
        return out

    def generate(self, prompt: np.ndarray, n_gen: int, backend: str, profile: bool = True) -> dict:
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(n_gen, dtype=np.int32)
        per = np.empty(n_gen, dtype=np.float32) if profile else None
        stats = GenStats()
        code = BACKENDS[backend]
        self.native.check(
            self.native.lib.ak_generate(
                self.handle,
                Native._ptr(tokens),
                int(tokens.size),
                Native._ptr(out),
                n_gen,
                code,
                Native._ptr(per) if per is not None else None,
                n_gen if per is not None else 0,
                ctypes.byref(stats),
            )
        )
        return {
            "tokens": out,
            "per_token_ms": None if per is None else per.copy(),
            "prefill_ms": stats.prefill_ms,
            "decode_ms": stats.decode_ms,
            "ttft_ms": stats.ttft_ms,
            "graph_capture_ms": stats.graph_capture_ms,
            "decode_tps": stats.decode_tps,
            "prefill_tps": stats.prefill_tps,
            "vram_peak_mb": stats.vram_peak_mb,
            "weight_vram_mb": stats.weight_vram_mb,
            "host_weight_mb": stats.host_weight_mb,
            "n_gpu_layers": stats.n_gpu_layers,
            "persistent_blocks": stats.persistent_blocks,
            "launches_per_token": stats.launches_per_token,
            "graph_nodes": stats.graph_nodes,
            "backend": backend,
        }

    def speculate(self, draft: "Model", prompt: np.ndarray, n_gen: int, gamma: int = 4) -> dict:
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(n_gen, dtype=np.int32)
        accepts = np.empty(n_gen, dtype=np.int32)
        stats = GenStats()
        self.native.check(
            self.native.lib.ak_speculate(
                self.handle,
                draft.handle,
                Native._ptr(tokens),
                int(tokens.size),
                Native._ptr(out),
                n_gen,
                int(gamma),
                Native._ptr(accepts),
                n_gen,
                ctypes.byref(stats),
            )
        )
        return {
            "tokens": out,
            "accepts": accepts,
            "decode_ms": stats.decode_ms,
            "decode_tps": stats.decode_tps,
            "rounds": stats.launches_per_token,
            "accepted_tokens": stats.persistent_blocks,
        }
