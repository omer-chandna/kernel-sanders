"""NumPy reference for the AdaKernel decoder. The math matches the CUDA schedule."""

from __future__ import annotations

import numpy as np


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    xx = x.astype(np.float32, copy=False)
    var = np.sum(xx * xx, axis=-1, keepdims=True) / xx.shape[-1]
    return (xx * np.reciprocal(np.sqrt(var + eps)) * weight).astype(np.float32)


def rope_neox(q: np.ndarray, pos: int, theta: float) -> np.ndarray:
    heads, head_dim = q.shape
    half = head_dim // 2
    index = np.arange(half, dtype=np.float32)
    freq = np.exp(-np.log(np.float32(theta)) * (2.0 * index / np.float32(head_dim)))
    ang = np.float32(pos) * freq
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    x1 = q[:, :half]
    x2 = q[:, half:]
    out = np.empty_like(q)
    out[:, :half] = x1 * cos - x2 * sin
    out[:, half:] = x2 * cos + x1 * sin
    return out


def silu_mul(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-gate))
    return (gate * sig * up).astype(np.float32)


def attention(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, seq: int, n_rep: int) -> np.ndarray:
    head_dim = q.shape[-1]
    scale = np.float32(1.0 / np.sqrt(head_dim))
    out = np.empty_like(q)
    for head in range(q.shape[0]):
        kv = head // n_rep
        k = k_cache[:seq, kv]
        v = v_cache[:seq, kv]
        scores = k @ q[head] * scale
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / np.sum(weights)
        out[head] = weights @ v
    return out.astype(np.float32)


def make_weights(config: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    layers = config["n_layers"]
    d = config["d_model"]
    hd = config["head_dim"]
    n_heads = config["n_heads"]
    n_kv = config["n_kv_heads"]
    ff = config["d_ff"]
    vocab = config["vocab"]
    scale = np.float32(config.get("init_scale", 0.02))

    def matrix(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * scale)

    weights = {
        "tok_embd": matrix(vocab, d),
        "attn_norm": np.ones((layers, d), dtype=np.float32),
        "wq": matrix(layers, n_heads * hd, d),
        "wk": matrix(layers, n_kv * hd, d),
        "wv": matrix(layers, n_kv * hd, d),
        "wo": matrix(layers, d, n_heads * hd),
        "q_norm": np.ones((layers, hd), dtype=np.float32),
        "k_norm": np.ones((layers, hd), dtype=np.float32),
        "ffn_norm": np.ones((layers, d), dtype=np.float32),
        "wgate": matrix(layers, ff, d),
        "wup": matrix(layers, ff, d),
        "wdown": matrix(layers, d, ff),
        "out_norm": np.ones(d, dtype=np.float32),
    }
    if config.get("tied_embeddings", 0):
        weights["wout"] = None
    else:
        weights["wout"] = matrix(vocab, d)
    return weights


def forward_logits(config: dict, weights: dict, token: int, pos: int, cache: list[dict]) -> np.ndarray:
    d = config["d_model"]
    hd = config["head_dim"]
    n_heads = config["n_heads"]
    n_kv = config["n_kv_heads"]
    n_rep = n_heads // n_kv
    eps = config["rms_eps"]
    theta = config["rope_theta"]
    x = weights["tok_embd"][token].copy()
    for layer in range(config["n_layers"]):
        h = rmsnorm(x, weights["attn_norm"][layer], eps)
        q = (weights["wq"][layer] @ h).reshape(n_heads, hd)
        k = (weights["wk"][layer] @ h).reshape(n_kv, hd)
        v = (weights["wv"][layer] @ h).reshape(n_kv, hd)
        if config.get("qk_norm", 1):
            q = rmsnorm(q, weights["q_norm"][layer], eps)
            k = rmsnorm(k, weights["k_norm"][layer], eps)
        q = rope_neox(q, pos, theta)
        k = rope_neox(k, pos, theta)
        cache[layer]["k"][pos] = k
        cache[layer]["v"][pos] = v
        attn = attention(q, cache[layer]["k"], cache[layer]["v"], pos + 1, n_rep).reshape(-1)
        x = x + weights["wo"][layer] @ attn
        h = rmsnorm(x, weights["ffn_norm"][layer], eps)
        gate = weights["wgate"][layer] @ h
        up = weights["wup"][layer] @ h
        hidden = silu_mul(gate, up)
        x = x + weights["wdown"][layer] @ hidden
    h = rmsnorm(x, weights["out_norm"], eps)
    output = weights["tok_embd"] if config.get("tied_embeddings", 0) else weights["wout"]
    return output @ h


def generate(config: dict, weights: dict, prompt: list[int], n_gen: int) -> np.ndarray:
    cache = []
    for _ in range(config["n_layers"]):
        cache.append(
            {
                "k": np.zeros((config["max_seq"], config["n_kv_heads"], config["head_dim"]), dtype=np.float32),
                "v": np.zeros((config["max_seq"], config["n_kv_heads"], config["head_dim"]), dtype=np.float32),
            }
        )
    logits = None
    for pos, token in enumerate(prompt):
        logits = forward_logits(config, weights, int(token), pos, cache)
    out = []
    pos = len(prompt)
    for _ in range(n_gen):
        token = int(np.argmax(logits))
        out.append(token)
        logits = forward_logits(config, weights, token, pos, cache)
        pos += 1
    return np.asarray(out, dtype=np.int32)


def prefill_logits(config: dict, weights: dict, prompt: list[int]) -> np.ndarray:
    cache = []
    for _ in range(config["n_layers"]):
        cache.append(
            {
                "k": np.zeros((config["max_seq"], config["n_kv_heads"], config["head_dim"]), dtype=np.float32),
                "v": np.zeros((config["max_seq"], config["n_kv_heads"], config["head_dim"]), dtype=np.float32),
            }
        )
    logits = None
    for pos, token in enumerate(prompt):
        logits = forward_logits(config, weights, int(token), pos, cache)
    return logits
