from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 0x50495443          # PITCH.defaultSeed, ASCII "PITC"
SUPPORTED_BITS = range(2, 9)       # 2...8
SUPPORTED_DIMS = [2 ** k for k in range(1, 11)]   # 2...1024

def pcg_hash(v):
    v = np.asarray(v, dtype=np.uint64)
    state = (v * 747796405 + 2891336453) & 0xFFFFFFFF
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & 0xFFFFFFFF
    return ((word >> 22) ^ word) & 0xFFFFFFFF

def rand_signs(seed: int, d: int) -> np.ndarray:
    h = pcg_hash((np.uint64(seed) + np.arange(d, dtype=np.uint64)) & 0xFFFFFFFF)
    return np.where(h & 1, 1.0, -1.0)

def wht(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float64, copy=True)
    n, d = x.shape
    h = 1
    while h < d:
        x = x.reshape(n, d // (2 * h), 2, h)
        a = x[:, :, 0, :].copy(); b = x[:, :, 1, :].copy()
        x[:, :, 0, :] = a + b; x[:, :, 1, :] = a - b
        x = x.reshape(n, d)
        h *= 2
    return x

def rotate(X, seed):   return wht(X * rand_signs(seed, X.shape[1])) / math.sqrt(X.shape[1])
def unrotate(Y, seed): return wht(Y) * rand_signs(seed, Y.shape[1]) / math.sqrt(Y.shape[1])

def lloyd_max_density(grid, weights, k, iters=3000, tol=1e-13):
    w = weights / weights.sum()
    pd = np.cumsum(w ** (1 / 3)); pd /= pd[-1]
    c = np.interp((np.arange(k) + 0.5) / k, pd, grid)
    for _ in range(iters):
        idx = np.searchsorted((c[1:] + c[:-1]) / 2, grid)
        mass = np.bincount(idx, weights=w, minlength=k)
        mom = np.bincount(idx, weights=w * grid, minlength=k)
        new = np.where(mass > 0, mom / np.maximum(mass, 1e-300), c)
        if np.max(np.abs(new - c)) < tol:
            c = new; break
        c = new
    return c

def compute_sphere_codebook(d: int, b: int) -> np.ndarray:
    x = np.linspace(-1, 1, 200001)[1:-1]
    c = lloyd_max_density(x, (1 - x * x) ** ((d - 3) / 2), 2 ** b)
    return (c - c[::-1]) / 2

_codebooks: dict | None = None
def turbo_codebook(d: int, b: int) -> np.ndarray:
    global _codebooks
    if _codebooks is None:
        path = HERE / "codebooks.json"
        if not path.exists():
            raise FileNotFoundError("codebooks.json missing: run gen_fixtures.py codebooks first")
        _codebooks = json.loads(path.read_text())
    return np.array(_codebooks[f"{d}_{b}"], dtype=np.float32).astype(np.float64)

def code_stride_bytes(d: int, b: int) -> int:
    return ((d * b + 7) // 8 + 3) & ~3

def pack_codes(codes: np.ndarray, b: int) -> bytes:
    n, d = codes.shape
    words_per_vec = code_stride_bytes(d, b) // 4
    out = np.zeros((n, words_per_vec), dtype=np.uint64)
    for i in range(d):
        start = i * b; w = start >> 5; off = start & 31
        val = codes[:, i].astype(np.uint64)
        out[:, w] |= (val << np.uint64(off)) & 0xFFFFFFFF
        if off + b > 32:
            out[:, w + 1] |= val >> np.uint64(32 - off)
    return out.astype("<u4").tobytes()

def unpack_codes(buf: bytes, n: int, d: int, b: int) -> np.ndarray:
    words = np.frombuffer(buf, dtype="<u4").astype(np.uint64).reshape(n, -1)
    codes = np.zeros((n, d), dtype=np.int64)
    mask = (1 << b) - 1
    for i in range(d):
        start = i * b; w = start >> 5; off = start & 31
        v = words[:, w] >> np.uint64(off)
        if off + b > 32:
            v |= words[:, w + 1] << np.uint64(32 - off)
        codes[:, i] = (v & np.uint64(mask)).astype(np.int64)
    return codes

def turbo_encode(X: np.ndarray, b: int, seed: int = DEFAULT_SEED):
    """Returns (codes (n,d) int, norms (n,) float)."""
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    norms = np.linalg.norm(X, axis=1)
    safe = np.where(norms > 0, norms, 1.0)[:, None]
    U = rotate(X / safe, seed)
    cb32 = turbo_codebook(d, b).astype(np.float32)
    edges = (np.float32(0.5) * (cb32[1:] + cb32[:-1])).astype(np.float64)
    codes = np.searchsorted(edges, U, side="left")
    return codes, norms

def turbo_decode(codes: np.ndarray, norms: np.ndarray, b: int, seed: int = DEFAULT_SEED):
    d = codes.shape[1]
    U = turbo_codebook(d, b)[codes]
    return unrotate(U, seed) * np.asarray(norms, dtype=np.float64)[:, None]

def polar_encode(X: np.ndarray, b: int, seed: int = DEFAULT_SEED):
    Y = rotate(np.asarray(X, dtype=np.float64), seed)
    a, c = Y[:, 0::2], Y[:, 1::2]
    r = np.hypot(a, c)
    theta = np.mod(np.arctan2(c, a), 2 * np.pi)
    RL = 2 ** b - 1; N = 2 ** b
    rmax = r.max(axis=1)
    safe = np.where(rmax > 0, rmax, 1.0)[:, None]
    rq = np.clip(np.floor(r / safe * RL + 0.5), 0, RL)
    tq = np.mod(np.floor(theta / (2 * np.pi) * N + 0.5), N)
    codes = np.empty(Y.shape, dtype=np.int64)
    codes[:, 0::2] = rq; codes[:, 1::2] = tq
    return codes, rmax

def polar_decode(codes: np.ndarray, rmax: np.ndarray, b: int, seed: int = DEFAULT_SEED):
    RL = 2 ** b - 1; N = 2 ** b
    r = codes[:, 0::2] / RL * np.asarray(rmax, dtype=np.float64)[:, None]
    theta = codes[:, 1::2] / N * 2 * np.pi
    Y = np.empty(codes.shape)
    Y[:, 0::2] = r * np.cos(theta); Y[:, 1::2] = r * np.sin(theta)
    return unrotate(Y, seed)

ENCODERS = {"turboQuant": turbo_encode, "polarQuant": polar_encode}
DECODERS = {"turboQuant": turbo_decode, "polarQuant": polar_decode}

def bits_per_coordinate(d: int, b: int) -> float:
    return (code_stride_bytes(d, b) + 4) * 8 / d
    
CHANNEL_DEFAULT_GROUP = 64
FP16_MAX = 65504.0


def channel_group_count(n: int, G: int) -> int:
    return (n + G - 1) // G


def channel_encode(X: np.ndarray, b: int, G: int = CHANNEL_DEFAULT_GROUP):
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    if not np.isfinite(X).all() or np.abs(X).max() > FP16_MAX:
        raise ValueError("inputs must be finite with magnitude at most 65504 (fp16 range)")
    L = np.float32(2 ** b - 1)
    codes = np.empty((n, d), dtype=np.int64)
    ranges = np.empty((channel_group_count(n, G), 2, d), dtype=np.float16)
    for g, s in enumerate(range(0, n, G)):
        blk = X[s:s + G]
        lo16, hi16 = blk.min(0).astype(np.float16), blk.max(0).astype(np.float16)
        lo = lo16.astype(np.float32)
        step = (hi16.astype(np.float32) - lo) / L
        step[step == 0] = np.float32(1)
        q = np.floor((blk - lo) / step + np.float32(0.5))
        codes[s:s + G] = np.clip(q, 0, L).astype(np.int64)
        ranges[g, 0], ranges[g, 1] = lo16, hi16
    return codes, ranges


def channel_decode(codes: np.ndarray, ranges: np.ndarray, b: int, G: int = CHANNEL_DEFAULT_GROUP):
    n, d = codes.shape
    L = np.float32(2 ** b - 1)
    out = np.empty((n, d), dtype=np.float32)
    for g, s in enumerate(range(0, n, G)):
        lo = ranges[g, 0].astype(np.float32)
        step = (ranges[g, 1].astype(np.float32) - lo) / L
        step[step == 0] = np.float32(1)
        out[s:s + G] = codes[s:s + G].astype(np.float32) * step + lo
    return out


def channel_bits_per_coordinate(d: int, b: int, G: int = CHANNEL_DEFAULT_GROUP) -> float:
    return code_stride_bytes(d, b) * 8 / d + 32 / G
