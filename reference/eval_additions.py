from functools import lru_cache
import numpy as np
import pitch_reference as R

F16 = lambda a: np.asarray(a, dtype=np.float64).astype(np.float16).astype(np.float64)


# ----------------------------------------------------------------------------- MLX-style
def per_token_groups_fp16(X, b, g=64):
    """X (n, d). Affine min/max quantisation over groups of g consecutive channels of each token."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    g = min(g, d)
    L = 2 ** b - 1
    Y = X.reshape(n, d // g, g)
    lo, hi = F16(Y.min(2, keepdims=True)), F16(Y.max(2, keepdims=True))
    st = (hi - lo) / L
    st[st == 0] = 1.0
    return (np.clip(np.round((Y - lo) / st), 0, L) * st + lo).reshape(n, d)


def bits_per_token_groups(d, b, g=64):
    return b + 32 / min(g, d)


# ----------------------------------------------------------------------------- llama.cpp
def q4_0(X):
    """ggml quantize_row_q4_0_ref: blocks of 32; d = (signed value of largest magnitude) / -8;
    q = min(15, trunc(x/d + 8.5)); x_hat = (q - 8) * fp16(d)."""
    X = np.asarray(X, dtype=np.float32)
    n, dim = X.shape
    B = X.reshape(n, dim // 32, 32)
    idx = np.abs(B).argmax(2)[..., None]
    mx = np.take_along_axis(B, idx, 2)
    d = mx / np.float32(-8)
    inv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, 1), 0).astype(np.float32)
    q = np.minimum(15, np.trunc(B * inv + np.float32(8.5)))
    return ((q - 8) * F16(d)).reshape(n, dim).astype(np.float64)


def q8_0(X):
    """ggml quantize_row_q8_0_ref: blocks of 32; d = amax / 127; q = round(x / d)."""
    X = np.asarray(X, dtype=np.float32)
    n, dim = X.shape
    B = X.reshape(n, dim // 32, 32)
    d = np.abs(B).max(2, keepdims=True) / np.float32(127)
    inv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, 1), 0).astype(np.float32)
    q = np.sign(B * inv) * np.floor(np.abs(B * inv) + 0.5)          # roundf: half away from zero
    return (q * F16(d)).reshape(n, dim).astype(np.float64)

@lru_cache(maxsize=None)
def _codebook(d, b):
    return R.compute_sphere_codebook(d, b)


@lru_cache(maxsize=None)
def _haar(d, seed):
    rng = np.random.default_rng(seed + 1000 * d)
    Q, Rm = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(Rm))


def _turbo_haar(X, b, seed):
    """TurboQuant-MSE on (n, d) with a Haar rotation, for dimensions that are not powers of two."""
    d = X.shape[1]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    Pi = _haar(d, seed)
    U = (X / safe) @ Pi.T
    cb = _codebook(d, b)
    codes = np.searchsorted((cb[1:] + cb[:-1]) / 2, U, side="left")
    return (cb[codes] @ Pi) * norms


def turbo_split(X, b_hi, b_lo, n_calib=32, frac=0.25, seed=R.DEFAULT_SEED):
    """X: one head's (T, d) vectors in token order."""
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    k = max(1, int(round(d * frac)))
    energy = (X[:n_calib] ** 2).mean(0)
    out_idx = np.sort(np.argsort(energy)[::-1][:k])
    rest = np.setdiff1d(np.arange(d), out_idx)
    Y = np.empty_like(X)
    Y[:, out_idx] = _turbo_haar(X[:, out_idx], b_hi, seed)
    Y[:, rest] = _turbo_haar(X[:, rest], b_lo, seed + 1)
    return Y


def bits_turbo_split(d, b_hi, b_lo, frac=0.25):
    k = max(1, int(round(d * frac)))
    return (k * b_hi + (d - k) * b_lo + 64) / d        # two fp32 norms


# ----------------------------------------------------------------------------- causal per-channel keys
def _channel_quant(blk, lo, hi, b):
    L = 2 ** b - 1
    st = (hi - lo) / L
    st = np.where(st == 0, 1.0, st)
    return np.clip(np.round((blk - lo) / st), 0, L) * st + lo


def causal_kivi_scores(Q, qpos, K, b, G, mode):
    from attention_extra import per_channel_groups
    K = np.asarray(K, dtype=np.float64)
    T = K.shape[0]
    full = K if mode == "check" else per_channel_groups(K, b, G)
    S = Q @ full.T
    if mode == "full":
        return S
    for s in range(0, T, G):
        e = min(s + G, T)
        sel = np.where((qpos >= s) & (qpos < e))[0]
        if sel.size == 0:
            continue
        blk = K[s:e]
        if mode == "check":
            S[np.ix_(sel, np.arange(s, e))] = Q[sel] @ blk.T
        elif mode == "residual":
            S[np.ix_(sel, np.arange(s, e))] = Q[sel] @ F16(blk).T
        elif mode == "reencode":
            lo_t = F16(np.minimum.accumulate(blk, axis=0))
            hi_t = F16(np.maximum.accumulate(blk, axis=0))
            t_rel = qpos[sel] - s                                 
            Kt = _channel_quant(blk[None, :, :], lo_t[t_rel][:, None, :], hi_t[t_rel][:, None, :], b)
            S[np.ix_(sel, np.arange(s, e))] = np.einsum("md,mjd->mj", Q[sel], Kt)
        else:
            raise ValueError(mode)
    return S
