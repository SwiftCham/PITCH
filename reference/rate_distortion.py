#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pitch_reference as R

F32 = 32

def minmax_rtn(Y, b):
    L = 2 ** b - 1
    lo = Y.min(1, keepdims=True); hi = Y.max(1, keepdims=True)
    sc = (hi - lo) / L; sc[sc == 0] = 1.0
    return np.clip(np.round((Y - lo) / sc), 0, L) * sc + lo

# --------------------------------------------------------------------------- methods: (X, b) -> (X_hat, bits/coord)
def naive_rtn(X, b):
    return minmax_rtn(X, b), b + 2 * F32 / X.shape[1]

def rotate_rtn(X, b, seed=R.DEFAULT_SEED):
    return R.unrotate(minmax_rtn(R.rotate(X, seed), b), seed), b + 2 * F32 / X.shape[1]

def pitch_method(name):
    def run(X, b):
        codes, scales = R.ENCODERS[name](X, b)
        Xh = R.DECODERS[name](codes, scales.astype(np.float32).astype(np.float64), b)
        return Xh, R.bits_per_coordinate(X.shape[1], b)
    return run

def polar_seed_per_vector(X, b):
    Xh, bpc = pitch_method("polarQuant")(X, b)
    return Xh, bpc + F32 / X.shape[1]

def legacy_turbo_minmax_residual(X, b, seed=R.DEFAULT_SEED):
    """Pre-Step-2 kernel: min/max rounding + per-coordinate residual sign * RMS.
    Stored b+1 bits/coord plus scale, offset, residualScale and a per-vector seed."""
    d = X.shape[1]
    Y = R.rotate(X, seed)
    L = 2 ** b - 1
    lo = Y.min(1, keepdims=True); hi = Y.max(1, keepdims=True)
    sc = (hi - lo) / L; sc[sc == 0] = 1.0
    dq = np.clip(np.round((Y - lo) / sc), 0, L) * sc + lo
    r = Y - dq
    rs = np.sqrt((r ** 2).mean(1, keepdims=True))
    return R.unrotate(dq + np.where(r >= 0, rs, -rs), seed), b + 1 + 4 * F32 / d

# ---- recursive PolarQuant prototype (Step 3) -------------------------------
def _polar_forward(Y):
    angles = []
    a, c = Y[:, 0::2], Y[:, 1::2]
    r = np.hypot(a, c); angles.append(np.mod(np.arctan2(c, a), 2 * np.pi))
    while r.shape[1] > 1:
        a, c = r[:, 0::2], r[:, 1::2]
        angles.append(np.arctan2(c, a))
        r = np.hypot(a, c)
    return angles, r

def _polar_inverse(angles, r):
    for psi in reversed(angles[1:]):
        out = np.empty((r.shape[0], r.shape[1] * 2))
        out[:, 0::2] = r * np.cos(psi); out[:, 1::2] = r * np.sin(psi)
        r = out
    Y = np.empty((r.shape[0], r.shape[1] * 2))
    Y[:, 0::2] = r * np.cos(angles[0]); Y[:, 1::2] = r * np.sin(angles[0])
    return Y

_polar_cb = {}
def _polar_level_codebooks(d, bits_hi):
    """Level-l angles (l >= 2) of a Gaussian vector have density ∝ sin(2 psi)^(2^(l-1) - 1)."""
    key = (d, bits_hi)
    if key not in _polar_cb:
        psi = np.linspace(0, np.pi / 2, 200001)[1:-1]
        _polar_cb[key] = [None] + [R.lloyd_max_density(psi, np.sin(2 * psi) ** (2 ** (l - 1) - 1), 2 ** bits_hi)
                                   for l in range(2, int(math.log2(d)) + 1)]
    return _polar_cb[key]

def polar_recursive_prototype(X, alloc, seed=R.DEFAULT_SEED):
    bits_l1, bits_hi = alloc
    d = X.shape[1]
    angles, r = _polar_forward(R.rotate(X, seed))
    N = 2 ** bits_l1
    q = [np.mod(np.round(angles[0] / (2 * np.pi) * N), N) / N * 2 * np.pi]
    cbs = _polar_level_codebooks(d, bits_hi)
    for l, a in enumerate(angles[1:], start=1):
        cb = cbs[l]; q.append(cb[np.searchsorted((cb[1:] + cb[:-1]) / 2, a)])
    bits = (d // 2) * bits_l1 + (d // 2 - 1) * bits_hi + F32
    return R.unrotate(_polar_inverse(q, r), seed), bits / d

ALL_BITS = list(R.SUPPORTED_BITS)
METHODS = {
    "naive_rtn":                         (naive_rtn, ALL_BITS),
    "rotate_rtn":                        (rotate_rtn, ALL_BITS),
    "pitch_turboquant":                  (pitch_method("turboQuant"), ALL_BITS),
    "pitch_polarquant":                  (pitch_method("polarQuant"), ALL_BITS),
    "pitch_polarquant_seed_per_vector":  (polar_seed_per_vector, ALL_BITS),
    "legacy_turbo_minmax_residual":      (legacy_turbo_minmax_residual, [3, 4, 8]),
    "polar_recursive_prototype":         (polar_recursive_prototype, [(2, 1), (3, 2), (4, 2), (4, 3), (5, 4), (6, 5), (7, 6)]),
}

def rel_err(X, Xh):
    return float((((X - Xh) ** 2).sum(1) / (X ** 2).sum(1)).mean())

def load(path):
    payload = json.loads(Path(path).read_text())
    return payload["model"], {t: np.array([v["data"] for v in payload["vectors"] if v["type"] == t],
                                          dtype=np.float32).astype(np.float64) for t in ("key", "value")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in args.files:
        model, data = load(f)
        for kind, X in data.items():
            for name, (fn, settings) in METHODS.items():
                for s in settings:
                    Xh, bpc = fn(X, s)
                    rows.append(dict(model=model, kind=kind, method=name, setting=str(s),
                                     bits_per_coord=round(bpc, 4), rel_err=rel_err(X, Xh)))
            print(f"  {model} {kind}: done")
    out = args.out / "rate_distortion.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    main()
