#!/usr/bin/env python3
"""Verification controls quoted in Section 6.1 of the paper.

    python3 controls.py haar kv_vectors.json kv_vectors_qwen.json
        Hadamard versus a dense Haar rotation on the real KV vectors: same data, same codebook
        and norm handling as the specification, only the rotation changed (Haar over --seeds).

    python3 controls.py worstcase
        Worst-case inputs for the randomized Hadamard transform against Theorem 1, and the
        mitigation of a second Hadamard round after a fixed random permutation.

    python3 controls.py channel attn_blocks_gpt2.json attn_blocks_qwen.json
        Confirms the Metal per-channel mode is the quantizer the paper evaluated: the float32
        specification the kernels follow against the float64 version used for the accuracy results.
"""
import argparse, base64, json, math
from pathlib import Path
import numpy as np
import pitch_reference as R


def rel_err(X, Xh):
    return float((((X - Xh) ** 2).sum(1) / (X ** 2).sum(1)).mean())


# ----------------------------------------------------------------------------- haar
def _load_kv(path):
    payload = json.loads(Path(path).read_text())
    return payload["model"], {t: np.array([v["data"] for v in payload["vectors"] if v["type"] == t],
                                          dtype=np.float32).astype(np.float64) for t in ("key", "value")}


def _hadamard(X, b):
    codes, norms = R.turbo_encode(X, b)
    return R.turbo_decode(codes, norms.astype(np.float32).astype(np.float64), b)


def _haar(X, b, seed):
    d = X.shape[1]
    Q, Rm = np.linalg.qr(np.random.default_rng([0x48414152, seed]).standard_normal((d, d)))
    Pi = Q * np.sign(np.diag(Rm))
    norms = np.linalg.norm(X, axis=1)
    safe = np.where(norms > 0, norms, 1.0)[:, None]
    U = (X / safe) @ Pi.T
    cb32 = R.turbo_codebook(d, b).astype(np.float32)
    edges = (np.float32(0.5) * (cb32[1:] + cb32[:-1])).astype(np.float64)
    Uh = R.turbo_codebook(d, b)[np.searchsorted(edges, U, side="left")]
    return (Uh @ Pi) * norms.astype(np.float32).astype(np.float64)[:, None]


def haar(a):
    print(f"{'model':18s} {'kind':5s} {'b':>2s} {'Hadamard':>10s} {'Haar mean':>10s} {'Haar min-max':>21s} {'ratio':>6s}")
    ratios = []
    for path in a.files:
        model, sets = _load_kv(path)
        for kind, X in sets.items():
            for b in R.SUPPORTED_BITS:
                h = rel_err(X, _hadamard(X, b))
                q = np.array([rel_err(X, _haar(X, b, s)) for s in range(a.seeds)])
                ratios.append(h / q.mean())
                print(f"{model[-18:]:18s} {kind:5s} {b:2d} {h:10.4g} {q.mean():10.4g} "
                      f"[{q.min():9.4g}, {q.max():9.4g}] {h / q.mean():6.3f}")
    r = np.array(ratios)
    print(f"\nHadamard / Haar error ratio over all sets and bit-widths: mean {r.mean():.3f}, "
          f"range {r.min():.3f}-{r.max():.3f}")


def worstcase(a):
    rng = np.random.default_rng(0)                # one stream, consumed in a fixed order
    BOUND = lambda b: math.sqrt(3) * math.pi / 2 * 4.0 ** -b

    def tq(X, b, seed=R.DEFAULT_SEED):
        c, n = R.turbo_encode(X, b, seed); return R.turbo_decode(c, n.astype(np.float32).astype(np.float64), b, seed)

    def tq_haar(X, b, Q):
        d = X.shape[1]; n = np.linalg.norm(X, axis=1, keepdims=True); U = (X / n) @ Q.T
        cb = R.turbo_codebook(d, b).astype(np.float32); e = (np.float32(.5) * (cb[1:] + cb[:-1])).astype(float)
        return (R.turbo_codebook(d, b)[np.searchsorted(e, U)] @ Q) * n

    def haar_matrix(d):
        A = rng.standard_normal((d, d)); Q, Rr = np.linalg.qr(A); return Q * np.sign(np.diag(Rr))

    def tq_double(X, b, s1=R.DEFAULT_SEED, s2=0x12345678):
        # mitigation: two independent RHT rounds (still O(d log d))
        n = np.linalg.norm(X, axis=1, keepdims=True)
        U = R.rotate(R.rotate(X / n, s1), s2); d = X.shape[1]
        cb = R.turbo_codebook(d, b).astype(np.float32); e = (np.float32(.5) * (cb[1:] + cb[:-1])).astype(float)
        Uh = R.turbo_codebook(d, b)[np.searchsorted(e, U)]
        return R.unrotate(R.unrotate(Uh, s2), s1) * n

    def tq_perm(X, b, perm, s1=R.DEFAULT_SEED, s2=0x12345678):
        # mitigation: RHT -> fixed random permutation -> RHT
        n = np.linalg.norm(X, axis=1, keepdims=True); d = X.shape[1]
        U = R.rotate(R.rotate(X / n, s1)[:, perm], s2)
        cb = R.turbo_codebook(d, b).astype(np.float32); e = (np.float32(.5) * (cb[1:] + cb[:-1])).astype(float)
        Uh = R.turbo_codebook(d, b)[np.searchsorted(e, U)]
        Y = R.unrotate(Uh, s2); inv = np.argsort(perm); return R.unrotate(Y[:, inv], s1) * n

    def sparse(d, spikes):
        X = np.zeros((d, d))
        for i in range(d):
            for j in range(spikes): X[i, (i + j * 7) % d] = 1
        return X

    for d in (64, 128):
        Q = haar_matrix(d)
        print(f"\n=== d={d} ===  (mean normalised error; bound = sqrt(3)pi/2 * 4^-b)")
        print(f"{'input':34s}" + "".join(f"  b={b}:RHT/2xRHT/Haar      " for b in (2, 3, 4)))
        inputs = {}
        inputs["one-hot, every position (d vecs)"] = np.eye(d)
        k = np.zeros((d, d))
        for i in range(d): k[i, i] = 1; k[i, (i + 1) % d] = 1
        inputs["two equal spikes"] = k
        for ratio in (10, 30):
            X = rng.standard_normal((2048, d)); ch = rng.integers(0, d, 2048)
            X[np.arange(2048), ch] *= ratio
            inputs[f"Gaussian + one channel x{ratio}"] = X
        inputs["Gaussian (control)"] = rng.standard_normal((2048, d))
        for name, X in inputs.items():
            row = f"{name:34s}"
            for b in (2, 3, 4):
                row += f"  {rel_err(X, tq(X, b)):.4f}/{rel_err(X, tq_double(X, b)):.4f}/{rel_err(X, tq_haar(X, b, Q)):.4f}"
            print(row)
        print(f"{'Theorem 1 bound':34s}" + "".join(f"  {BOUND(b):.4f}                " for b in (2, 3, 4)))

    print("\n--- RHT + permutation + RHT, worst over sparse inputs ---")
    for d in (64, 128):
        perm = np.random.default_rng(7).permutation(d)
        worst = {b: 0 for b in (2, 3, 4, 5, 6)}
        for spikes in (1, 2, 3, 4):
            X = sparse(d, spikes)
            for b in worst: worst[b] = max(worst[b], rel_err(X, tq_perm(X, b, perm)))
        print(d, {b: (round(v, 4), round(v / BOUND(b), 2)) for b, v in worst.items()})
    print("\n--- single RHT, worst over the same sparse inputs (ratio to bound) ---")
    for d in (64, 128):
        worst = {b: 0 for b in (2, 3, 4, 5, 6)}
        for spikes in (1, 2, 3, 4):
            X = sparse(d, spikes)
            for b in worst: worst[b] = max(worst[b], rel_err(X, tq(X, b)))
        print(d, {b: (round(v, 4), round(v / BOUND(b), 2)) for b, v in worst.items()})


def channel(a):
    from attention_extra import per_channel_groups
    G = 64
    here = Path(__file__).resolve().parent
    paths = a.files or [p for p in (here / "attn_blocks_gpt2.json", here / "attn_blocks_qwen.json") if p.exists()]
    for path in paths:
        data = json.loads(Path(path).read_text())
        d = data["dim"]
        for b in (2, 3, 4, 8):
            total = differ = 0
            worst_steps = 0.0
            for blk in data["blocks"]:
                K = np.frombuffer(base64.b64decode(blk["keys"]), dtype="<f4").reshape(blk["seq_len"], d)
                ref = per_channel_groups(K.astype(np.float64), b, G)
                codes, ranges = R.channel_encode(K, b, G)
                ours = R.channel_decode(codes, ranges, b, G).astype(np.float64)
                # size of one quantization step for each coordinate, to express differences in steps
                step = np.empty_like(ref)
                for g, s in enumerate(range(0, K.shape[0], G)):
                    st = (ranges[g, 1].astype(np.float64) - ranges[g, 0].astype(np.float64)) / (2 ** b - 1)
                    step[s:s + G] = np.where(st == 0, 1.0, st)
                diff_steps = np.abs(ref - ours) / step
                differ += int((diff_steps > 0.5).sum())
                total += diff_steps.size
                worst_steps = max(worst_steps, float(diff_steps.max()))
            print(f"{data['model']:>24s}  {b}-bit: {differ} of {total} coordinates differ by a code "
                  f"({100 * differ / total:.4f}%), largest difference {worst_steps:.3f} steps")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("haar", help="Hadamard vs Haar rotation on real KV vectors")
    p.add_argument("files", nargs="+"); p.add_argument("--seeds", type=int, default=8)
    sub.add_parser("worstcase", help="sparse worst cases for the Hadamard rotation")
    p = sub.add_parser("channel", help="per-channel Metal specification vs the evaluated quantizer")
    p.add_argument("files", nargs="*")
    a = ap.parse_args()
    {"haar": haar, "worstcase": worstcase, "channel": channel}[a.what](a)


if __name__ == "__main__":
    main()
