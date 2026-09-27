#!/usr/bin/env python3
import argparse, base64, csv, json
from pathlib import Path
import numpy as np
import pitch_reference as R

BITS = [2, 3, 4, 8]

def f32(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float64).reshape(shape)

def minmax_rtn(X, b):
    L = 2 ** b - 1
    lo = X.min(1, keepdims=True); hi = X.max(1, keepdims=True)
    st = (hi - lo) / L; st[st == 0] = 1.0
    return np.clip(np.round((X - lo) / st), 0, L) * st + lo

def compressors(d):
    """name -> (bits label, bits/coordinate, function X -> X̂)."""
    out = {("fp16", 16): (16.0, lambda X: X.astype(np.float16).astype(np.float64))}
    for b in BITS:
        out[("naive_rtn", b)] = (b + 64 / d, lambda X, b=b: minmax_rtn(X, b))
        out[("rotate_rtn", b)] = (b + 64 / d, lambda X, b=b: R.unrotate(minmax_rtn(R.rotate(X, R.DEFAULT_SEED), b), R.DEFAULT_SEED))
        for m in ("turboQuant", "polarQuant"):
            def run(X, m=m, b=b):
                c, s = R.ENCODERS[m](X, b)
                return R.DECODERS[m](c, s.astype(np.float32).astype(np.float64), b)
            out[(m, b)] = (R.bits_per_coordinate(d, b), run)
    return out

def log_softmax(S):
    m = S.max(axis=-1, keepdims=True)
    return S - m - np.log(np.exp(S - m).sum(axis=-1, keepdims=True))

MIXED = [(4, 2), (4, 3), (8, 2), (8, 3), (8, 4)]          # (key bits, value bits)

def build_variants(comps):
    """Each variant: which compressor applies to K and to V (None = exact), plus its storage cost."""
    vs = []
    for key, (bpc, _) in comps.items():
        method, bits = key
        modes = [("both", key, key), ("keys_only", key, None), ("values_only", None, key)]
        for mode, kk, vk in modes:
            for sink in (0, 1):
                if method == "fp16" and (mode != "both" or sink):
                    continue
                vs.append(dict(method=method, bits=str(bits), kv_mode=mode, sink=sink, k=kk, v=vk, bpc=bpc))
    for method in ("naive_rtn", "rotate_rtn", "turboQuant", "polarQuant"):
        for kb, vb in MIXED:
            kk, vk = (method, kb), (method, vb)
            vs.append(dict(method=method, bits=f"K{kb}V{vb}", kv_mode="mixed", sink=0, k=kk, v=vk,
                           bpc=(comps[kk][0] + comps[vk][0]) / 2))
    return vs

def evaluate(path):
    data = json.loads(Path(path).read_text())
    d, scale = data["dim"], data["scale"]
    comps = compressors(d)
    fp16 = comps[("fp16", 16)][1]
    variants = build_variants(comps)
    acc = [dict(out=0.0, outv=0.0, logit_se=0.0, logit_n=0, kl=0.0, top1=0, n=0, ss=0.0, s2=0.0) for _ in variants]
    stats = dict(logit_sum=0.0, logit_sq=0.0, logit_n=0, sink=0.0, n=0)

    for blk in data["blocks"]:
        T, pos = blk["seq_len"], np.array(blk["positions"])
        G, P = len(blk["heads"]), len(pos)
        K, V = f32(blk["keys"], (T, d)), f32(blk["values"], (T, d))
        Q = f32(blk["queries"], (G, P, d)).reshape(G * P, d)
        qpos = np.tile(pos, G)
        mask = np.arange(T)[None, :] > qpos[:, None]              # causal
        valid = ~mask
        vnorm2 = float((V ** 2).sum(1).mean())

        S = (Q @ K.T) * scale
        S[mask] = -np.inf
        logP = log_softmax(S); Pm = np.exp(logP); O = Pm @ V
        stats["logit_sum"] += float(S[valid].sum()); stats["logit_sq"] += float((S[valid] ** 2).sum())
        stats["logit_n"] += int(valid.sum()); stats["sink"] += float(Pm[:, 0].sum()); stats["n"] += Q.shape[0]

        Khat = {k: fn(K) for k, (_, fn) in comps.items()}
        Vhat = {k: fn(V) for k, (_, fn) in comps.items()}
        K16, V16 = fp16(K[:1]), fp16(V[:1])
        for var, a in zip(variants, acc):
            Kh = Khat[var["k"]] if var["k"] else K
            Vh = Vhat[var["v"]] if var["v"] else V
            if var["sink"]:
                Kh, Vh = Kh.copy(), Vh.copy()
                Kh[:1], Vh[:1] = K16, V16
            Sh = (Q @ Kh.T) * scale
            Sh[mask] = -np.inf
            logPh = log_softmax(Sh); Ph = np.exp(logPh); Oh = Ph @ Vh
            err = ((Oh - O) ** 2).sum(1)
            a["out"] += float((err / (O ** 2).sum(1)).sum()); a["outv"] += float(err.sum() / vnorm2)
            diff = Sh[valid] - S[valid]
            a["logit_se"] += float((diff ** 2).sum()); a["logit_n"] += diff.size
            a["kl"] += float((Pm[valid] * (logP[valid] - logPh[valid])).sum())
            a["top1"] += int((Pm.argmax(1) == Ph.argmax(1)).sum())
            a["n"] += Q.shape[0]
            a["ss"] += float((S[valid] * Sh[valid]).sum()); a["s2"] += float((S[valid] ** 2).sum())

    mean = stats["logit_sum"] / stats["logit_n"]
    logit_std = (stats["logit_sq"] / stats["logit_n"] - mean ** 2) ** 0.5
    rows = []
    for var, a in zip(variants, acc):
        rows.append(dict(model=data["model"], method=var["method"], bits=var["bits"], kv_mode=var["kv_mode"],
                         sink_fp16=var["sink"], bits_per_coord=round(var["bpc"], 4),
                         ratio_vs_fp16=round(16 / var["bpc"], 4),
                         out_rel_err=a["out"] / a["n"], out_err_vnorm=a["outv"] / a["n"],
                         logit_rmse=(a["logit_se"] / a["logit_n"]) ** 0.5, logit_std=logit_std,
                         kl=a["kl"] / a["n"], top1_agree=a["top1"] / a["n"],
                         score_slope=a["ss"] / a["s2"], sink_mass=stats["sink"] / stats["n"],
                         n_queries=a["n"]))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results/attention.csv"))
    args = ap.parse_args()
    rows = []
    for f in args.files:
        rows += evaluate(f)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'model':8s} {'method':11s} {'b':>2s} {'mode':11s} {'sink':>4s} {'xfp16':>5s} {'out err':>9s} "
          f"{'err/|v|':>9s} {'logit':>7s} {'KL':>9s} {'top1':>6s} {'slope':>6s}")
    for r in rows:
        if r["kv_mode"] not in ("both", "mixed") or r["sink_fp16"]:
            continue                                      # full table is in the CSV
        print(f"{r['model'][:8]:8s} {r['method']:11s} {r['bits']:>2} {r['kv_mode']:11s} {r['sink_fp16']:>4} "
              f"{r['ratio_vs_fp16']:5.2f} {r['out_rel_err']:9.2e} {r['out_err_vnorm']:9.2e} {r['logit_rmse']:7.3f} "
              f"{r['kl']:9.2e} {r['top1_agree']:6.3f} {r['score_slope']:6.3f}")
    for m in dict.fromkeys(r["model"] for r in rows):
        r0 = next(r for r in rows if r["model"] == m)
        print(f"{m}: exact logit std {r0['logit_std']:.3f}, mean attention on token 0 {r0['sink_mass']:.3f}")
    print(f"wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
