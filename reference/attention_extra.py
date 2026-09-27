#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np
import pitch_reference as R
from attention_eval import f32, minmax_rtn, log_softmax

SEEDS = [R.DEFAULT_SEED, 1, 2, 3, 4, 5, 6, 7]

# ----------------------------------------------------------------------------- compressors
def turbo(X, b, seed=R.DEFAULT_SEED):
    c, s = R.turbo_encode(X, b, seed)
    return R.turbo_decode(c, s.astype(np.float32).astype(np.float64), b, seed)

def rotate_rtn(X, b, seed=R.DEFAULT_SEED):
    return R.unrotate(minmax_rtn(R.rotate(X, seed), b), seed)

def per_channel_groups(X, b, G):
    """KIVI-style key quantisation: min/max per channel over consecutive groups of G tokens."""
    out = np.empty_like(X)
    L = 2 ** b - 1
    for s in range(0, X.shape[0], G):
        blk = X[s:s + G]
        lo = blk.min(0, keepdims=True); hi = blk.max(0, keepdims=True)
        lo16, hi16 = lo.astype(np.float16).astype(np.float64), hi.astype(np.float16).astype(np.float64)
        st = (hi16 - lo16) / L; st[st == 0] = 1.0
        out[s:s + G] = np.clip(np.round((blk - lo16) / st), 0, L) * st + lo16
    return out

def per_token_fp16(X, b):
    """Per-vector min/max with fp16 scale and zero (KIVI's value quantiser)."""
    L = 2 ** b - 1
    lo = X.min(1, keepdims=True).astype(np.float16).astype(np.float64)
    hi = X.max(1, keepdims=True).astype(np.float16).astype(np.float64)
    st = (hi - lo) / L; st[st == 0] = 1.0
    return np.clip(np.round((X - lo) / st), 0, L) * st + lo

def centred_turbo(X, b, n_calib, seed=R.DEFAULT_SEED):
    mu = X[:n_calib].mean(0, keepdims=True)
    return turbo(X - mu, b, seed)            # mu is NOT added back: softmax is invariant to it

def configs(d, G, n_calib):
    """name -> (bits/coord for K, fn_K, bits/coord for V, fn_V). fn_K(K) returns K-hat."""
    tq = lambda b: R.bits_per_coordinate(d, b)
    mm = lambda b: b + 64 / d
    C = {}
    # A: equal-storage points (plus the originals for reference)
    for b in (3, 4, 5):
        C[f"turbo_b{b}"] = (tq(b), lambda K, b=b: turbo(K, b), tq(b), lambda V, b=b: turbo(V, b))
    for b in (3, 4):
        C[f"naive_rtn_b{b}"] = (mm(b), lambda K, b=b: minmax_rtn(K, b), mm(b), lambda V, b=b: minmax_rtn(V, b))
        C[f"rotate_rtn_b{b}"] = (mm(b), lambda K, b=b: rotate_rtn(K, b), mm(b), lambda V, b=b: rotate_rtn(V, b))
    # B: KIVI-style per-channel keys, per-token values, fp16 side info
    for b in (2, 3, 4):
        C[f"kivi_G{G}_b{b}"] = (b + 32 / G, lambda K, b=b: per_channel_groups(K, b, G),
                                b + 32 / d, lambda V, b=b: per_token_fp16(V, b))
    # B': per-channel keys with TurboQuant values (hybrid)
    for b in (2, 3, 4):
        C[f"kivi_K{b}_turboV{b}"] = (b + 32 / G, lambda K, b=b: per_channel_groups(K, b, G),
                                     tq(b), lambda V, b=b: turbo(V, b))
    # C: mean-centred keys (values as plain TurboQuant). mu costs d floats per head: amortised ~0.
    for b in (2, 3, 4):
        C[f"centred_turbo_b{b}"] = (tq(b), lambda K, b=b: centred_turbo(K, b, n_calib),
                                    tq(b), lambda V, b=b: turbo(V, b))
    return C

# ----------------------------------------------------------------------------- evaluation
BUCKETS = [(1, 15), (16, 63), (64, 10 ** 9)]

def attend(Q, K, V, mask, scale):
    S = (Q @ K.T) * scale
    S[mask] = -np.inf
    lp = log_softmax(S)
    return S, lp, np.exp(lp) @ V

def per_query(P_exact, lp_exact, lp_hat, valid):
    with np.errstate(invalid="ignore"):
            kl = np.where(valid, P_exact * (lp_exact - np.where(valid, lp_hat, 0.0)), 0.0).sum(1)
    top1 = P_exact.argmax(1) == np.exp(lp_hat).argmax(1)
    return kl, top1

def evaluate(path, G, n_calib):
    data = json.loads(Path(path).read_text())
    d, scale, model = data["dim"], data["scale"], data["model"]
    C = configs(d, G, n_calib)
    kl_acc = {n: [] for n in C}; top_acc = {n: [] for n in C}; pos_acc = []
    seed_kl = {b: {s: [] for s in SEEDS} for b in (2, 4)}
    q_share, k_share = [], []                          # per block: channel indices differ between heads
    top4 = lambda e: float(np.sort(e)[::-1][:4].sum() / e.sum())

    for blk in data["blocks"]:
        T, pos = blk["seq_len"], np.array(blk["positions"])
        G_heads = len(blk["heads"])
        K, V = f32(blk["keys"], (T, d)), f32(blk["values"], (T, d))
        Q = f32(blk["queries"], (G_heads, len(pos), d)).reshape(-1, d)
        qpos = np.tile(pos, G_heads)
        mask = np.arange(T)[None, :] > qpos[:, None]; valid = ~mask
        S, lp, O = attend(Q, K, V, mask, scale); P = np.exp(lp)
        pos_acc.append(qpos)
        q_share.append(top4((Q ** 2).mean(0))); k_share.append(top4((K ** 2).mean(0)))

        for name, (_, fk, _, fv) in C.items():
            _, lph, _ = attend(Q, fk(K), fv(V), mask, scale)
            kl, t1 = per_query(P, lp, lph, valid)
            kl_acc[name].append(kl); top_acc[name].append(t1)
        for b in seed_kl:
            for s in SEEDS:
                _, lph, _ = attend(Q, turbo(K, b, s), turbo(V, b, s), mask, scale)
                seed_kl[b][s].append(per_query(P, lp, lph, valid)[0])

    qpos_all = np.concatenate(pos_acc)
    rows = []
    for name, (bk, _, bv, _) in C.items():
        kl = np.concatenate(kl_acc[name]); t1 = np.concatenate(top_acc[name])
        row = dict(model=model, config=name, bits_per_coord=round((bk + bv) / 2, 4),
                   ratio_vs_fp16=round(32 / (bk + bv), 3),
                   kl_mean=kl.mean(), kl_median=float(np.median(kl)), kl_p90=float(np.quantile(kl, 0.9)),
                   top1=t1.mean())
        for lo, hi in BUCKETS:
            sel = (qpos_all >= lo) & (qpos_all <= hi)
            tag = f"pos{lo}-{hi if hi < 10**9 else 'end'}"
            row[f"kl_{tag}"] = kl[sel].mean() if sel.any() else np.nan
            row[f"top1_{tag}"] = t1[sel].mean() if sel.any() else np.nan
        rows.append(row)

    seed_rows = []
    for b, per_seed in seed_kl.items():
        means = np.array([np.concatenate(v).mean() for v in per_seed.values()])
        seed_rows.append(dict(model=model, bits=b, n_seeds=len(means), kl_mean=means.mean(),
                              kl_std=means.std(ddof=1), kl_min=means.min(), kl_max=means.max()))

    conc = dict(model=model, q_energy_top4=float(np.mean(q_share)), k_energy_top4=float(np.mean(k_share)),
                q_energy_top4_max=float(np.max(q_share)), k_energy_top4_max=float(np.max(k_share)),
                uniform_top4=4 / d)
    return rows, seed_rows, conc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--group", type=int, default=64, help="KIVI token group size G")
    ap.add_argument("--calib", type=int, default=32, help="tokens used to estimate the key mean")
    ap.add_argument("--out", type=Path, default=Path("results/attention_extra.csv"))
    a = ap.parse_args()
    rows, seeds, concs = [], [], []
    for f in a.files:
        r, s, c = evaluate(f, a.group, a.calib)
        rows += r; seeds += s; concs.append(c)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for tag, data in (("", rows), ("_seeds", seeds), ("_concentration", concs)):
        p = a.out.with_name(a.out.stem + tag + a.out.suffix)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        print(f"wrote {p}")

    print(f"\n{'model':10s} {'config':20s} {'bits':>5s} {'xfp16':>6s} {'KL mean':>9s} {'median':>9s} "
          f"{'p90':>9s} {'top1':>6s} {'top1 pos<16':>11s}")
    for r in rows:
        print(f"{r['model'][:10]:10s} {r['config']:20s} {r['bits_per_coord']:5.2f} {r['ratio_vs_fp16']:6.2f} "
              f"{r['kl_mean']:9.3g} {r['kl_median']:9.3g} {r['kl_p90']:9.3g} {r['top1']:6.3f} "
              f"{r['top1_pos1-15']:11.3f}")
    print("\nseed variance (TurboQuant, K and V at b bits):")
    for s in seeds:
        print(f"  {s['model'][:10]:10s} b={s['bits']}  KL {s['kl_mean']:.4g} +- {s['kl_std']:.2g} "
              f"(min {s['kl_min']:.4g}, max {s['kl_max']:.4g}, {s['n_seeds']} seeds)")
    print("\nchannel concentration (share of mean-square energy in the top 4 of d channels):")
    for c in concs:
        print(f"  {c['model'][:10]:10s} queries {c['q_energy_top4']:.2f}  keys {c['k_energy_top4']:.2f}  "
              f"(uniform {c['uniform_top4']:.2f}; worst block q {c['q_energy_top4_max']:.2f}, k {c['k_energy_top4_max']:.2f})")

if __name__ == "__main__":
    main()
