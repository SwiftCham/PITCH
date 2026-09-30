#!/usr/bin/env python3
"""
python3 attention_more.py attn_blocks_gpt2.json attn_blocks_qwen.json
"""
import argparse, base64, csv, json
from pathlib import Path
import numpy as np
from attention_eval import log_softmax
from attention_extra import turbo, per_channel_groups, per_token_fp16, per_query
import eval_additions as E


def f32(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float64).reshape(shape)


def configs(d, G):
    """name -> (bits/coord, key spec, value fn). key spec: fn(K) or ("causal", mode, b)."""
    tq = lambda b: (lambda X: turbo(X, b))
    C = {
        "turbo_b4": (4 + 32 / d, tq(4), tq(4)),
        "turbo_b3": (3 + 32 / d, tq(3), tq(3)),
        f"kivi_G{G}_b4": (4 + 32 / G, lambda K: per_channel_groups(K, 4, G), lambda V: per_token_fp16(V, 4)),
        f"kivi_G{G}_b3": (3 + 32 / G, lambda K: per_channel_groups(K, 3, G), lambda V: per_token_fp16(V, 3)),
        "mlx_b4": (E.bits_per_token_groups(d, 4), lambda X: E.per_token_groups_fp16(X, 4),
                   lambda X: E.per_token_groups_fp16(X, 4)),
        "mlx_b8": (E.bits_per_token_groups(d, 8), lambda X: E.per_token_groups_fp16(X, 8),
                   lambda X: E.per_token_groups_fp16(X, 8)),
        "q4_0": (4.5, E.q4_0, E.q4_0),
        "q8_0": (8.5, E.q8_0, E.q8_0),
        "turbo_split_b4": (E.bits_turbo_split(d, 5, 3), lambda X: E.turbo_split(X, 5, 3),
                           lambda X: E.turbo_split(X, 5, 3)),
        "turbo_split_b3": (E.bits_turbo_split(d, 4, 2), lambda X: E.turbo_split(X, 4, 2),
                           lambda X: E.turbo_split(X, 4, 2)),
    }
    for b in (3, 4):
        for mode in ("residual", "reencode"):
            C[f"kivi_causal_{mode}_b{b}"] = (b + 32 / G, ("causal", mode, b), lambda V, b=b: per_token_fp16(V, b))
    return C


def evaluate(path, G):
    data = json.loads(Path(path).read_text())
    d, scale, model = data["dim"], data["scale"], data["model"]
    C = configs(d, G)
    kl = {n: [] for n in C}; top = {n: [] for n in C}
    for blk in data["blocks"]:
        T, pos = blk["seq_len"], np.array(blk["positions"])
        K, V = f32(blk["keys"], (T, d)), f32(blk["values"], (T, d))
        Q = f32(blk["queries"], (len(blk["heads"]), len(pos), d)).reshape(-1, d)
        qpos = np.tile(pos, len(blk["heads"]))
        mask = np.arange(T)[None, :] > qpos[:, None]
        valid = ~mask
        S = (Q @ K.T) * scale; S[mask] = -np.inf
        lp = log_softmax(S); Pm = np.exp(lp)
        for name, (_, fk, _) in C.items():          # values do not affect KL or top-1
            if isinstance(fk, tuple):
                Sh = E.causal_kivi_scores(Q, qpos, K, fk[2], G, fk[1]) * scale
            else:
                Sh = (Q @ fk(K).T) * scale
            Sh[mask] = -np.inf
            k_, t_ = per_query(Pm, lp, log_softmax(Sh), valid)
            kl[name].append(k_); top[name].append(t_)
    rows = []
    for name, (bits, _, _) in C.items():
        k_ = np.concatenate(kl[name]); t_ = np.concatenate(top[name])
        rows.append(dict(model=model, config=name, bits_per_coord=round(bits, 4),
                         ratio_vs_fp16=round(16 / bits, 3), kl_mean=k_.mean(),
                         kl_median=float(np.median(k_)), kl_p90=float(np.quantile(k_, 0.9)), top1=t_.mean()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("results/attention_more.csv"))
    a = ap.parse_args()
    rows = []
    for f in a.files:
        rows += evaluate(f, a.group)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}\n")
    print(f"{'model':18s} {'config':26s} {'bits':>5s} {'xfp16':>6s} {'KL mean':>9s} {'median':>9s} {'p90':>9s} {'top1':>6s}")
    for r in rows:
        print(f"{r['model'][-18:]:18s} {r['config']:26s} {r['bits_per_coord']:5.2f} {r['ratio_vs_fp16']:6.2f} "
              f"{r['kl_mean']:9.3g} {r['kl_median']:9.3g} {r['kl_p90']:9.3g} {r['top1']:6.3f}")


if __name__ == "__main__":
    main()
