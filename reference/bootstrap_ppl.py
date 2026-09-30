#!/usr/bin/env python3
"""95% confidence intervals for perplexity by paired bootstrap over windows.
    python3 bootstrap_ppl.py results/perplexity_windows_gpt2.csv
    python3 bootstrap_ppl.py results/perplexity_windows_Qwen_Qwen2.5-0.5B.csv --pairs kivi_G64_b4:turbo_split_b4
"""
import argparse, csv
from collections import defaultdict
import numpy as np

PAIRS = ["kivi_G64_b4:turbo_b4", "kivi_G64_b4:kivi_K4_turboV4", "kivi_G64_b4:turbo_split_b4",
         "kivi_G64_b4:mlx_b4", "kivi_G64_b4:q4_0", "turbo_split_b4:turbo_b4",
         # causal against non-causal, both through the same attention path
         "kivi_causal_reencode_b4:kivi_full_path_b4", "kivi_causal_residual_b4:kivi_full_path_b4",
         "kivi_causal_reencode_b3:kivi_full_path_b3", "kivi_causal_residual_b3:kivi_full_path_b3",
         "kivi_full_path_b4:exact_path", "exact_path:exact"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--pairs", nargs="*", default=PAIRS, help="a:b compares a against b")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for path in a.files:
        data = defaultdict(dict)
        model = None
        for r in csv.DictReader(open(path)):
            model = r["model"]
            data[r["config"]][int(r["window"])] = (float(r["nll_sum"]), int(r["tokens"]))
        windows = sorted(data["exact"])
        W = len(windows)
        tokens = np.array([data["exact"][w][1] for w in windows], dtype=float)
        nll = {c: np.array([v[w][0] for w in windows]) for c, v in data.items() if len(v) == W}
        rng = np.random.default_rng(a.seed)
        idx = rng.integers(0, W, size=(a.resamples, W))
        boot = {c: np.exp(s[idx].sum(1) / tokens[idx].sum(1)) for c, s in nll.items()}
        point = {c: float(np.exp(s.sum() / tokens.sum())) for c, s in nll.items()}
        ci = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))

        print(f"\n{model}: {W} windows, {a.resamples} paired resamples")
        print(f"{'config':28s} {'PPL':>9s} {'95% CI':>20s} {'change vs exact':>16s} {'95% CI':>20s}")
        for c in nll:
            lo, hi = ci(boot[c])
            rel = 100 * (boot[c] / boot["exact"] - 1)
            rlo, rhi = ci(rel)
            print(f"{c:28s} {point[c]:9.3f} [{lo:8.3f}, {hi:8.3f}] {100 * (point[c] / point['exact'] - 1):+15.2f}% "
                  f"[{rlo:+7.2f}%, {rhi:+7.2f}%]")
        print("\npaired comparisons (ratio of perplexities; below 1 means the first is better)")
        for p in a.pairs:
            x, y = p.split(":")
            if x not in boot or y not in boot:
                continue
            r = boot[x] / boot[y]
            lo, hi = ci(r)
            verdict = "first better" if hi < 1 else "second better" if lo > 1 else "not distinguishable"
            print(f"  {x:26s} vs {y:24s} {point[x] / point[y]:7.4f} [{lo:.4f}, {hi:.4f}]  {verdict}")


if __name__ == "__main__":
    main()
