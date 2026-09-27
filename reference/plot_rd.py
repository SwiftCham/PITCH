#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {  # method: (label, colour, marker, linestyle)
    "naive_rtn":                        ("Round-to-nearest, no rotation", "#9e9e9e", "o", "--"),
    "rotate_rtn":                       ("Rotate + round-to-nearest", "#424242", "s", "-"),
    "legacy_turbo_minmax_residual":     ("Earlier PITCH TurboQuant variant (removed)", "#d62728", "v", ":"),
    "pitch_polarquant_seed_per_vector": ("PITCH PolarQuant, seed per vector", "#ff7f0e", "^", ":"),
    "pitch_polarquant":                 ("PITCH PolarQuant (single level)", "#ff7f0e", "^", "-"),
    "pitch_turboquant":                 ("PITCH TurboQuant-MSE", "#1f77b4", "D", "-"),
    "polar_recursive_prototype":        ("Recursive PolarQuant (prototype)", "#2ca02c", "P", "--"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="results/rate_distortion.csv")
    ap.add_argument("--out", default="results/rate_distortion.pdf")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(len(models), 2, figsize=(10, 3.6 * len(models) + 1), sharex=True, sharey=True, squeeze=False)
    for i, model in enumerate(models):
        for j, kind in enumerate(("key", "value")):
            ax = axes[i][j]
            for m, (label, col, mk, ls) in STYLE.items():
                pts = sorted((float(r["bits_per_coord"]), float(r["rel_err"])) for r in rows
                             if r["model"] == model and r["kind"] == kind and r["method"] == m)
                if pts: ax.plot(*zip(*pts), color=col, marker=mk, ls=ls, lw=1.4, ms=4.5, label=label)
            ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.25)
            ax.set_title(f"{model}, {kind}s", fontsize=10, pad=22 if i == 0 else 6)
            if i == 0:   # KV caches are normally fp16, so show the ratio a deployment would see
                top = ax.secondary_xaxis("top", functions=(lambda b: 16 / np.maximum(b, 1e-9),
                                                          lambda r: 16 / np.maximum(r, 1e-9)))
                top.set_xticks([8, 6, 4, 3, 2]); top.set_xticklabels(["8x", "6x", "4x", "3x", "2x"], fontsize=8)
                top.set_xlabel("compression vs fp16", fontsize=8)
            if i == len(models) - 1: ax.set_xlabel("Stored bits per coordinate (incl. side information)")
            if j == 0: ax.set_ylabel(r"$\|x-\hat{x}\|^2/\|x\|^2$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(args.out); fig.savefig(Path(args.out).with_suffix(".png"), dpi=200)
    print(f"saved {args.out} (+ .png)")

if __name__ == "__main__":
    main()
