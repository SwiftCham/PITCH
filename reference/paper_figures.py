#!/usr/bin/env python3
"""
paper_figures.py: regenerate every data figure in the paper from the results CSVs.

    Figure 4  theorem1.pdf         accuracy.csv                            (Swift benchmark)
    Figure 5  rate_distortion.pdf  rate_distortion.csv                     (rate_distortion.py)
    Figure 6  attention_kl.pdf     attention.csv + attention_extra.csv     (attention_eval.py, attention_extra.py)
    Figure 7  throughput.pdf       throughput.csv + pytorch_throughput.csv (Swift benchmark, pytorch_mps_benchmark.py)

    python3 reference/paper_figures.py --results results reference/results --out paper/images
    python3 reference/paper_figures.py --results results reference/results --out paper/images --only attention
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif", "axes.grid": True, "grid.alpha": 0.25,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL, FULL = 3.45, 7.1          # IEEEtran column / text width, inches
C = {"naive_rtn": "#9e9e9e", "rotate_rtn": "#424242", "turboQuant": "#1f77b4", "polarQuant": "#ff7f0e",
     "legacy": "#d62728", "recursive": "#2ca02c", "pytorch": "#8c564b", "perChannel": "#9467bd"}
MODELS = [("gpt2", "GPT-2"), ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B")]
f = float


class Missing(Exception):
    pass


def finder(dirs):
    def read(name, required=True):
        for d in dirs:
            p = Path(d) / name
            if p.exists():
                return list(csv.DictReader(open(p)))
        if required:
            raise Missing(f"{name} not found in {', '.join(map(str, dirs))}")
        return []
    return read


def theorem1(read, out):
    A = [r for r in read("accuracy.csv") if r["source"] == "gaussian" and r["method"] == "turboQuant"]
    b = np.arange(2, 9)
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.fill_between(b, 4.0 ** -b, math.sqrt(3) * math.pi / 2 * 4.0 ** -b, color="#1f77b4", alpha=0.12,
                    label=r"$4^{-b}$ to Theorem 1 bound")
    ax.plot(b, 4.0 ** -b, color="#1f77b4", lw=0.8, ls="--")
    ax.plot(b, math.sqrt(3) * math.pi / 2 * 4.0 ** -b, color="#1f77b4", lw=0.8, ls="--")
    for d, mk in zip((64, 128, 256, 512, 1024), "os^vD"):
        pts = sorted((int(r["bits"]), f(r["rel_err"])) for r in A if r["dim"] == str(d))
        ax.plot(*zip(*pts), marker=mk, ms=3.2, lw=0.9, label=f"$d={d}$")
    ax.set_yscale("log"); ax.set_xlabel("Bits per coordinate $b$ (codes only)")
    ax.set_ylabel(r"$\|x-\hat{x}\|^2/\|x\|^2$")
    ax.legend(ncol=1, frameon=False, loc="lower left", fontsize=6.3)
    fig.savefig(out / "theorem1.pdf"); plt.close(fig)


def rate_distortion(read, out):
    rows = read("rate_distortion.csv")
    style = {  # method: (label, colour, marker, linestyle)
        "naive_rtn": ("Round-to-nearest, no rotation", C["naive_rtn"], "o", "--"),
        "rotate_rtn": ("Rotate + round-to-nearest", C["rotate_rtn"], "s", "-"),
        "pitch_polarquant_seed_per_vector": ("PITCH PolarQuant, seed stored per vector", C["polarQuant"], "^", ":"),
        "pitch_polarquant": ("PITCH PolarQuant (single level)", C["polarQuant"], "^", "-"),
        "pitch_turboquant": ("PITCH TurboQuant-MSE", C["turboQuant"], "D", "-"),
        "polar_recursive_prototype": ("Recursive PolarQuant (Python prototype)", C["recursive"], "P", "--"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.0), sharex=True, sharey=True)
    for i, (mk, mn) in enumerate(MODELS):
        for j, kind in enumerate(("key", "value")):
            ax = axes[i][j]
            for m, (label, col, marker, ls) in style.items():
                pts = sorted((f(r["bits_per_coord"]), f(r["rel_err"])) for r in rows
                             if r["model"] == mk and r["kind"] == kind and r["method"] == m)
                if pts:
                    ax.plot(*zip(*pts), color=col, marker=marker, ls=ls, lw=1.1, ms=3.2, label=label)
            ax.set_yscale("log"); ax.set_xlim(1.8, 11.2)
            ax.set_title(f"{mn}, {kind}s", pad=18 if i == 0 else 4)
            if i == 0:
                top = ax.secondary_xaxis("top", functions=(lambda x: 16 / np.maximum(x, 1e-9),
                                                           lambda x: 16 / np.maximum(x, 1e-9)))
                top.set_xticks([8, 6, 4, 3, 2]); top.set_xticklabels(["8x", "6x", "4x", "3x", "2x"])
                top.set_xlabel("compression vs fp16", fontsize=7)
            if i == 1: ax.set_xlabel("Stored bits per coordinate (including side information)")
            if j == 0: ax.set_ylabel(r"$\|x-\hat{x}\|^2/\|x\|^2$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out / "rate_distortion.pdf"); plt.close(fig)


def attention(read, out):
    rows = read("attention.csv")
    X = read("attention_extra.csv", required=False)
    if not X:
        print("  note: attention_extra.csv not found; Figure 6 drawn without per-channel keys")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    for ax, (mk, mn) in zip(axes, MODELS):
        R = [r for r in rows if r["model"] == mk and r["sink_fp16"] == "0"]
        for m, label in (("naive_rtn", "Round-to-nearest"), ("rotate_rtn", "Rotate + round"),
                         ("polarQuant", "PITCH PolarQuant"), ("turboQuant", "PITCH TurboQuant")):
            pts = sorted((f(r["ratio_vs_fp16"]), f(r["kl"])) for r in R if r["method"] == m and r["kv_mode"] == "both")
            ax.plot(*zip(*pts), color=C[m], marker="o", ms=3, lw=1.0, label=f"{label}, uniform $b$")
            mx = [(f(r["ratio_vs_fp16"]), f(r["kl"]), r["bits"]) for r in R if r["method"] == m and r["kv_mode"] == "mixed"]
            ax.scatter([x for x, _, _ in mx], [y for _, y, _ in mx], color=C[m], marker="x", s=16, lw=1.0)
            if m == "turboQuant":
                for x, y, lab in mx:
                    if lab in ("K8V3", "K4V2"):
                        # K4/V2 sits on the per-channel line at 4.57x, so it is labelled from
                        # empty space below-right with a thin pointer
                        off = (3, 3) if lab == "K8V3" else (14, -24)
                        ax.annotate(lab.replace("V", "/V"), (x, y), fontsize=6.5, xytext=off,
                                    textcoords="offset points", color=C[m],
                                    arrowprops=None if lab == "K8V3" else
                                    dict(arrowstyle="-", color=C[m], lw=0.6, shrinkA=0, shrinkB=2))
        pc = sorted((f(r["ratio_vs_fp16"]), f(r["kl_mean"])) for r in X
                    if r["model"] == mk and r["config"].startswith("kivi_G64_b"))
        if pc:
            ax.plot(*zip(*pc), color=C["perChannel"], marker="D", ms=3, lw=1.3,
                    label="PITCH per-channel keys, uniform $b$")
        fp = [r for r in R if r["method"] == "fp16"][0]
        ax.axhline(f(fp["kl"]), color="k", lw=0.7, ls=":", label="fp16 cache")
        ax.set_yscale("log"); ax.set_xlabel("Compression vs fp16"); ax.set_title(mn)
        ax.set_ylabel(r"KL$(p\,\|\,\hat p)$ per query")
    axes[0].scatter([], [], color="gray", marker="x", s=16, label="asymmetric K/V bits")
    axes[0].legend(frameon=False, loc="lower right", fontsize=6.3)
    fig.tight_layout(); fig.savefig(out / "attention_kl.pdf"); plt.close(fig)


def throughput(read, out):
    T = [r for r in read("throughput.csv") if r["api"] == "gpu_resident" and r["bits"] == "4"]
    P = [r for r in read("pytorch_throughput.csv") if r["bits"] == "4"]
    B = [1, 16, 256, 4096]
    pick = lambda rows, m, d, b, phase: f([r for r in rows if r["method"] == m and r["dim"] == str(d)
                                           and r["batch"] == str(b) and r["phase"] == phase][0]
                                          ["sustained_wall_us_per_call"]) * 1e3 / b
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    for ax, phase in zip(axes, ("encode", "decode")):
        for d, ls in ((64, "-"), (1024, "--")):
            for m in ("turboQuant", "polarQuant"):
                ax.plot(B, [pick(T, m, d, b, phase) for b in B], color=C[m], ls=ls, marker="o", ms=3, lw=1.0,
                        label=f"PITCH {'TurboQuant' if m == 'turboQuant' else 'PolarQuant'}, $d={d}$")
            ax.plot(B, [pick(P, "turboQuant", d, b, phase) for b in B], color=C["pytorch"], ls=ls, marker="s",
                    ms=3, lw=1.0, label=f"PyTorch/MPS TurboQuant, $d={d}$")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
        ax.set_xlabel("Vectors per dispatch (batch size)"); ax.set_title(f"{phase.capitalize()}, 4-bit")
    axes[0].set_ylabel("ns per vector (sustained wall)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(out / "throughput.pdf"); plt.close(fig)


FIGURES = {"theorem1": theorem1, "rate_distortion": rate_distortion, "attention": attention, "throughput": throughput}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", default=[Path("results"), Path("reference/results")],
                    help="folders to search for the CSVs, in order")
    ap.add_argument("--out", type=Path, default=Path("paper/images"))
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), help="regenerate only these figures")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    read = finder(a.results)
    for name in (a.only or FIGURES):
        try:
            FIGURES[name](read, a.out); print(f"wrote {a.out / (name if name != 'attention' else 'attention_kl')}.pdf")
        except Missing as e:
            print(f"skipped {name}: {e}")


if __name__ == "__main__":
    main()
