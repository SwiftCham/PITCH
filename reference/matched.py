#!/usr/bin/env python3
import csv, sys
import numpy as np

REF = "rotate_rtn"
ORDER = ["naive_rtn", "pitch_turboquant", "pitch_polarquant", "pitch_polarquant_seed_per_vector",
         "legacy_turbo_minmax_residual", "polar_recursive_prototype"]

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "results/rate_distortion.csv")))

def curve(model, kind, method):
    return np.array(sorted((float(r["bits_per_coord"]), float(r["rel_err"])) for r in rows
                           if r["model"] == model and r["kind"] == kind and r["method"] == method))

def at(model, kind, method, bpc):
    """Log-linear interpolation of a method's error at a given stored-bit rate (nan outside range)."""
    c = curve(model, kind, method)
    if len(c) == 0 or bpc < c[0, 0] - 1e-9 or bpc > c[-1, 0] + 1e-9:
        return np.nan
    return float(np.exp(np.interp(bpc, c[:, 0], np.log(c[:, 1]))))

models = list(dict.fromkeys(r["model"] for r in rows))
print("1) error / rotate-and-round error at equal stored bits (Table 5)")
for model in models:
    for kind in ("key", "value"):
        print(f"\n  {model} {kind}")
        for m in ORDER:
            pts = [(bpc, err / at(model, kind, REF, bpc)) for bpc, err in curve(model, kind, m)]
            used = [(b, r) for b, r in pts if not np.isnan(r)]
            if not used:
                continue
            per = "  ".join(f"{b:.2f}b:{r:.2f}" for b, r in used)
            print(f"    {m:34s} mean {np.mean([r for _, r in used]):5.2f}   [{per}]")

print("\n2) earlier TurboQuant variant / current TurboQuant, at the earlier variant's stored bits")
for model in models:
    for kind in ("key", "value"):
        ratios = []
        for bpc, err in curve(model, kind, "legacy_turbo_minmax_residual"):
            tq = at(model, kind, "pitch_turboquant", bpc)
            if not np.isnan(tq):
                ratios.append((bpc, err / tq))
        txt = "  ".join(f"{b:.2f}b:{r:.1f}x" for b, r in ratios)
        print(f"  {model:22s} {kind:5s} {txt}")
