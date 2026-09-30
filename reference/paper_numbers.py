#!/usr/bin/env python3
"""Prints the numbers the paper quotes from the results CSVs.

    python3 paper_numbers.py matched [results/rate_distortion.csv]
        Error relative to rotate-and-round at equal stored bits (Table 3 and Section 6.2).

    python3 paper_numbers.py speedup results/throughput.csv results/pytorch_throughput.csv
        Speed-up of PITCH over PyTorch/MPS, single-call costs and dispatch overhead (Section 6.5).

    python3 paper_numbers.py perchannel [results/per_channel_throughput.csv]
        Per-channel kernel timings against TurboQuant (Section 6.5, "Per-channel keys").
"""
import argparse, csv, statistics as st, sys
import numpy as np

def matched(a):
    REF = "rotate_rtn"
    ORDER = ["naive_rtn", "pitch_turboquant", "pitch_polarquant", "pitch_polarquant_seed_per_vector",
             "polar_recursive_prototype"]
    rows = list(csv.DictReader(open(a.csv or "results/rate_distortion.csv")))

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
    print("1) error / rotate-and-round error at equal stored bits (Table 3)")
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

def speedup(a):
    key = lambda r: (r["method"], r["bits"], r["dim"], r["batch"], r["phase"])
    P = {key(r): r for r in csv.DictReader(open(a.pitch)) if r["api"] == "gpu_resident"}
    T = {key(r): r for r in csv.DictReader(open(a.torch))}
    common = sorted(set(P) & set(T))
    if not common:
        sys.exit("no matching configurations between the two files")
    missing = len(T) - len(common)
    if missing:
        print(f"note: {missing} PyTorch rows have no PITCH counterpart")

    ratio = {k: float(T[k]["sustained_wall_us_per_call"]) / float(P[k]["sustained_wall_us_per_call"]) for k in common}
    batches = sorted({k[3] for k in common}, key=int)
    label = {("turboQuant", "encode"): "TQ encode", ("turboQuant", "decode"): "TQ decode",
             ("polarQuant", "encode"): "PQ encode", ("polarQuant", "decode"): "PQ decode"}

    print("\nspeed-up, median [min, max] over d and b\n")
    print(f"{'':10s}" + "".join(f"{b:>20s}" for b in batches))
    for (m, ph), name in label.items():
        cells = []
        for b in batches:
            v = [ratio[k] for k in common if k[0] == m and k[4] == ph and k[3] == b]
            cells.append((st.median(v), min(v), max(v)))
        print(f"{name:10s}" + "".join(f"{c[0]:8.1f} [{c[1]:4.1f},{c[2]:5.1f}]" for c in cells))

    allr = list(ratio.values())
    print(f"\nsustained speed-up over all configurations: {min(allr):.1f}-{max(allr):.1f}x")

    b1 = [k for k in common if k[3] == "1"]
    for ph in ("encode", "decode"):
        pw = st.median(float(P[k]["wall_us_median"]) for k in b1 if k[4] == ph)
        tw = st.median(float(T[k]["wall_us_median"]) for k in b1 if k[4] == ph)
        print(f"single call {ph}: PITCH {pw:.0f} us vs PyTorch {tw:.0f} us ({tw / pw:.1f}x)")
    po = st.median(float(P[k]["sustained_wall_us_per_call"]) for k in b1)
    to = st.median(float(T[k]["sustained_wall_us_per_call"]) for k in b1)
    print(f"sustained per-dispatch overhead at batch 1: PITCH {po:.1f} us vs PyTorch {to:.1f} us")


def perchannel(a):
    rows = list(csv.DictReader(open(a.csv or "results/per_channel_throughput.csv")))
    get = lambda m, b, d, n, ph, col: float(next(r[col] for r in rows if (r["method"], r["bits"], r["dim"], r["tokens"], r["phase"])
                                                  == (m, str(b), str(d), str(n), ph)))
    dims = sorted({int(r["dim"]) for r in rows})

    print("ns per token, sustained GPU time, 4,096 tokens, 4 bits")
    print(f"{'d':>6} {'PC enc':>9} {'TQ enc':>9} {'ratio':>7} {'PC dec':>9} {'TQ dec':>9} {'ratio':>7}")
    for d in dims:
        pe, te = get("perChannel", 4, d, 4096, "encode", "ns_per_token_sustained_gpu"), get("turboQuant", 4, d, 4096, "encode", "ns_per_token_sustained_gpu")
        pd, td = get("perChannel", 4, d, 4096, "decode", "ns_per_token_sustained_gpu"), get("turboQuant", 4, d, 4096, "decode", "ns_per_token_sustained_gpu")
        print(f"{d:>6} {pe:>9.2f} {te:>9.2f} {pe / te:>7.2f} {pd:>9.2f} {td:>9.2f} {pd / td:>7.2f}")

    print("\nbit-width dependence at d = 64, 4,096 tokens (ns per token, sustained GPU)")
    for ph in ("encode", "decode"):
        print(f"  {ph}: " + ", ".join(f"{b} bits {get('perChannel', b, 64, 4096, ph, 'ns_per_token_sustained_gpu'):.2f}" for b in (2, 4, 8)))

    print("\nappending one completed group (64 tokens), 4 bits")
    for d in (64, 128):
        if d in dims:
            print(f"  d={d}: GPU {get('perChannel', 4, d, 64, 'encode', 'gpu_us_median'):.1f} us single call, "
                  f"{get('perChannel', 4, d, 64, 'encode', 'sustained_gpu_us_per_call'):.1f} us sustained; "
                  f"wall {get('perChannel', 4, d, 64, 'encode', 'wall_us_median'):.0f} us single call")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("matched"); p.add_argument("csv", nargs="?")
    p = sub.add_parser("speedup"); p.add_argument("pitch"); p.add_argument("torch")
    p = sub.add_parser("perchannel"); p.add_argument("csv", nargs="?")
    a = ap.parse_args()
    {"matched": matched, "speedup": speedup, "perchannel": perchannel}[a.what](a)


if __name__ == "__main__":
    main()
