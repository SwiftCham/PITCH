#!/usr/bin/env python3
import csv, statistics as st, sys

pitch_path, torch_path = sys.argv[1], sys.argv[2]
key = lambda r: (r["method"], r["bits"], r["dim"], r["batch"], r["phase"])
P = {key(r): r for r in csv.DictReader(open(pitch_path)) if r["api"] == "gpu_resident"}
T = {key(r): r for r in csv.DictReader(open(torch_path))}
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

print("\nTable 7: speed-up, median [min, max] over d and b\n")
print(f"{'':10s}" + "".join(f"{b:>20s}" for b in batches))
lines = []
for (m, ph), name in label.items():
    cells = []
    for b in batches:
        v = [ratio[k] for k in common if k[0] == m and k[4] == ph and k[3] == b]
        cells.append((st.median(v), min(v), max(v)))
    print(f"{name:10s}" + "".join(f"{c[0]:8.1f} [{c[1]:4.1f},{c[2]:5.1f}]" for c in cells))
    # LaTeX row in the paper's \spd{median}{min}{max} format
    lines.append(f"{name} & " + " & ".join(f"\\spd{{{c[0]:.1f}}}{{{c[1]:.1f}}}{{{c[2]:.1f}}}" for c in cells) + r" \\")

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

print("\nLaTeX rows for Table 7:")
print("\n".join(lines))
