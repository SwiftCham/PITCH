#!/usr/bin/env python3
import argparse, base64, json
from pathlib import Path
import numpy as np
import pitch_reference as R

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Tests" / "PITCHTests" / "Fixtures" / "parity.json"

def f32(a):  return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", type=Path, default=HERE / "kv_vectors_qwen.json")
    args = ap.parse_args()
    rng = np.random.default_rng(2026)

    groups = {}
    if args.kv.exists():
        vecs = json.loads(args.kv.read_text())["vectors"]
        keys = [v["data"] for v in vecs if v["type"] == "key"][:16]
        vals = [v["data"] for v in vecs if v["type"] == "value"][:16]
        groups["kv64"] = np.array(keys + vals, dtype=np.float32)
        kv_source = str(args.kv.name)
    else:  # Gaussian with a few outlier channels, so the fixture never depends on the data files
        g = rng.standard_normal((32, 64)); g[:, :4] *= 8
        groups["kv64"] = g.astype(np.float32)
        kv_source = "synthetic (kv file not found)"
    groups["gauss2"] = rng.standard_normal((8, 2)).astype(np.float32)
    groups["gauss16"] = rng.standard_normal((8, 16)).astype(np.float32)
    groups["gauss256"] = rng.standard_normal((8, 256)).astype(np.float32)
    groups["gauss1024"] = rng.standard_normal((2, 1024)).astype(np.float32)

    cases = []
    for gname, X32 in groups.items():
        X = X32.astype(np.float64)                  # the kernel sees exactly these float32 values
        d = X.shape[1]
        for method in ("turboQuant", "polarQuant"):
            for bits in (2, 3, 4, 8):
                seed = 12345 if (gname == "gauss16" and bits == 4) else R.DEFAULT_SEED
                codes, scales = R.ENCODERS[method](X, bits, seed)
                scales32 = np.asarray(scales, dtype=np.float32)
                recon = R.DECODERS[method](codes, scales32.astype(np.float64), bits, seed)
                rel = float((((X - recon) ** 2).sum(1) / (X ** 2).sum(1)).mean())
                cases.append(dict(group=gname, method=method, bits=bits, seed=seed,
                                  codes=base64.b64encode(R.pack_codes(codes, bits)).decode(),
                                  scales=f32(scales32), reconstruction=f32(recon),
                                  relative_error=rel))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        generator="reference/gen_parity_fixture.py", kv_source=kv_source,
        groups=[dict(name=n, dim=int(X.shape[1]), count=int(X.shape[0]), input=f32(X)) for n, X in groups.items()],
        cases=cases), indent=1))
    print(f"wrote {len(cases)} cases over {len(groups)} groups to {OUT} ({OUT.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
