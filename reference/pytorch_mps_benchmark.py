#!/usr/bin/env python3
import argparse, csv, math, statistics, time
from pathlib import Path
import numpy as np
import torch
import pitch_reference as R

DIMS = [64, 128, 256, 512, 1024]
BATCHES = [1, 16, 256, 4096]
BITS = [2, 4, 8]
REPEATS = 100
SUSTAINED_CALLS = 16
SUSTAINED_REPEATS = 20
WARMUP_SECONDS = 0.25

def hadamard(d, device):
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H.to(device)

class TorchCodec:
    def __init__(self, method, dim, bits, device, seed=R.DEFAULT_SEED):
        self.method, self.dim, self.bits, self.device = method, dim, bits, device
        self.signs = torch.tensor(R.rand_signs(seed, dim), dtype=torch.float32, device=device)
        self.H = hadamard(dim, device) / math.sqrt(dim)      # symmetric and orthogonal: H^-1 = H
        if 32 % bits != 0 or (dim * bits) % 32 != 0:
            raise ValueError("benchmark packing supports bits dividing 32 with dim * bits a multiple of 32")
        self.words = dim * bits // 32
        self.per_word = 32 // bits
        # Code j of a word sits at bit j*bits: pack/unpack by integer multiply / floor-divide,
        # which every MPS build supports (no int64 bit-shift ops needed).
        self.place = (2 ** (torch.arange(self.per_word, device=device) * bits)).to(torch.int64)
        if method == "turboQuant":
            cb = R.turbo_codebook(dim, bits).astype(np.float32)
            self.codebook = torch.tensor(cb, device=device)
            self.edges = (0.5 * (self.codebook[1:] + self.codebook[:-1])).contiguous()

    def rotate(self, X):   return (X * self.signs) @ self.H
    def unrotate(self, Y): return (Y @ self.H) * self.signs

    def pack(self, codes):                          # (n, d) int64 -> (n, words) int64 holding uint32 values
        n = codes.shape[0]
        return (codes.view(n, self.words, self.per_word) * self.place).sum(dim=2)   # fields never overlap

    def unpack(self, words):
        n = words.shape[0]
        return torch.remainder(torch.div(words[:, :, None], self.place, rounding_mode="floor"),
                               2 ** self.bits).reshape(n, self.dim)

    def encode(self, X):
        if self.method == "turboQuant":
            norms = X.norm(dim=1)
            U = self.rotate(X / torch.where(norms > 0, norms, torch.ones_like(norms))[:, None])
            codes = torch.searchsorted(self.edges, U.contiguous(), right=False)
            return self.pack(codes), norms
        Y = self.rotate(X)
        a, c = Y[:, 0::2], Y[:, 1::2]
        r = torch.sqrt(a * a + c * c)
        rmax = r.amax(dim=1)
        safe = torch.where(rmax > 0, rmax, torch.ones_like(rmax))[:, None]
        RL, N = 2 ** self.bits - 1, 2 ** self.bits
        rq = torch.clamp(torch.floor(r / safe * RL + 0.5), 0, RL)
        theta = torch.remainder(torch.atan2(c, a), 2 * math.pi)
        tq = torch.remainder(torch.floor(theta / (2 * math.pi) * N + 0.5), N)
        codes = torch.stack([rq, tq], dim=2).reshape(X.shape[0], self.dim).long()
        return self.pack(codes), rmax

    def decode(self, words, scales):
        codes = self.unpack(words)
        if self.method == "turboQuant":
            return self.unrotate(self.codebook[codes]) * scales[:, None]
        RL, N = 2 ** self.bits - 1, 2 ** self.bits
        r = codes[:, 0::2].float() / RL * scales[:, None]
        theta = codes[:, 1::2].float() / N * (2 * math.pi)
        Y = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=2).reshape(-1, self.dim)
        return self.unrotate(Y)

def sync(device):
    if device.type == "mps": torch.mps.synchronize()

def parity_check(device):
    """Refuse to benchmark unless this implementation matches pitch_reference.py."""
    rng = np.random.default_rng(0)
    for method in ("turboQuant", "polarQuant"):
        for dim in (64, 1024):
            for bits in BITS:
                X = rng.standard_normal((32, dim)).astype(np.float32)
                ref_codes, ref_scales = R.ENCODERS[method](X.astype(np.float64), bits)
                codec = TorchCodec(method, dim, bits, device)

                words, _ = codec.encode(torch.tensor(X, device=device))
                ours = R.unpack_codes(words.cpu().numpy().astype("<u4").tobytes(), 32, dim, bits)
                mismatch = (ours != ref_codes).mean()

                ref_words = np.frombuffer(R.pack_codes(ref_codes, bits), dtype="<u4").astype(np.int64).reshape(32, -1)
                recon = codec.decode(torch.tensor(ref_words, device=device),
                                     torch.tensor(ref_scales, dtype=torch.float32, device=device)).cpu().numpy()
                ref_recon = R.DECODERS[method](ref_codes, ref_scales.astype(np.float32).astype(np.float64), bits)
                rel = np.linalg.norm(recon - ref_recon) / np.linalg.norm(ref_recon)
                if mismatch > 0.002 or rel > 1e-4:
                    raise SystemExit(f"parity FAILED {method} d={dim} b={bits}: "
                                     f"{mismatch:.4%} codes differ, decode rel diff {rel:.2e}")
    print("parity check passed: PyTorch codecs match pitch_reference.py")

def time_calls(fn, device):
    """Returns (single-call seconds list, sustained seconds-per-call list)."""
    warm_until = time.perf_counter() + WARMUP_SECONDS
    while time.perf_counter() < warm_until:
        for _ in range(SUSTAINED_CALLS): fn()
        sync(device)
    single = []
    for _ in range(REPEATS):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        single.append(time.perf_counter() - t0)
    sustained = []
    for _ in range(SUSTAINED_REPEATS):
        sync(device)
        t0 = time.perf_counter()
        for _ in range(SUSTAINED_CALLS): fn()
        sync(device)
        sustained.append((time.perf_counter() - t0) / SUSTAINED_CALLS)
    return single, sustained

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("results/pytorch_throughput.csv"))
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"torch {torch.__version__} on {device}")
    parity_check(device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dim in DIMS:
        for batch in BATCHES:
            X = torch.randn(batch, dim, device=device)
            for method in ("turboQuant", "polarQuant"):
                for bits in BITS:
                    codec = TorchCodec(method, dim, bits, device)
                    words, scales = codec.encode(X)
                    enc = time_calls(lambda: codec.encode(X), device)
                    dec = time_calls(lambda: codec.decode(words, scales), device)
                    for phase, (single, sustained) in (("encode", enc), ("decode", dec)):
                        med = statistics.median(single)
                        q = statistics.quantiles(single, n=10)
                        smed = statistics.median(sustained)
                        rows.append(dict(method=method, bits=bits, dim=dim, batch=batch, phase=phase,
                                         api=f"pytorch_{device.type}", wall_us_median=med * 1e6,
                                         wall_us_p10=q[0] * 1e6, wall_us_p90=q[-1] * 1e6,
                                         sustained_wall_us_per_call=smed * 1e6,
                                         ns_per_vector_sustained=smed * 1e9 / batch,
                                         input_gb_per_s_sustained=batch * dim * 4 / smed / 1e9))
        print(f"  d={dim} done")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
