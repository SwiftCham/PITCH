#!/usr/bin/env python3
"""Mechanism experiments for the paper's key-bias explanation.

1. Exact bias removal ("*_debias").  Qwen's key projection adds a learned bias b_k before RoPE,
   so a cached key is k_t = R_t (W x_t + b_k) = R_t W x_t + R_t b_k.  R_t b_k is known exactly
   (b_k is a weight, R_t is deterministic), so it can be subtracted before quantization and added
   back after decoding at zero storage cost.  If per-vector methods recover, the bias is the cause.

2. Offset removal + TurboQuant ("hybrid_*").  Per-channel means over groups of G tokens are
   subtracted (stored in fp16: 16/G bits per coordinate), TurboQuant compresses the remainder,
   and the means are added back.  This separates "remove per-channel offsets" from "quantize per
   channel": if the hybrid matches per-channel keys, offset removal is what matters.

    python3 mechanism_eval.py --model Qwen/Qwen2.5-0.5B
    python3 mechanism_eval.py --model Qwen/Qwen2.5-1.5B --configs exact turbo_b4 turbo_debias_b4 hybrid_turbo_b4 kivi_G64_b4
    python3 bootstrap_ppl.py results/mechanism_windows_Qwen_Qwen2.5-0.5B.csv --pairs <see PAIRS below>
"""
import argparse, csv, importlib, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
import perplexity_eval as P
from attention_extra import turbo, per_token_fp16
import eval_additions as E

F16 = lambda a: np.asarray(a, dtype=np.float64).astype(np.float16).astype(np.float64)

DEFAULT = ["exact", "turbo_b4", "turbo_b3", "centred_turbo_b4", "kivi_G64_b4", "q4_0", "mlx_b4",
           "turbo_b8", "q8_0", "mlx_b8",                  # float32 check of the paper's bf16 8-bit results
           "turbo_debias_b4", "turbo_debias_b3", "q4_0_debias", "mlx_debias_b4",
           "hybrid_turbo_b4", "hybrid_turbo_b4_G128", "hybrid_debias_turbo_b4"]
PAIRS = ["turbo_debias_b4:turbo_b4", "turbo_debias_b4:kivi_G64_b4", "turbo_debias_b4:centred_turbo_b4",
         "q4_0_debias:q4_0", "mlx_debias_b4:mlx_b4", "hybrid_turbo_b4:kivi_G64_b4",
         "hybrid_turbo_b4:centred_turbo_b4", "hybrid_debias_turbo_b4:kivi_G64_b4"]


def offset_turbo(X, b, G):
    """Per-channel fp16 means over groups of G tokens removed, TurboQuant on the remainder."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.empty_like(X)
    for s in range(0, X.shape[0], G):
        blk = X[s:s + G]
        mu = F16(blk.mean(0))
        Y[s:s + G] = turbo(blk - mu, b) + mu
    return Y


def key_bits(name, d):
    """Stored bits per coordinate for keys (values keep their usual cost)."""
    if name.startswith("hybrid"):
        G = 128 if name.endswith("G128") else 64
        return 4 + 32 / d + 16 / G
    return None


def make_configs(G):
    C = P.make_configs(G)
    T = lambda b: (lambda X: turbo(X, b))
    C["q4_0"] = (E.q4_0, E.q4_0)
    C["q8_0"] = (E.q8_0, E.q8_0)
    C["mlx_b8"] = (lambda X: E.per_token_groups_fp16(X, 8),) * 2
    C["mlx_b4"] = (lambda X: E.per_token_groups_fp16(X, 4),) * 2
    # exact bias removal: ("debias", key quantizer); values as in the base configuration
    for b in (3, 4):
        C[f"turbo_debias_b{b}"] = (("debias", T(b)), T(b))
    C["q4_0_debias"] = (("debias", E.q4_0), E.q4_0)
    C["mlx_debias_b4"] = (("debias", lambda X: E.per_token_groups_fp16(X, 4)),
                          lambda X: E.per_token_groups_fp16(X, 4))
    # offset removal + TurboQuant
    C["hybrid_turbo_b4"] = (("head", lambda X: offset_turbo(X, 4, 64)), T(4))
    C["hybrid_turbo_b4_G128"] = (("head", lambda X: offset_turbo(X, 4, 128)), T(4))
    C["hybrid_debias_turbo_b4"] = (("debias", ("head", lambda X: offset_turbo(X, 4, 64))), T(4))
    C["_check"] = (("debias", None), None)
    return C


MODEL = {"m": None}
STASH, CHECK = {}, {"on": False, "worst_rope": 0.0, "worst_debias": 0.0}


def rope_parts(module, T, dtype, device):
    """The model's own cos/sin for positions 0..T-1, and its rotate_half."""
    pos = torch.arange(T, device=device)[None]
    cos, sin = MODEL["m"].model.rotary_emb(torch.zeros(1, T, module.head_dim, dtype=dtype, device=device), pos)
    rotate_half = importlib.import_module(type(module).__module__).rotate_half
    return cos[:, None], sin[:, None], rotate_half           # (1, 1, T, D)


def rotated_bias(module, key):
    b = module.k_proj.bias
    if b is None:
        raise SystemExit("this model has no key-projection bias, so the *_debias configurations do not apply")
    B, Hkv, T, D = key.shape
    cos, sin, rh = rope_parts(module, T, key.dtype, key.device)
    bb = b.detach().view(1, Hkv, 1, D).to(key.dtype)
    return bb * cos + rh(bb) * sin                             # R_t b_k, (1, Hkv, T, D)


def attention(module, query, key, value, attention_mask, **kwargs):
    fk, fv = P.ACTIVE["k"], P.ACTIVE["v"]
    if CHECK["on"]:
        B, Hkv, T, D = key.shape
        pre = STASH[id(module.k_proj)].view(B, T, Hkv, D).transpose(1, 2)   # raw k_proj output, with bias
        cos, sin, rh = rope_parts(module, T, key.dtype, key.device)
        scale = key.abs().max().clamp_min(1e-30)
        rope_pre = pre * cos + rh(pre) * sin                   # must equal the model's own post-RoPE key
        CHECK["worst_rope"] = max(CHECK["worst_rope"], ((rope_pre - key).abs().max() / scale).item())
        nob = pre - module.k_proj.bias.view(1, Hkv, 1, D)      # W x without the bias
        want = nob * cos + rh(nob) * sin                       # R_t W x
        CHECK["worst_debias"] = max(CHECK["worst_debias"],
                                    ((key - rotated_bias(module, key) - want).abs().max() / scale).item())
    if isinstance(fk, tuple) and fk[0] == "debias":
        Rb = rotated_bias(module, key)
        k_hat = P._compress(key - Rb, fk[1]) + Rb
        v_hat = P._compress(value, fv)
        return P.sdpa_attention_forward(module, query, k_hat, v_hat, attention_mask, **kwargs)
    return P.pitch_attention(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("pitch_mech", attention)


@torch.no_grad()
def window_nll(model, ids, window, n_windows):
    res = []
    for w in range(n_windows):
        chunk = ids[w * window:(w + 1) * window].unsqueeze(0)
        if chunk.shape[1] < 2:
            break
        n = chunk.shape[1] - 1
        res.append((model(chunk, labels=chunk).loss.item() * n, n))
    return res


def ppl(res):
    return math.exp(sum(s for s, _ in res) / sum(n for _, n in res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--configs", nargs="*", default=DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("results/mechanism_windows.csv"))
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="model precision; recent transformers otherwise load a model in its configured "
                         "precision (bfloat16 for Qwen and Llama), which is too coarse for the self-check")
    a = ap.parse_args()
    P.ACTIVE["G"] = a.group
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = P.load_tokens(tok, a.window * a.windows + 1)
    import transformers
    dt = getattr(torch, a.dtype)
    def load(impl):
        try:
            return AutoModelForCausalLM.from_pretrained(a.model, attn_implementation=impl, dtype=dt).eval()
        except TypeError:                                   # older transformers
            return AutoModelForCausalLM.from_pretrained(a.model, attn_implementation=impl, torch_dtype=dt).eval()
    ref_model = load("sdpa")
    print(f"transformers {transformers.__version__}; model loaded in {ref_model.dtype}")
    ref = window_nll(ref_model, ids, a.window, a.windows)
    del ref_model
    model = load("pitch_mech")
    MODEL["m"] = model
    configs = make_configs(a.group)
    unknown = set(a.configs) - set(configs)
    if unknown:
        raise SystemExit(f"unknown configs {sorted(unknown)}")

    # Self-check: capture raw k_proj outputs and verify the RoPE and the rotated bias in every layer.
    layers = model.model.layers
    if layers[0].self_attn.k_proj.bias is None:
        print("note: no key-projection bias in this model; only the hybrid configurations are meaningful")
        a.configs = [c for c in a.configs if "debias" not in c]
    else:
        hooks = [l.self_attn.k_proj.register_forward_hook(lambda m, i, o: STASH.__setitem__(id(m), o))
                 for l in layers]
        P.ACTIVE["k"], P.ACTIVE["v"] = configs["_check"]
        CHECK["on"] = True
        check = window_nll(model, ids, a.window, 1)
        CHECK["on"] = False
        for h in hooks:
            h.remove()
        STASH.clear()
        dnll = abs(check[0][0] / check[0][1] - ref[0][0] / ref[0][1])
        print(f"self-check: RoPE reproduces the model's keys to {CHECK['worst_rope']:.1e}, "
              f"key - R_t b_k equals R_t W x to {CHECK['worst_debias']:.1e} (relative), "
              f"identity debias changes NLL by {dnll:.1e}")
        tol = 1e-5 if a.dtype == "float32" else 3e-2           # bfloat16 rounds at 2^-8 = 3.9e-3
        if CHECK["worst_rope"] > tol or CHECK["worst_debias"] > tol or dnll > 1e-4:
            raise SystemExit("self-check failed: the bias subtraction does not match this model; no results written")

    d = model.config.hidden_size // model.config.num_attention_heads
    d = getattr(model.config, "head_dim", None) or d
    tag = a.model.replace("/", "_")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out = a.out.with_name(f"{a.out.stem}_{tag}{a.out.suffix}")
    rows = [dict(model=a.model, config="exact", window=i, nll_sum=s, tokens=n) for i, (s, n) in enumerate(ref)]
    print(f"  {'exact':26s} ppl {ppl(ref):9.3f}")
    for name in a.configs:
        if name == "exact":
            continue
        P.ACTIVE["k"], P.ACTIVE["v"] = configs[name]
        res = window_nll(model, ids, a.window, a.windows)
        rows += [dict(model=a.model, config=name, window=i, nll_sum=s, tokens=n) for i, (s, n) in enumerate(res)]
        kb = key_bits(name, d)
        note = f"   [keys {kb:.3f} bits/coord]" if kb else ""
        print(f"  {name:26s} ppl {ppl(res):9.3f}  ({100 * (ppl(res) / ppl(ref) - 1):+.2f}%){note}", flush=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}\nfor intervals: python3 bootstrap_ppl.py {out} --pairs {' '.join(PAIRS)}")


if __name__ == "__main__":
    main()
