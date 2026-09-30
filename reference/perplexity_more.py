#!/usr/bin/env python3
"""Perplexity for the review additions, with per-window losses for confidence intervals.

Reuses perplexity_eval.py (unchanged) and adds:
  * MLX-style and llama.cpp-style KV quantisation, TurboQuant with the outlier split;
  * causal per-channel keys (KIVI residual, and PITCH's re-encode option), which need their
    own attention computation because a query's own group is treated differently;
  * results/perplexity_windows_<model>.csv: summed NLL per window, for bootstrap_ppl.py.

    python3 perplexity_more.py --model gpt2
    python3 perplexity_more.py --model Qwen/Qwen2.5-0.5B
    python3 perplexity_more.py --model Qwen/Qwen2.5-0.5B --configs turbo_b4 kivi_G64_b4
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
import perplexity_eval as P
from attention_extra import per_token_fp16
from attention_eval import log_softmax
import eval_additions as E

DEFAULT = ["exact", "fp16", "turbo_b4", "turbo_b3", "turbo_b8",
           "kivi_G64_b4", "kivi_G64_b3", "kivi_K4_turboV4",
           "mlx_b4", "mlx_b8", "q4_0", "q8_0", "turbo_split_b4", "turbo_split_b3",
           "exact_path", "kivi_full_path_b4", "kivi_full_path_b3",
           "kivi_causal_residual_b4", "kivi_causal_reencode_b4",
           "kivi_causal_residual_b3", "kivi_causal_reencode_b3"]


def make_configs(G):
    C = P.make_configs(G)
    for b in (4, 8):
        C[f"mlx_b{b}"] = (lambda X, b=b: E.per_token_groups_fp16(X, b),) * 2
    C["q4_0"] = (E.q4_0, E.q4_0)
    C["q8_0"] = (E.q8_0, E.q8_0)
    C["turbo_split_b4"] = (("head", lambda X: E.turbo_split(X, 5, 3)),) * 2
    C["turbo_split_b3"] = (("head", lambda X: E.turbo_split(X, 4, 2)),) * 2
    for b in (3, 4):
        for mode in ("residual", "reencode"):
            C[f"kivi_causal_{mode}_b{b}"] = (("causal", mode, b), lambda X, b=b: per_token_fp16(X, b))
    C["_causal_check"] = (("causal", "check", 4), None)
    C["exact_path"] = (("causal", "check", 4), None)
    for b in (3, 4):
        C[f"kivi_full_path_b{b}"] = (("causal", "full", b), lambda X, b=b: per_token_fp16(X, b))
    return C


def causal_attention(module, query, key, value, attention_mask, scaling=None, _float64=False, **kwargs):
    _, mode, bits = P.ACTIVE["k"]
    B, Hq, T, D = query.shape
    Hkv = key.shape[1]
    rep = Hq // Hkv
    scale = scaling if scaling is not None else D ** -0.5
    q = query.detach().to(torch.float64).cpu().numpy()
    k = key.detach().to(torch.float64).cpu().numpy()
    v = P._compress(value, P.ACTIVE["v"]).detach().to(torch.float64).cpu().numpy()
    out = np.empty((B, Hq, T, D))
    future = np.triu(np.ones((T, T), dtype=bool), 1)
    qpos = np.tile(np.arange(T), rep)
    for bi in range(B):
        for h in range(Hkv):
            Qh = q[bi, h * rep:(h + 1) * rep].reshape(rep * T, D)
            S = E.causal_kivi_scores(Qh, qpos, k[bi, h], bits, P.ACTIVE["G"], mode).reshape(rep, T, T) * scale
            S[:, future] = -np.inf
            Pm = np.exp(log_softmax(S))
            out[bi, h * rep:(h + 1) * rep] = Pm @ v[bi, h]
    o = torch.from_numpy(out)
    if not _float64:
        o = o.to(query.dtype).to(query.device)
    return o.transpose(1, 2).contiguous(), None


VERIFY = {"on": False, "worst": 0.0, "fp32": 0.0}


def dispatch(module, query, key, value, attention_mask, **kwargs):
    fk = P.ACTIVE["k"]
    if VERIFY["on"]:
        # This path computes in float64, so it is checked against PyTorch's attention run in
        # float64 on the same inputs; PyTorch's normal float32 result is returned, so later
        # layers see exactly what they would without the check. The float32-vs-float64
        # difference of PyTorch's own attention is recorded for information.
        ours, _ = causal_attention(module, query, key, value, attention_mask, _float64=True, **kwargs)
        m = attention_mask.double() if (attention_mask is not None and attention_mask.is_floating_point()) \
            else attention_mask
        ref64, _ = P.sdpa_attention_forward(module, query.double(), key.double(), value.double(), m, **kwargs)
        ref32, w = P.sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
        scale = ref64.abs().max().clamp_min(1e-30)
        VERIFY["worst"] = max(VERIFY["worst"], ((ours - ref64).abs().max() / scale).item())
        VERIFY["fp32"] = max(VERIFY["fp32"], ((ref32.double() - ref64).abs().max() / scale).item())
        return ref32, w
    if isinstance(fk, tuple) and fk[0] == "causal":
        return causal_attention(module, query, key, value, attention_mask, **kwargs)
    return P.pitch_attention(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("pitch_more", dispatch)


@torch.no_grad()
def window_nll(model, ids, window, n_windows, device):
    """Summed NLL and token count per window."""
    res = []
    for w in range(n_windows):
        chunk = ids[w * window:(w + 1) * window].unsqueeze(0).to(device)
        if chunk.shape[1] < 2:
            break
        n = chunk.shape[1] - 1
        res.append((model(chunk, labels=chunk).loss.item() * n, n))
    return res


def ppl(res):
    return math.exp(sum(s for s, _ in res) / sum(n for _, n in res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--configs", nargs="*", default=DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("results/perplexity_windows.csv"))
    a = ap.parse_args()
    P.ACTIVE["G"] = a.group
    device = "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = P.load_tokens(tok, a.window * a.windows + 1)

    ref_model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="sdpa").to(device).eval()
    ref = window_nll(ref_model, ids, a.window, a.windows, device)
    del ref_model
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="pitch_more").to(device).eval()
    configs = make_configs(a.group)

    P.ACTIVE["k"], P.ACTIVE["v"] = configs["_causal_check"]
    VERIFY["on"] = True
    window_nll(model, ids, a.window, 1, device)
    VERIFY["on"] = False
    print(f"PyTorch's own float32 attention differs from its float64 result by up to a relative "
          f"{VERIFY['fp32']:.1e} (for information)")
    if VERIFY["worst"] > 1e-9:
        raise SystemExit(f"causal path check failed: attention output differs from PyTorch's float64 "
                         f"attention by a relative {VERIFY['worst']:.2e} in some layer (expected below 1e-9)")
    print(f"causal path check passed: every layer matches PyTorch's float64 attention to a relative "
          f"{VERIFY['worst']:.1e}")

    unknown = set(a.configs) - set(configs)
    if unknown:
        raise SystemExit(f"unknown configs {sorted(unknown)}; choose from {sorted(k for k in configs if k[0] != '_')}")

    tag = a.model.replace("/", "_")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out = a.out.with_name(f"{a.out.stem}_{tag}{'' if a.group == 64 else f'_G{a.group}'}{a.out.suffix}")
    rows = [dict(model=a.model, config="exact", window=i, nll_sum=s, tokens=n) for i, (s, n) in enumerate(ref)]
    print(f"  {'exact':26s} ppl {ppl(ref):9.3f}")
    for name in a.configs:
        if name == "exact":
            continue
        P.ACTIVE["k"], P.ACTIVE["v"] = configs[name]
        res = window_nll(model, ids, a.window, a.windows, device)
        rows += [dict(model=a.model, config=name, window=i, nll_sum=s, tokens=n) for i, (s, n) in enumerate(res)]
        print(f"  {name:26s} ppl {ppl(res):9.3f}  ({100 * (ppl(res) / ppl(ref) - 1):+.2f}%)", flush=True)
        with open(out, "w", newline="") as fh:              # rewritten after each config, so a long run can be stopped
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
