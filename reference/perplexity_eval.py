#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import pitch_reference as R
from attention_extra import turbo, rotate_rtn, per_channel_groups, per_token_fp16, centred_turbo
from attention_eval import minmax_rtn

# ----------------------------------------------------------------------------- configurations
# name -> (key compressor, value compressor); each maps an (n, d) float64 array to its reconstruction.
def make_configs(G):
    T = lambda b: (lambda X: turbo(X, b))
    C = {"exact": (None, None), "fp16": ("fp16", "fp16")}
    for b in (2, 3, 4, 8):
        C[f"turbo_b{b}"] = (T(b), T(b))
    C["turbo_K8V3"] = (T(8), T(3))
    C["turbo_K4V2"] = (T(4), T(2))
    C["turbo_b5"] = (T(5), T(5))
    for b in (3, 4):
        C[f"rotate_rtn_b{b}"] = (lambda X, b=b: rotate_rtn(X, b),) * 2
        C[f"naive_rtn_b{b}"] = (lambda X, b=b: minmax_rtn(X, b),) * 2
    # per-head compressors are tagged ("head", fn): fn gets one head's (T, D) keys in token order
    for b in (3, 4):
        C[f"kivi_G{G}_b{b}"] = (("head", lambda X, b=b: per_channel_groups(X, b, G)),
                                lambda X, b=b: per_token_fp16(X, b))
        C[f"kivi_K{b}_turboV{b}"] = (("head", lambda X, b=b: per_channel_groups(X, b, G)), T(b))
    C["centred_turbo_b4"] = (("head", lambda X: centred_turbo(X, 4, 32)), T(4))
    return C

ACTIVE = {"k": None, "v": None, "G": 64}

def _compress(t, fn, per_channel=False):
    """t: (B, H, T, D) torch tensor. Per-vector compression on (B*H*T, D), or per-channel per head."""
    if fn is None:
        return t
    if fn == "fp16":
        return t.to(torch.float16).to(t.dtype)
    B, H, Tn, D = t.shape
    x = t.detach().to(torch.float64).cpu().numpy()
    if isinstance(fn, tuple) and fn[0] == "head":
        out = np.stack([np.stack([fn[1](x[b, h]) for h in range(H)]) for b in range(B)])
    else:
        out = fn(x.reshape(-1, D)).reshape(B, H, Tn, D)
    return torch.from_numpy(out).to(t.dtype).to(t.device)

def pitch_attention(module, query, key, value, attention_mask, **kwargs):
    key = _compress(key, ACTIVE["k"])
    value = _compress(value, ACTIVE["v"])
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

AttentionInterface.register("pitch", pitch_attention)

# ----------------------------------------------------------------------------- evaluation
def load_tokens(tok, n_tokens):
    from datasets import load_dataset
    # Newer huggingface_hub versions reject un-namespaced ids; "Salesforce/wikitext" is the canonical repo.
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[:n_tokens] if n_tokens else ids

@torch.no_grad()
def nll(model, ids, window, n_windows, device):
    total, count = 0.0, 0
    for w in range(n_windows):
        chunk = ids[w * window:(w + 1) * window].unsqueeze(0).to(device)
        if chunk.shape[1] < 2:
            break
        out = model(chunk, labels=chunk)
        n = chunk.shape[1] - 1
        total += out.loss.item() * n; count += n
    return total / count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=40, help="non-overlapping windows evaluated")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--configs", nargs="*", help="run only these configs (default: all)")
    ap.add_argument("--out", type=Path, default=Path("results/perplexity.csv"))
    a = ap.parse_args()
    ACTIVE["G"] = a.group
    device = "cpu"          # the compressors run in NumPy; CPU keeps the comparison exact and simple
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = load_tokens(tok, a.window * a.windows + 1)

    ref_model = AutoModelForCausalLM.from_pretrained(a.model,
                                                     attn_implementation="sdpa").to(device).eval()
    ref = nll(ref_model, ids, a.window, a.windows, device); del ref_model
    model = AutoModelForCausalLM.from_pretrained(a.model,
                                                 attn_implementation="pitch").to(device).eval()
    ACTIVE["k"] = ACTIVE["v"] = None
    hooked = nll(model, ids, a.window, a.windows, device)
    if abs(hooked - ref) > 1e-4:
        raise SystemExit(f"hook check failed: exact-through-hook NLL {hooked:.6f} != reference {ref:.6f}")
    print(f"hook check passed (NLL {ref:.6f}); {a.windows} windows of {a.window} tokens")

    rows = []
    configs = make_configs(a.group)
    if a.configs:
        unknown = set(a.configs) - set(configs)
        if unknown:
            raise SystemExit(f"unknown configs {sorted(unknown)}; choose from {sorted(configs)}")
        configs = {k: v for k, v in configs.items() if k in a.configs}
    for name, (fk, fv) in configs.items():
        ACTIVE["k"], ACTIVE["v"] = fk, fv
        loss = nll(model, ids, a.window, a.windows, device)
        rows.append(dict(model=a.model, config=name, nll=loss, ppl=math.exp(loss),
                         delta_ppl=math.exp(loss) - math.exp(ref)))
        print(f"  {name:18s} ppl {math.exp(loss):9.3f}  (+{math.exp(loss) - math.exp(ref):.3f})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tag = "_extra" if a.configs else ""
    out = a.out.with_name(f"{a.out.stem}_{a.model.replace('/', '_')}{tag}{a.out.suffix}")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
