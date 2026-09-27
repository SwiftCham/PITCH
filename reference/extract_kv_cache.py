#!/usr/bin/env python3
"""
Extract real KV-cache vectors from a transformer model for PITCH evaluation.

Saves reference/kv_vectors.json, then prints a baseline (naive uniform quantisation
without any rotation) so you can compare directly against `swift test` output.

Usage:
    pip install -r reference/requirements.txt
    python reference/extract_kv_cache.py                                   # GPT-2, no auth
    python reference/extract_kv_cache.py --model qwen --out reference/kv_vectors_qwen.json
    python reference/extract_kv_cache.py --model llama                     # needs HF login

Sampling: for every (text, layer), `--samples-per-layer` (head, token) pairs are drawn
uniformly at random, with a fixed seed, from ALL KV heads and ALL token positions, so
the data covers every head and position rather than the first tokens of head 0.
Keys are taken from the cache as stored, i.e. after RoPE for models that use it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SAMPLE_TEXTS = [
    "The transformer architecture introduced the attention mechanism as the primary"
    " building block for sequence modelling tasks in natural language processing.",
    "Apple Silicon uses a unified memory architecture where the CPU and GPU share"
    " the same physical memory pool, eliminating costly data transfers.",
    "Quantisation reduces the numerical precision of neural network weights and"
    " activations to compress model size and accelerate inference on-device.",
    "The key-value cache in autoregressive language models stores intermediate"
    " attention states so they do not need to be recomputed at each decoding step.",
    "Metal is Apple's low-level GPU programming framework, providing direct access"
    " to the GPU for both compute and graphics workloads on macOS and iOS.",
]

MODELS = {
    "gpt2":  "gpt2",
    "qwen":  "Qwen/Qwen2.5-0.5B",
    "llama": "meta-llama/Llama-3.2-1B",
    "phi":   "microsoft/Phi-3.5-mini-instruct",
}


# helpers

def is_power_of_two(n: int) -> bool:
    return n >= 2 and (n & (n - 1)) == 0


def naive_quantise(vec: np.ndarray, bits: int) -> np.ndarray:
    """Uniform min-max scalar quantisation — no rotation, pure baseline."""
    levels = (1 << bits) - 1
    lo, hi = float(vec.min()), float(vec.max())
    scale = (hi - lo) / levels if hi != lo else 1.0
    q = np.round((vec - lo) / scale).clip(0, levels)
    return q * scale + lo


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def to_layerwise_kv(past_kv):
    """
    Normalise whatever `out.past_key_values` is into a list of (key, value) tuples,
    one per layer. Handles, in order of preference:
      - newest Cache API: `.layers` is a list of DynamicLayer objects exposing
        `.keys` / `.values` tensor attributes (transformers >= ~4.44)
      - mid-generation `DynamicCache` objects via `.key_cache` / `.value_cache` lists
      - older Cache objects exposing `to_legacy_cache()`
      - plain legacy tuple-of-tuples format
    """
    if hasattr(past_kv, "layers"):
        return [(layer.keys, layer.values) for layer in past_kv.layers]

    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return list(zip(past_kv.key_cache, past_kv.value_cache))

    if hasattr(past_kv, "to_legacy_cache"):
        legacy = past_kv.to_legacy_cache()
        return [(layer[0], layer[1]) for layer in legacy]

    # Plain legacy format: tuple of (k, v) tuples
    return list(past_kv)


# extraction

def extract_vectors(model_name: str, texts: list[str], samples_per_layer: int, seed: int) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Running on: {device}")

    vectors: list[dict] = []
    rng = np.random.default_rng(seed)

    for text_idx, text in enumerate(texts):
        print(f"  [{text_idx + 1}/{len(texts)}] '{text[:60]}…'")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        seq_len = inputs["input_ids"].shape[1]

        layer_iter = to_layerwise_kv(out.past_key_values)

        for layer_idx, (k, v) in enumerate(layer_iter):
            # k / v shape: [batch=1, num_kv_heads, seq_len, head_dim]
            _, num_heads, _, head_dim = k.shape

            if not is_power_of_two(head_dim) or head_dim > 1024:
                continue

            # Uniform random (head, token) pairs over the whole layer, without replacement.
            total = num_heads * seq_len
            picks = rng.choice(total, size=min(samples_per_layer, total), replace=False)
            k_cpu = k[0].cpu().float().numpy()
            v_cpu = v[0].cpu().float().numpy()
            for flat in sorted(picks.tolist()):
                head_idx, tok_idx = divmod(flat, seq_len)
                base = {"layer": layer_idx, "head": int(head_idx),
                        "token": int(tok_idx), "text_idx": text_idx, "dim": int(head_dim)}
                vectors.append({**base, "type": "key", "data": k_cpu[head_idx, tok_idx].tolist()})
                vectors.append({**base, "type": "value", "data": v_cpu[head_idx, tok_idx].tolist()})

    return vectors


# baseline analysis

def print_baseline(vectors: list[dict]) -> None:
    print("\n── Baseline: naive uniform quantisation (no rotation) ──────────────")
    print(f"   {'bits':>4}  {'MSE':>10}  {'cosine sim':>12}  {'n':>6}")
    print("   " + "─" * 38)
    for bits in (3, 4, 8):
        mses, coss = [], []
        for v in vectors:
            orig = np.array(v["data"], dtype=np.float32)
            recon = naive_quantise(orig, bits)
            mses.append(mse(orig, recon))
            coss.append(cosine_sim(orig, recon))
        print(f"   {bits:>4}  {np.mean(mses):>10.6f}  {np.mean(coss):>12.6f}  {len(vectors):>6}")

    print()
    print("Run  swift test --filter KVEvaluation  to see PITCH results.")
    print("────────────────────────────────────────────────────────────────────\n")


# main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), default="gpt2",
                        help="Which model to extract from (default: gpt2)")
    parser.add_argument("--out", default="reference/kv_vectors.json",
                        help="Output JSON path (default: reference/kv_vectors.json)")
    parser.add_argument("--samples-per-layer", type=int, default=32,
                        help="Random (head, token) pairs per layer per text (default: 32)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vectors = extract_vectors(MODELS[args.model], SAMPLE_TEXTS, args.samples_per_layer, args.seed)

    print(f"\nExtracted {len(vectors)} vectors — saving to {out_path} …")
    payload = {"model": MODELS[args.model], "num_vectors": len(vectors),
               "sampling": {"samples_per_layer": args.samples_per_layer, "seed": args.seed},
               "vectors": vectors}
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"Saved {out_path}  ({size_kb:.0f} KB)")

    print_baseline(vectors)


if __name__ == "__main__":
    main()
