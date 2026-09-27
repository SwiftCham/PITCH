#!/usr/bin/env python3
import argparse, base64, json, sys
from pathlib import Path
import numpy as np
import torch

MODELS = {
    "gpt2":  "gpt2",
    "qwen":  "Qwen/Qwen2.5-0.5B",
    "llama": "meta-llama/Llama-3.2-1B",
}

# Three original passages of roughly 250-350 tokens, so attention spans a realistic context.
TEXTS = [
    "Lighthouses were once the most important pieces of maritime infrastructure along any coast. "
    "Before satellite navigation, a ship approaching land at night relied on a sequence of lights, each "
    "with its own rhythm of flashes, to work out exactly where it was. Keepers lived in small cottages at "
    "the base of the tower and climbed the stairs several times a night to trim the wick, polish the lens "
    "and wind the clockwork that turned it. The great glass lenses, built from rings of prisms, could bend "
    "the light of a single lamp into a beam visible more than twenty miles away. Storms were the hardest "
    "part of the job. Spray could reach the lantern room of a rock station, and supply boats sometimes "
    "could not land for weeks, so keepers kept careful stores of oil, food and fresh water. Over the last "
    "century almost every light was automated, the keepers left, and many of the cottages became museums "
    "or holiday homes. The towers still stand, however, and most of them still flash their old patterns "
    "each night, even though few ships now depend on them. For many coastal towns they have become a "
    "symbol of the place itself, printed on postcards, painted on signs, and visited by people who have "
    "never needed a light to find their way home.",
    "Bread dough is a small ecosystem. When flour and water are mixed, enzymes in the flour begin breaking "
    "starch into simpler sugars, and yeast feeds on those sugars, releasing carbon dioxide and alcohol. The "
    "gas is trapped by a stretchy network of gluten proteins that forms as the dough is kneaded or simply "
    "left to rest, which is why a well developed dough can hold large bubbles without tearing. Temperature "
    "controls the pace of everything. Warm dough rises quickly but develops less flavour, while a long, cool "
    "fermentation gives bacteria time to produce acids and aromatic compounds that make the crust and crumb "
    "taste more complex. Sourdough takes this further by relying on a culture of wild yeasts and lactic acid "
    "bacteria kept alive with regular feeding. Bakers learn to read the dough by touch: how quickly it springs "
    "back when pressed, how it smells, and how much it has grown. In the oven, the trapped gas expands, the "
    "yeast dies, the proteins set, and sugars on the surface brown through reactions that create hundreds of "
    "new flavour compounds. A loaf that looks simple is really the result of hours of chemistry and biology "
    "working together, guided by a baker who mostly decides when to wait.",
    "A river changes character along its length. Near its source it is usually narrow, cold and fast, cutting "
    "down into rock and carrying stones that grind the channel deeper during every flood. Few plants can root "
    "in the moving gravel, and the insects that live there cling to the undersides of stones or build small "
    "cases to anchor themselves against the current. Further downstream the valley widens, the slope eases, "
    "and the river begins to wander, eroding the outside of each bend while depositing sand and silt on the "
    "inside. These meanders slowly migrate across the valley floor, sometimes cutting through a narrow neck "
    "and leaving behind a curved lake. The slower water is warmer and richer in nutrients, so plants grow "
    "along the banks and fish that prefer calmer water become common. Near the sea the river may split into "
    "many channels that spread across a delta of its own sediment, where fresh and salt water mix with each "
    "tide. Each stretch supports a different community of life, and changes made in one place, such as a dam "
    "or a new channel, can alter the flow of water and sediment for everything downstream.",
]

def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()

def cache_layers(past):
    """(key, value) per layer from any transformers cache format."""
    if hasattr(past, "layers"):
        return [(l.keys, l.values) for l in past.layers]
    if hasattr(past, "key_cache"):
        return list(zip(past.key_cache, past.value_cache))
    if hasattr(past, "to_legacy_cache"):
        return [(l[0], l[1]) for l in past.to_legacy_cache()]
    return list(past)

class QueryCapture:
    """Recomputes each layer's attention queries (and keys, for the self-check) as the model uses them."""

    def __init__(self, model):
        self.q, self.k, self.handles = {}, {}, []
        cfg = model.config
        if cfg.model_type == "gpt2":
            self.heads, self.kv_heads = cfg.n_head, cfg.n_head
            self.dim = cfg.n_embd // cfg.n_head
            for i, block in enumerate(model.transformer.h):
                self.handles.append(block.attn.c_attn.register_forward_hook(self._gpt2(i, cfg)))
        else:
            self.heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
            attn0 = model.model.layers[0].self_attn
            self.dim = getattr(attn0, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            for i, layer in enumerate(model.model.layers):
                self.handles.append(layer.self_attn.register_forward_pre_hook(
                    self._rope(i, layer.self_attn), with_kwargs=True))

    def _gpt2(self, i, cfg):
        H, D, E = self.heads, self.dim, cfg.n_embd
        def hook(module, args, out):
            q, k, _ = out.split(E, dim=2)
            B, T, _ = q.shape
            self.q[i] = q.reshape(B, T, H, D).transpose(1, 2).detach()
            self.k[i] = k.reshape(B, T, H, D).transpose(1, 2).detach()
        return hook

    def _rope(self, i, attn):
        apply_rope = getattr(sys.modules[type(attn).__module__], "apply_rotary_pos_emb", None)
        if apply_rope is None:
            raise SystemExit(f"{type(attn).__name__}: no apply_rotary_pos_emb in its module")
        H, KV, D = self.heads, self.kv_heads, self.dim
        def hook(module, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings")
            if h is None or pe is None:
                raise SystemExit("attention layer did not receive position_embeddings; "
                                 "use transformers >= 4.48")
            cos, sin = pe
            B, T, _ = h.shape
            q = module.q_proj(h).view(B, T, H, D)
            k = module.k_proj(h).view(B, T, KV, D)
            if hasattr(module, "q_norm"): q = module.q_norm(q)       # Qwen3-style QK-norm, if present
            if hasattr(module, "k_norm"): k = module.k_norm(k)
            q, k = apply_rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
            self.q[i], self.k[i] = q.detach(), k.detach()
        return hook

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="gpt2")
    ap.add_argument("--out", type=Path, default=Path("reference/attn_blocks_gpt2.json"))
    ap.add_argument("--queries", type=int, default=48, help="query positions sampled per block")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[args.model]
    print(f"Loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    cap = QueryCapture(model)
    group = cap.heads // cap.kv_heads
    print(f"  {cap.heads} query heads, {cap.kv_heads} KV heads (group {group}), head dim {cap.dim}, on {device}")

    rng = np.random.default_rng(args.seed)
    blocks, worst_key_diff = [], 0.0
    for t_idx, text in enumerate(TEXTS):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens).to(device)
        cap.q.clear(); cap.k.clear()
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        T = inputs["input_ids"].shape[1]
        for layer, (k_cache, v_cache) in enumerate(cache_layers(out.past_key_values)):
            # Self-check: recomputed keys must equal the cached keys (validates the RoPE path).
            diff = float((cap.k[layer] - k_cache).norm() / k_cache.norm())
            worst_key_diff = max(worst_key_diff, diff)
            if diff > 1e-3:
                raise SystemExit(f"self-check FAILED: layer {layer} recomputed keys differ from cache by {diff:.2e}")

            j = int(rng.integers(cap.kv_heads))
            heads = list(range(j * group, (j + 1) * group))        # HF repeat_kv: KV head j serves these
            positions = np.sort(rng.choice(np.arange(1, T), size=min(args.queries, T - 1), replace=False))
            q = cap.q[layer][0, heads][:, positions].float().cpu().numpy()          # (G, P, D)
            blocks.append(dict(text_idx=t_idx, layer=layer, kv_head=j, heads=heads, seq_len=int(T),
                               positions=positions.tolist(),
                               keys=b64(k_cache[0, j].float().cpu().numpy()),
                               values=b64(v_cache[0, j].float().cpu().numpy()),
                               queries=b64(q)))
        print(f"  text {t_idx + 1}/{len(TEXTS)}: {T} tokens")

    for h in cap.handles: h.remove()
    print(f"self-check passed: recomputed keys match the KV cache (worst relative difference {worst_key_diff:.1e})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(model=name, dim=cap.dim, group_size=group,
                                        scale=1.0 / np.sqrt(cap.dim), num_blocks=len(blocks),
                                        sampling=dict(queries=args.queries, max_tokens=args.max_tokens, seed=args.seed),
                                        blocks=blocks)))
    print(f"wrote {len(blocks)} blocks to {args.out} ({args.out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
