"""
Stage-2 additive ablation ladder (BUILD-PLAN Stage 2, research/01 + 00).

Starts from a GPT-2-era baseline (sinusoidal pos, LayerNorm, GELU MLP, MHA) and
adds ONE modern component per rung, proving each earns its place. Every rung uses
the SAME data, token budget, and seed (train() reseeds to 1337) so the only
variable is the component added. Reports val loss, Δ vs previous rung, params, tok/s.

  .venv/bin/python scripts/ablate.py                 # default: 2500 steps/rung
  .venv/bin/python scripts/ablate.py --max-steps 4000
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# cumulative ladder: each rung overrides one more thing toward the modern stack.
LADDER = [
    ("baseline (GPT-2 era)", dict(pos="learned", norm="layernorm", mlp="gelu", n_kv_heads=8)),
    ("+ RoPE",               dict(pos="rope",       norm="layernorm", mlp="gelu", n_kv_heads=8)),
    ("+ RMSNorm",            dict(pos="rope",       norm="rmsnorm",   mlp="gelu", n_kv_heads=8)),
    ("+ SwiGLU",             dict(pos="rope",       norm="rmsnorm",   mlp="swiglu", n_kv_heads=8)),
    ("+ GQA (full modern)",  dict(pos="rope",       norm="rmsnorm",   mlp="swiglu", n_kv_heads=2)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "5m.yaml"))
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "_assets", "ablation_5m.json"))
    a = ap.parse_args()

    with open(a.config) as f:
        base = yaml.safe_load(f)

    results = []
    for label, override in LADDER:
        cfg = dict(base)
        cfg.update(override)
        cfg["max_steps"] = a.max_steps
        cfg["name"] = "abl_" + label.split()[0].strip("+()").lower() + "_" + override["mlp"]
        print(f"\n===== {label} =====")
        m = train(cfg, save=False, verbose=True)
        m["label"] = label
        results.append(m)

    # table
    print("\n" + "=" * 72)
    print(f"{'rung':24} {'params':>8} {'val loss':>9} {'Δ prev':>8} {'tok/s':>9}")
    print("-" * 72)
    prev = None
    for m in results:
        d = "" if prev is None else f"{m['val_loss']-prev:+.3f}"
        print(f"{m['label']:24} {m['params']/1e6:>6.2f}M {m['val_loss']:>9.3f} "
              f"{d:>8} {m['tok_s']/1e3:>7.0f}K")
        prev = m["val_loss"]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(max_steps=a.max_steps, results=results), f, indent=2)
    print(f"\n# wrote {a.out}")


if __name__ == "__main__":
    main()
