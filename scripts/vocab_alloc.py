"""
Vocab-allocation sweep — pure parameter allocation: embedding table vs body (RUN 005, Lesson A2).

Holds total params ~constant (~5.3M) and sweeps the tokenizer vocab V. For each V the embedding
table is V*d; the MLP hidden (d_ff) is solved so total params stay on target, so the ONLY thing
that moves is how the fixed budget is split between the lookup table and the SwiGLU body. Depth,
width, and attention are fixed (this is NOT depth-vs-width — that's Exp 2).

Two controls make the comparison honest:
  1. metric = bits-per-byte (research/06 §2.3). Per-token NLL is incomparable across vocabularies
     (a finer vocab emits shorter, lower-entropy tokens), so it would rig the result toward small V.
  2. budget = equal BYTES of text seen, not equal steps. A small-V model sees more bytes per token,
     so equal-steps would secretly feed it more data. steps = BYTE_BUDGET / (batch*seq*bytes_per_token).

Caveat (logged, not hidden): only d_ff absorbs the budget, so at large V the body becomes
attention-heavy with a vanishing MLP — an asymmetry to read in the writeup, not a clean knob.

  .venv/bin/python scripts/vocab_alloc.py                          # default grid + 100MB budget
  .venv/bin/python scripts/vocab_alloc.py --byte-budget 60000000 --vocabs 1024,4096,16384
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import bytes_per_token, prepare_vocab  # noqa: E402
from src.train import train  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def solve_d_ff(target_params, vocab, d, n_layers, n_heads, n_kv_heads):
    """MLP hidden size so total params == target, holding depth/width/attention fixed."""
    hd = d // n_heads
    attn_layer = 2 * d * d + 2 * d * (n_kv_heads * hd)      # wq+wo, wk+wv
    body_fixed = n_layers * (attn_layer + 2 * d) + d        # attn + 2 RMSNorm/layer + final norm
    swiglu_coeff = n_layers * 3 * d                         # params per unit of d_ff
    emb = vocab * d
    return round((target_params - emb - body_fixed) / swiglu_coeff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "5m.yaml"))
    ap.add_argument("--vocabs", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--target-params", type=int, default=5_300_000)
    ap.add_argument("--byte-budget", type=int, default=100_000_000)
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "_assets", "vocab_alloc.json"))
    a = ap.parse_args()

    with open(a.config) as f:
        base = yaml.safe_load(f)
    d, nl, nh, nkv = base["d_model"], base["n_layers"], base["n_heads"], base["n_kv_heads"]
    B, S = base["batch_size"], base["max_seq_len"]
    vocabs = [int(v) for v in a.vocabs.split(",")]

    results = []
    for V in vocabs:
        out_dir = os.path.join(ROOT, "data", f"v{V}")
        print(f"\n===== building vocab {V} -> {out_dir} =====")
        real_v = prepare_vocab(V, out_dir)
        os.environ["SMALLTANK_DATA_DIR"] = out_dir          # point trainer at this artifact set
        bpt = bytes_per_token("val")
        steps = max(round(a.byte_budget / (B * S * bpt)), 1)
        d_ff = solve_d_ff(a.target_params, real_v, d, nl, nh, nkv)
        if d_ff < 8:
            print(f"  WARN: d_ff={d_ff} — body is starved to near-zero MLP at V={real_v} (degenerate rung)")
            d_ff = max(d_ff, 8)
        cfg = dict(base, name=f"alloc_v{V}", vocab_size=real_v, d_ff=d_ff,
                   max_steps=steps, eval_interval=max(steps // 4, 1))
        print(f"  real_vocab={real_v} bytes/tok={bpt:.3f} d_ff={d_ff} steps={steps} "
              f"(~{a.byte_budget/1e6:.0f}MB)")
        m = train(cfg, save=False, verbose=True)
        m.update(target_vocab=V, vocab=real_v, d_ff=d_ff, steps=steps, byte_budget=a.byte_budget,
                 emb_frac=100 * real_v * d / m["params"])
        results.append(m)
        del os.environ["SMALLTANK_DATA_DIR"]

    print("\n" + "=" * 88)
    print(f"{'vocab':>7} {'d_ff':>6} {'params':>8} {'emb%':>5} {'B/tok':>6} "
          f"{'steps':>6} {'val':>7} {'BPB':>7} {'tok/s':>7}")
    print("-" * 88)
    best = min(results, key=lambda m: m["bpb"])
    for m in results:
        star = " *" if m is best else ""
        print(f"{m['vocab']:>7} {m['d_ff']:>6} {m['params']/1e6:>6.2f}M {m['emb_frac']:>4.0f}% "
              f"{m['bytes_per_token']:>6.2f} {m['steps']:>6} {m['val_loss']:>7.3f} "
              f"{m['bpb']:>7.3f} {m['tok_s']/1e3:>6.0f}K{star}")
    print("-" * 88)
    print(f"BPB-optimal vocab: {best['vocab']} (bpb {best['bpb']:.3f})")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(target_params=a.target_params, byte_budget=a.byte_budget,
                       d_model=d, n_layers=nl, results=results), f, indent=2)
    print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
