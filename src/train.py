"""
Training loop (research/02-training-recipes.md).

  python -m src.train --config configs/5m.yaml
  python -m src.train --config configs/5m.yaml --max-steps 6000 --wandb
  # component overrides (Stage-2 ablations):
  python -m src.train --config configs/5m.yaml --pos sinusoidal --norm layernorm --mlp gelu

bf16 autocast (no GradScaler needed), AdamW (weight decay on 2D matrices only),
cosine LR schedule with warmup, grad clipping, periodic val-loss eval, checkpoint
of the best model. The train() function is reusable by scripts/ablate.py.
"""
import argparse
import math
import os
import time

import torch
import yaml

from src.data import bytes_per_token, get_batch, load_tokenizer
from src.model import ModelConfig, TinyLM

LN2 = math.log(2)  # nats -> bits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "research", "checkpoints")


def lr_at(step, cfg):
    warm, total = cfg["warmup_steps"], cfg["max_steps"]
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    if step >= total:
        return cfg["min_lr"]
    r = (step - warm) / max(1, total - warm)
    return cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * r)) * (cfg["lr"] - cfg["min_lr"])


def make_optimizer(model, wd, lr):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def estimate_val_loss(model, cfg, device, n=50):
    model.eval()
    losses = torch.zeros(n)
    for i in range(n):
        x, y = get_batch("val", cfg["batch_size"], cfg["max_seq_len"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def train(cfg, device="cuda", save=True, verbose=True, seed=1337):
    """Train one model from a fully-formed cfg dict. Returns a metrics dict."""
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = load_tokenizer()
    cfg["vocab_size"] = tok.get_vocab_size()

    model = TinyLM(ModelConfig.from_dict(cfg)).to(device)
    n_params = model.num_params()
    emb_frac = 100 * (cfg["vocab_size"] * cfg["d_model"]) / n_params
    if verbose:
        print(f"[train] {cfg['name']}: {n_params/1e6:.2f}M params | embedding {emb_frac:.0f}% "
              f"| pos={cfg.get('pos','rope')} norm={cfg.get('norm','rmsnorm')} "
              f"mlp={cfg.get('mlp','swiglu')} n_kv={cfg['n_kv_heads']}")

    opt = make_optimizer(model, cfg["weight_decay"], cfg["lr"])
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{cfg['name']}.pt")
    bpt = bytes_per_token("val")  # vocab-invariant normalizer (research/06 §2.3)
    best_val = float("inf")
    best_bpb = float("inf")
    tps = cfg["batch_size"] * cfg["max_seq_len"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for step in range(cfg["max_steps"] + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        x, y = get_batch("train", cfg["batch_size"], cfg["max_seq_len"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % cfg["eval_interval"] == 0:
            val = estimate_val_loss(model, cfg, device, cfg["eval_steps"])
            bpb = val / (LN2 * bpt)  # bits-per-byte: vocab-invariant, the primary gate
            if verbose:
                torch.cuda.synchronize()
                tok_s = tps * max(step, 1) / (time.perf_counter() - t0)
                print(f"step {step:>5} | train {loss.item():.3f} | val {val:.3f} "
                      f"| bpb {bpb:.3f} | {tok_s/1e3:.0f}K tok/s | {step*tps/1e6:.0f}M seen")
            if val < best_val:
                best_val = val
                best_bpb = bpb
                if save:
                    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step,
                                "val_loss": val, "bpb": bpb}, ckpt_path)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return dict(name=cfg["name"], params=n_params, val_loss=best_val, bpb=best_bpb,
                bytes_per_token=bpt, tok_s=cfg["max_steps"] * tps / elapsed,
                peak_gb=torch.cuda.max_memory_allocated() / 1e9, ckpt=ckpt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--pos"); ap.add_argument("--norm"); ap.add_argument("--mlp")
    ap.add_argument("--n-kv-heads", type=int)
    ap.add_argument("--name")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    for k in ("pos", "norm", "mlp", "name"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.n_kv_heads is not None:
        cfg["n_kv_heads"] = a.n_kv_heads
    if a.max_steps:
        cfg["max_steps"] = a.max_steps

    if a.wandb:
        import wandb
        wandb.init(project="small-tank", name=cfg["name"], config=cfg)

    m = train(cfg)
    print(f"[train] done. {m['name']}: best val {m['val_loss']:.3f} | bpb {m['bpb']:.3f} "
          f"(bytes/tok {m['bytes_per_token']:.2f}) | {m['params']/1e6:.2f}M | "
          f"{m['tok_s']/1e3:.0f}K tok/s -> {m['ckpt']}")


if __name__ == "__main__":
    main()
