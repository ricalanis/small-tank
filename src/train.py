"""
Training loop (research/02-training-recipes.md).

  python -m src.train --config configs/5m.yaml
  python -m src.train --config configs/5m.yaml --max-steps 6000 --wandb

bf16 autocast (no GradScaler needed), AdamW (weight decay on 2D matrices only),
cosine LR schedule with warmup, grad clipping, periodic val-loss eval, checkpoint
of the best model. Keep it simple — WSD / over-training come at Stage 3.
"""
import argparse
import math
import os
import time

import torch
import yaml

from src.data import get_batch, load_tokenizer
from src.model import ModelConfig, TinyLM

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.max_steps:
        cfg["max_steps"] = a.max_steps

    device = "cuda"
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # keep model vocab in sync with the trained tokenizer
    tok = load_tokenizer()
    cfg["vocab_size"] = tok.get_vocab_size()

    model = TinyLM(ModelConfig.from_dict(cfg)).to(device)
    n_params = model.num_params()
    emb_frac = 100 * (cfg["vocab_size"] * cfg["d_model"]) / n_params
    print(f"[train] {cfg['name']}: {n_params/1e6:.2f}M params | embedding {emb_frac:.0f}% "
          f"| vocab {cfg['vocab_size']} | seq {cfg['max_seq_len']} | bs {cfg['batch_size']}")

    opt = make_optimizer(model, cfg["weight_decay"], cfg["lr"])

    if a.wandb:
        import wandb
        wandb.init(project="small-tank", name=cfg["name"], config=cfg)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{cfg['name']}.pt")
    best_val = float("inf")
    tokens_per_step = cfg["batch_size"] * cfg["max_seq_len"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for step in range(cfg["max_steps"] + 1):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch("train", cfg["batch_size"], cfg["max_seq_len"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % cfg["eval_interval"] == 0:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tok_s = (tokens_per_step * max(step, 1)) / dt
            val = estimate_val_loss(model, cfg, device, cfg["eval_steps"])
            seen = step * tokens_per_step
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"step {step:>5} | train {loss.item():.3f} | val {val:.3f} | lr {lr:.2e} "
                  f"| {tok_s/1e3:.0f}K tok/s | {seen/1e6:.0f}M seen | {peak:.1f}GB")
            if a.wandb:
                wandb.log({"train_loss": loss.item(), "val_loss": val, "lr": lr,
                           "tokens_seen": seen, "tok_s": tok_s}, step=step)
            if val < best_val:
                best_val = val
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step,
                            "val_loss": val}, ckpt_path)

    print(f"[train] done. best val loss {best_val:.3f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
