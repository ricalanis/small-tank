#!/usr/bin/env python3
"""
Experiment 0 — throughput + VRAM probe (research/08 §3, #0).

Purpose: replace the spec-scaled, 2-4x-conflicting throughput/VRAM ESTIMATES in
docs 04/07 with GROUND TRUTH measured on this actual RTX 3060 Ti (sm_86, 8 GB).
This is the single highest-leverage run in the backlog: it retires DECISIONS.md
D6 (throughput figure) and §2.1 (the 04-vs-07 conflict), and tells you what
actually fits in 8 GB before you write a line of training code.

It builds a *representative* 2026-stack model (RoPE, RMSNorm, SwiGLU, GQA, SDPA)
at each candidate size, runs real train steps (fwd + bwd + optimizer), and reports
tokens/sec, peak VRAM (torch.cuda.max_memory_allocated — NOT nvidia-smi, per 07),
and an approximate MFU. OOM is caught and reported, not fatal.

This file is also the reference block that will seed src/model.py in Stage 1.

Usage:
    .venv/bin/python scripts/probe.py                    # default matrix
    .venv/bin/python scripts/probe.py --seq 512 --batches 8,16,32,48
    .venv/bin/python scripts/probe.py --configs 125m --optim adamw8bit --grad-ckpt
    .venv/bin/python scripts/probe.py --configs 5m,30m-deep --eager   # SDPA vs eager (Exp 4 preview)
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --- candidate configs (param counts are computed + printed; labels are approximate) ---
# Architecture provenance: research/01; sizes/roles: RECOMMENDATION.md + DECISIONS.md D1/D3.
CONFIGS = {
    # micro-proxy: vocab 4096 keeps embedding ~14% of params (DECISIONS.md D3 parity).
    "5m":       dict(d=256, n_layers=6,  n_heads=8,  n_kv=2, vocab=4096),
    # the two 30M candidates Experiment 2 will decide between (DECISIONS.md D1):
    "30m-deep": dict(d=256, n_layers=18, n_heads=8,  n_kv=2, vocab=8192),   # deep-narrow (provisional default)
    "30m-wide": dict(d=512, n_layers=8,  n_heads=8,  n_kv=2, vocab=8192),   # wide-shallow (RECOMMENDATION)
    # level-2 graduation target; the 8 GB fit here is the key question:
    "125m":     dict(d=768, n_layers=12, n_heads=12, n_kv=4, vocab=16384),
}


def rms_norm(x, weight, eps=1e-5):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x.to(dt)) * weight


def precompute_rope(head_dim, seq, theta=500000.0, device="cuda"):
    # research/01: theta=500k is "free insurance" for later context extension.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv_freq)              # [seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)       # [seq, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, H, T, Dh] ; cos,sin: [T, Dh]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg, use_sdpa=True):
        super().__init__()
        self.d = cfg["d"]; self.nh = cfg["n_heads"]; self.nkv = cfg["n_kv"]
        self.hd = self.d // self.nh
        self.n_rep = self.nh // self.nkv
        self.use_sdpa = use_sdpa
        self.wq = nn.Linear(self.d, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(self.d, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(self.d, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, self.d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)    # [B, nh, T, hd]
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)   # [B, nkv, T, hd]
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # GQA: expand KV heads to match Q heads.
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        if self.use_sdpa:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # FlashAttn-2 path on Ampere
        else:
            # eager reference path (for the SDPA-vs-eager comparison, Exp 4)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            att = att.masked_fill(mask, float("-inf"))
            o = att.softmax(-1) @ v
        o = o.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d"]
        hidden = int(2 / 3 * 4 * d)
        hidden = ((hidden + 63) // 64) * 64        # round to multiple of 64 (research/01)
        self.w1 = nn.Linear(d, hidden, bias=False)  # gate
        self.w3 = nn.Linear(d, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, d, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg, use_sdpa=True):
        super().__init__()
        d = cfg["d"]
        self.n1 = nn.Parameter(torch.ones(d))
        self.attn = Attention(cfg, use_sdpa)
        self.n2 = nn.Parameter(torch.ones(d))
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x, self.n1), cos, sin)
        x = x + self.mlp(rms_norm(x, self.n2))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg, use_sdpa=True, grad_ckpt=False):
        super().__init__()
        self.cfg = cfg; self.grad_ckpt = grad_ckpt
        self.tok = nn.Embedding(cfg["vocab"], cfg["d"])
        self.blocks = nn.ModuleList([Block(cfg, use_sdpa) for _ in range(cfg["n_layers"])])
        self.nf = nn.Parameter(torch.ones(cfg["d"]))
        self.head = nn.Linear(cfg["d"], cfg["vocab"], bias=False)
        self.head.weight = self.tok.weight            # tied embeddings (research/03)

    def forward(self, idx, cos, sin, targets=None):
        x = self.tok(idx)
        for b in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(b, x, cos, sin, use_reentrant=False)
            else:
                x = b(x, cos, sin)
        x = rms_norm(x, self.nf)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss if loss is not None else logits


def param_count(m):
    # tied head shares tok.weight, so count unique params.
    seen, n = set(), 0
    for p in m.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p)); n += p.numel()
    return n


def make_optim(model, kind):
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    return torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)


def measure(name, cfg, batch, seq, optim_kind, use_sdpa, grad_ckpt, steps, warmup, peak_tflops):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = TinyLM(cfg, use_sdpa=use_sdpa, grad_ckpt=grad_ckpt).cuda().train()
        nparams = param_count(model)
        opt = make_optim(model, optim_kind)
        cos, sin = precompute_rope(cfg["d"] // cfg["n_heads"], seq)

        def one_step():
            idx = torch.randint(0, cfg["vocab"], (batch, seq), device="cuda")
            tgt = torch.randint(0, cfg["vocab"], (batch, seq), device="cuda")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(idx, cos, sin, tgt)
            loss.backward()
            opt.step()

        for _ in range(warmup):
            one_step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            one_step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        tokens = batch * seq * steps
        tok_s = tokens / dt
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        # approx fwd+bwd flops/token: 6*N + attention term (research/00, nanoGPT-style estimate)
        flops_per_tok = 6 * nparams + 12 * cfg["n_layers"] * cfg["d"] * seq
        mfu = (flops_per_tok * tok_s) / (peak_tflops * 1e12) * 100
        res = dict(config=name, params_M=round(nparams / 1e6, 1), batch=batch, seq=seq,
                   optim=optim_kind, sdpa=use_sdpa, grad_ckpt=grad_ckpt,
                   tok_s=round(tok_s), peak_gb=round(peak_gb, 2), mfu_pct=round(mfu, 1), oom=False)
        del model, opt
        torch.cuda.empty_cache()
        return res
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(config=name, batch=batch, seq=seq, optim=optim_kind, sdpa=use_sdpa,
                    grad_ckpt=grad_ckpt, oom=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return dict(config=name, batch=batch, seq=seq, optim=optim_kind, sdpa=use_sdpa,
                        grad_ckpt=grad_ckpt, oom=True)
        raise


def main():
    p = argparse.ArgumentParser(description="Experiment 0 — throughput + VRAM probe")
    p.add_argument("--configs", default="5m,30m-deep,30m-wide,125m", help="comma list from: " + ",".join(CONFIGS))
    p.add_argument("--batches", default="8,16,32", help="comma list of batch sizes to sweep")
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"])
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--eager", action="store_true", help="use eager attention instead of SDPA")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--peak-tflops", type=float, default=32.5,
                   help="assumed bf16 peak for MFU (3060 Ti ~32.5 TFLOPS FP16/FP32-acc; approximate)")
    p.add_argument("--out", default="research/_assets/exp0_results.json")
    args = p.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"# Experiment 0 — throughput + VRAM probe")
    print(f"# device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"seq={args.seq} optim={args.optim} sdpa={not args.eager} grad_ckpt={args.grad_ckpt}")
    print(f"# MFU assumes {args.peak_tflops} bf16 TFLOPS (approximate — see --peak-tflops)\n")
    hdr = f"{'config':10} {'params':>7} {'batch':>5} {'tok/s':>9} {'peakVRAM':>9} {'MFU%':>6}  status"
    print(hdr); print("-" * len(hdr))

    results = []
    for name in [c.strip() for c in args.configs.split(",")]:
        cfg = CONFIGS[name]
        for batch in [int(b) for b in args.batches.split(",")]:
            r = measure(name, cfg, batch, args.seq, args.optim, not args.eager,
                        args.grad_ckpt, args.steps, args.warmup, args.peak_tflops)
            results.append(r)
            if r.get("oom"):
                print(f"{name:10} {'':>7} {batch:>5} {'—':>9} {'—':>9} {'—':>6}  OOM")
                break  # larger batches will also OOM
            print(f"{name:10} {r['params_M']:>6}M {batch:>5} {r['tok_s']:>9} "
                  f"{r['peak_gb']:>7}GB {r['mfu_pct']:>6}  ok")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                       seq=args.seq, results=results), f, indent=2)
    print(f"\n# wrote {args.out}")
    print("# Next: paste the winning rows into research/log.md as RUN 001, and update")
    print("#   DECISIONS.md D6 + docs 04/07 with the measured tok/s (delete the losing estimate).")


if __name__ == "__main__":
    main()
