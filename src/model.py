"""
TinyLM — a from-scratch 2026-stack decoder-only transformer, with swappable
components for Stage-2 ablations (research/01-architecture-sota.md).

Default (modern) stack: RoPE (theta=500k), pre-norm RMSNorm, SwiGLU MLP,
Grouped-Query Attention, tied embeddings, attention via SDPA (FlashAttention-2
on Ampere). Each piece is selectable behind the config so you can ablate it:
  pos:  "rope" | "sinusoidal"
  norm: "rmsnorm" | "layernorm"
  mlp:  "swiglu" | "gelu"
  attn: controlled by n_kv_heads  (n_kv_heads == n_heads  =>  plain MHA)

Keep this file small and readable — it is meant to be retyped and understood.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    max_seq_len: int = 1024
    rope_theta: float = 500000.0
    pos: str = "rope"
    norm: str = "rmsnorm"
    mlp: str = "swiglu"
    d_ff: int = 0          # 0 => derive (2/3·4·d, ×64); >0 => explicit MLP hidden (body-resize knob)

    @classmethod
    def from_dict(cls, d):
        keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in keys})


def rms_norm(x, weight, eps=1e-5):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.to(dt) * weight


class Norm(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.kind = kind
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d)) if kind == "layernorm" else None

    def forward(self, x):
        if self.kind == "rmsnorm":
            return rms_norm(x, self.weight)
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, 1e-5)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None], sin[None, None]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.use_rope = cfg.pos == "rope"
        self.nh, self.nkv = cfg.n_heads, cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.n_rep = self.nh // self.nkv
        self.wq = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.use_rope:
            q, k = apply_rope(q, k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # FlashAttn-2 on Ampere
        o = o.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_ff > 0:
            hidden = cfg.d_ff                      # explicit override (vocab-allocation sweep)
        else:
            hidden = int(2 / 3 * 4 * cfg.d_model)
            hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeluMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.w2 = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = Norm(cfg.d_model, cfg.norm)
        self.attn = Attention(cfg)
        self.n2 = Norm(cfg.d_model, cfg.norm)
        self.mlp = SwiGLU(cfg) if cfg.mlp == "swiglu" else GeluMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.nf = Norm(cfg.d_model, cfg.norm)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight                          # tied embeddings

        hd = cfg.d_model // cfg.n_heads
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(cfg.max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)
        if cfg.pos == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)   # GPT-2-era baseline
        elif cfg.pos == "sinusoidal":
            self.register_buffer("sinusoidal", self._sinusoidal(cfg.max_seq_len, cfg.d_model),
                                 persistent=False)

        self.apply(self._init)
        for n, p in self.named_parameters():       # GPT-2 residual-projection init scaling
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _sinusoidal(seq, d):
        pe = torch.zeros(seq, d)
        pos = torch.arange(seq).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p)); n += p.numel()
        return n

    def forward(self, idx, targets=None):
        T = idx.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok(idx)
        if self.cfg.pos == "learned":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif self.cfg.pos == "sinusoidal":
            # original-Transformer convention: scale token emb by sqrt(d) so the unit-magnitude
            # sinusoid doesn't swamp the 0.02-init token signal (the baseline bug from RUN 003a).
            x = x * math.sqrt(self.cfg.d_model) + self.sinusoidal[:T]
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.nf(x)
        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        logits = self.head(x[:, [-1], :])     # inference: only the last position
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
