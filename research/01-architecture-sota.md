# SOTA Small-LM Architecture (2026)

> **TL;DR** — Modern small LMs (10M–500M params) are decoder-only transformers built from a stable, well-understood set of components: RoPE positional embeddings, RMSNorm in pre-norm position, SwiGLU MLP blocks, Grouped-Query Attention (GQA), and tied input/output embeddings. The recipes used by SmolLM2, Qwen3, Llama 3.2, and Gemma 3 have converged to nearly identical blueprints. This document gives you the exact architectural decisions, the quantitative reasoning behind each one, concrete config specs for 30M/125M/350M models trainable on an 8 GB RTX 3060 Ti, and experiments to internalize everything through fast iteration.

---

## Table of Contents

1. [The Canonical Decoder-Only Architecture](#1-the-canonical-decoder-only-architecture)
2. [RoPE and Its Variants](#2-rope-and-its-variants)
3. [Normalization: RMSNorm, LayerNorm, Pre vs Post](#3-normalization-rmsnorm-layernorm-pre-vs-post)
4. [MLP Blocks: SwiGLU and GeGLU](#4-mlp-blocks-swiglu-and-geglu)
5. [Attention Variants: MHA, GQA, MQA, MLA](#5-attention-variants-mha-gqa-mqa-mla)
6. [FlashAttention-2 on an RTX 3060 Ti](#6-flashattention-2-on-an-rtx-3060-ti)
7. [Tied Input/Output Embeddings](#7-tied-inputoutput-embeddings)
8. [Weight Initialization](#8-weight-initialization)
9. [QK-Norm and Logit Soft-Capping](#9-qk-norm-and-logit-soft-capping)
10. [Depth vs Width for Tiny Models](#10-depth-vs-width-for-tiny-models)
11. [μP: Maximal Update Parametrization](#11-μp-maximal-update-parametrization)
12. [Reference Architectures in the Wild](#12-reference-architectures-in-the-wild)
13. [Concrete Config Specs for Your 8 GB GPU](#13-concrete-config-specs-for-your-8-gb-gpu)
14. [VRAM Budget Math](#14-vram-budget-math)
15. [Learn-by-Doing Experiments](#15-learn-by-doing-experiments)

---

## 1. The Canonical Decoder-Only Architecture

Every competitive small LM in 2025–2026 uses a **causal (decoder-only) transformer** — no encoder, no cross-attention. The residual stream flows through alternating attention and MLP blocks:

```
Input tokens
    │
[Embedding table]  ← shape (V, d_model)
    │
┌───────────────────────────────┐  ×L layers
│  Pre-norm (RMSNorm)           │
│      ↓                        │
│  Causal Self-Attention (GQA)  │
│      ↓ (+residual)            │
│  Pre-norm (RMSNorm)           │
│      ↓                        │
│  SwiGLU MLP                   │
│      ↓ (+residual)            │
└───────────────────────────────┘
    │
Final RMSNorm
    │
LM Head (= Embedding table^T if tied)
    │
Logits over vocabulary
```

There is no learned absolute positional encoding — position information is injected via **RoPE** inside the attention QK dot-products.

**Why decoder-only?** For language modeling the autoregressive objective is natural. Encoder-decoders (T5-style) are rarely used for small generative models today because the additional encoder parameters buy little for the generation task.

---

## 2. RoPE and Its Variants

### 2.1 Core Idea

Rotary Positional Embedding ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) encodes relative position directly into the query-key dot product by rotating Q and K vectors using position-dependent rotation matrices. Key property: the inner product ⟨RoPE(q, m), RoPE(k, n)⟩ depends only on (m − n), not on absolute positions.

For head dimension d_head, RoPE applies 2D rotations to d_head/2 pairs of dimensions:

```
θ_i = base^(-2i/d_head),  i = 0, 1, ..., d_head/2 - 1
```

The `base` (rope_theta) controls the frequency range. The original paper used 10,000; modern models use much larger values.

### 2.2 Why Large rope_theta Matters

A larger base spreads the rotation frequencies more slowly, improving length extrapolation. Practical choices in 2025:

| Model | rope_theta | Context |
|---|---|---|
| GPT-2 | N/A (abs pos) | 1,024 |
| Llama 2 | 10,000 | 4,096 |
| SmolLM2 (135M/360M) | 100,000 | 8,192 |
| Llama 3.2 1B | 500,000 | 131,072 |
| Qwen3 1.7B | 1,000,000 | 40,960 |
| Gemma 3 (global attn) | 1,000,000 | 128,000 |

**Rule of thumb for a fresh small training:** use rope_theta = 500,000 and a training context of 2,048–4,096 tokens. You can extend later without full retraining using YaRN.

### 2.3 RoPE Extension Variants

When you need to extend context after training, several techniques exist:

- **NTK-Aware Scaling** ([blog, 2023](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/)): Scales the base by a factor k: `new_base = base * k^(d/(d-2))`. Training-free, works reasonably well for 2–4× extensions.
- **YaRN** ([Peng et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf)): "NTK-by-parts" interpolation that treats high-frequency dimensions differently from low-frequency ones, plus a temperature factor for long-context attention. Up to 15% better than NTK at long contexts. Used by Qwen, DeepSeek, Llama after fine-tuning.
- **For your project:** train at `rope_theta=500000`, context 2048–4096. If you later want 8192+, fine-tune with YaRN scaling.

---

## 3. Normalization: RMSNorm, LayerNorm, Pre vs Post

### 3.1 RMSNorm vs LayerNorm

**LayerNorm** ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)) normalizes by mean and variance, applies learned scale γ and shift β:

```python
LN(x) = γ * (x - mean(x)) / sqrt(var(x) + ε) + β
```

**RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) drops the mean-centering and shift parameter, only normalizes by RMS:

```python
RMSNorm(x) = γ * x / sqrt(mean(x²) + ε)
```

RMSNorm is ~10–15% faster (no mean computation), uses fewer parameters (no β), and achieves comparable quality. All modern models (Llama, Qwen, SmolLM2, Gemma) use RMSNorm. Experimental performance difference is negligible ([MachineLearningMastery, 2024](https://machinelearningmastery.com/layernorm-and-rms-norm-in-transformer-models/)).

### 3.2 Pre-Norm vs Post-Norm

**Post-norm** (original "Attention is All You Need"): normalizes _after_ the residual addition. Prone to vanishing gradients in deep networks; requires careful learning rate warmup.

**Pre-norm**: normalizes _before_ each sub-layer, adds unnormalized residual:

```
h = x + SubLayer(RMSNorm(x))
```

Pre-norm is the overwhelming 2025 choice: stable training, no warmup issues, better gradient flow. Used by SmolLM2, Qwen3, Llama 3.2.

**Peri-LN** (apply norm both before and after): used by OLMo2, Gemma 2/3. Slightly more stable at very large scale, but adds computation. For models under 500M, pre-norm is the pragmatic choice.

**Your choice:** Pre-norm with RMSNorm. This is the safe, universally-supported default. Use `rms_norm_eps=1e-5` (SmolLM2) or `1e-6` (Qwen3).

---

## 4. MLP Blocks: SwiGLU and GeGLU

### 4.1 The GLU Family

Gated Linear Units inject a multiplicative gating mechanism into the MLP:

```
GLU(x, W, V, b, c) = σ(xW + b) ⊙ (xV + c)
```

**SwiGLU** ([Noam Shazeer, 2020](https://arxiv.org/abs/2002.05202)) replaces sigmoid with Swish (SiLU):

```python
SwiGLU(x) = SiLU(x @ W_gate) * (x @ W_up)
# then project: out = SwiGLU_out @ W_down
```

This requires **three** weight matrices (gate, up, down) instead of two, with the intermediate size typically set to 2/3 × 4 × d_model to keep parameter count comparable to a standard 4× FFN. In practice, people round to a multiple of 64 or 256.

**GeGLU** is identical but uses GELU instead of SiLU. Used by Gemma. Performance difference from SwiGLU is marginal.

### 4.2 Why GLU Is Better

The gating mechanism allows the network to selectively suppress or amplify features, improving gradient flow and expressivity. Empirically SwiGLU/GeGLU improve perplexity and downstream task performance over ReLU and standard GELU across all scales ([EmergentMind survey, 2024](https://www.emergentmind.com/topics/swiglu-activation-function)).

### 4.3 Intermediate Size Calculation

For a model with hidden dim `d`:
- Standard 4× FFN: intermediate = 4 × d, total params per layer = 2 × d × 4d = 8d²
- SwiGLU 2/3× FFN: intermediate = 2/3 × 4d ≈ 2.67d, total per layer = 3 × d × 2.67d = 8d² (comparable)

Most implementations round intermediate to nearest multiple of 256. SmolLM2-135M uses intermediate=1536 (≈2.67 × 576).

```python
# Compute standard SwiGLU intermediate size
d_model = 576
intermediate_raw = int(2/3 * 4 * d_model)
intermediate = (intermediate_raw // 256 + 1) * 256  # round up to multiple of 256
# → 1536 for d_model=576
```

**Your choice:** SwiGLU (`hidden_act="silu"` in HuggingFace parlance for the gate, with 3 projection matrices).

---

## 5. Attention Variants: MHA, GQA, MQA, MLA

### 5.1 Multi-Head Attention (MHA)

The original: H query heads, H key heads, H value heads. KV-cache size ∝ H × 2 × L × d_head × seq_len.

### 5.2 Multi-Query Attention (MQA)

A single shared K and V head. Smallest KV-cache, but quality degrades at larger scales.

### 5.3 Grouped-Query Attention (GQA)

([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) Interpolation: G groups of query heads, each sharing one K and V head. If H=32 query heads and G=8 KV heads, each KV head serves H/G=4 query heads.

**KV-cache reduction:** G/H × original. With G=H/4, you get 4× smaller KV cache.

GQA is the 2025 consensus choice — nearly identical quality to MHA, dramatically smaller KV footprint, faster inference. Every major small model uses it.

**Head dimension:** Always use d_head = d_model / n_q_heads = 64 or 128. Standard is 64 for models <500M; 128 for larger (Qwen3 1.7B uses 128).

### 5.4 Multi-head Latent Attention (MLA)

([DeepSeek-V2, 2024](https://arxiv.org/abs/2405.04434)) Compresses K and V into a low-rank latent vector before caching, rather than sharing heads. More complex to implement and serve. Better KV compression than GQA but adds implementation complexity. **For your project:** use GQA; MLA is overkill and harder to implement correctly.

### 5.5 GQA Ratios Used by Real Models

| Model | n_q_heads | n_kv_heads | ratio |
|---|---|---|---|
| SmolLM2 135M | 9 | 3 | 3:1 |
| SmolLM2 360M | 15 | 5 | 3:1 |
| Qwen2.5 0.5B | 14 | 2 | 7:1 |
| Qwen3 1.7B | 16 | 8 | 2:1 |
| Llama 3.2 1B | 32 | 8 | 4:1 |
| Qwen2.5 1.5B | 12 | 2 | 6:1 |

**Your choice:** For small models, a 4:1 or 3:1 ratio is safe. Use n_kv_heads = n_q_heads // 4 (minimum 1).

---

## 6. FlashAttention-2 on an RTX 3060 Ti

### 6.1 What It Solves

Standard attention materializes the full N×N attention matrix in HBM (GPU RAM), which is O(N²) memory. For seq_len=2048, d_head=64, that's 2048² × 4 bytes = 16 MB just for one head's attention matrix — multiplied by batch size and heads.

[FlashAttention](https://arxiv.org/abs/2205.14135) / [FlashAttention-2](https://arxiv.org/abs/2307.08691) fuse the softmax, masking, and attention-weighted sum into a single kernel that operates in SRAM (on-chip cache), doing the computation in tiles and never materializing the full N×N matrix in HBM. Memory is **O(N)** instead of O(N²).

### 6.2 FlashAttention on Ampere (RTX 3060 Ti)

**Good news:** FlashAttention-2 supports Ampere GPUs (SM80), which includes the RTX 3060 Ti. FlashAttention-3 is Hopper-only (SM90) — do not attempt to use FA3 on your GPU.

On Ampere with FP16/BF16:
- FA2 delivers roughly 150–200 TFLOPS on an A100; the 3060 Ti's 16.2 TFLOPS (BF16) translates to roughly 2–5× throughput improvement over naive attention (kernel efficiency, not raw FLOPS).
- At seq_len=2048, FA2 provides ~10× memory savings vs naive attention, which is critical for fitting larger batches in 8 GB.

**Install:**
```bash
pip install flash-attn --no-build-isolation
# Requires CUDA 11.6+, PyTorch ≥ 2.0
# Build time: 5–20 minutes. Be patient.
```

**In code (PyTorch 2.x):**
```python
import torch
import torch.nn.functional as F

# PyTorch 2.x has FA2 built-in via scaled_dot_product_attention
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=True
):
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Or use the `flash_attn` package directly for more control.

**Your choice:** Use `torch.nn.functional.scaled_dot_product_attention(is_causal=True)` — it automatically uses FA2 on Ampere when available. This is the zero-friction path.

---

## 7. Tied Input/Output Embeddings

The input embedding table E ∈ ℝ^(V×d) maps token IDs to vectors. The LM head maps d-dimensional hidden states back to V-dimensional logits. Tying them means using the same matrix for both:

```python
logits = hidden_state @ embedding_table.T  # no separate lm_head weights
```

**Parameter savings:** For vocab_size=32,000 and d=512, the embedding table has 32,000 × 512 = 16.4M parameters. Sharing it saves that many parameters from the LM head.

**Why it works:** The embedding's role (map tokens to semantic space) and the LM head's role (score tokens given context) are inversely related — the same geometric structure serves both. Original justification in [Press & Wolf, 2017](https://arxiv.org/abs/1608.05859).

**Trade-off:** Recent work ([Weight Tying Biases Token Embeddings, 2026](https://arxiv.org/abs/2603.26663)) shows that tying optimizes the embedding for output prediction, which can compromise input representation quality. However, for small models where the embedding table is a large fraction of total params (>30% at 30M scale), the parameter savings dominate.

**Practical guidance:**
- **Model ≤ 500M params:** Always tie. Both SmolLM2-135M and SmolLM2-360M tie embeddings. Qwen3 ties for small variants.
- **Model > 1B params:** Untie. Llama 3.2 1B, Llama 3.1 8B, OLMo 2, DeepSeek-V3 all untie.
- **Your project:** Tie embeddings at all three scales (30M, 125M, 350M).

---

## 8. Weight Initialization

### 8.1 Standard Practice

Modern transformer LLMs use:

```python
# Embedding table: small normal
nn.init.normal_(embed.weight, mean=0.0, std=1/sqrt(d_model))

# Attention & MLP projection weights: normal with std scaled to avoid
# activation explosion through depth
std = 0.02 / sqrt(2 * n_layers)  # scale residual projections by 1/sqrt(2L)
nn.init.normal_(out_proj.weight, mean=0.0, std=std)
nn.init.normal_(down_proj.weight, mean=0.0, std=std)

# Other linear layers (QKV projections, gate/up projections): normal
nn.init.normal_(layer.weight, mean=0.0, std=0.02)

# Biases: zero (most models don't use biases in attention/MLP at all)
```

The `1/sqrt(2L)` scaling on **output projections of residual branches** (attention output and MLP down-projection) prevents the residual stream variance from growing with depth. This comes from [GPT-2 initialization scheme](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) and is used by GPT-NeoX, Pythia, SmolLM2.

### 8.2 The 0.02 Rule and Its Origin

`std=0.02` is the GPT-2 default for a d=768 model and roughly approximates `1/sqrt(d_model)` ≈ `1/sqrt(750)` ≈ 0.037. Scale this to your model:

```python
std_init = 1 / sqrt(d_model)          # for most layers
std_residual = std_init / sqrt(2 * L)  # for attn out_proj and mlp down_proj
```

### 8.3 TinyInit (2024)

Research from 2024 proposes **TinyInit**: `std = sqrt(1 / (2 * d_model * L))`, scaling by both depth and width simultaneously. This leads to faster loss convergence and more consistent parameter scales. Worth experimenting with.

### 8.4 No Biases (Optional)

Most small models (SmolLM2, Qwen3, Llama 3.2) omit biases from attention projections and MLP layers entirely — fewer parameters, no downside. Keep biases in LayerNorm/RMSNorm (the γ scale parameter).

---

## 9. QK-Norm and Logit Soft-Capping

### 9.1 The Attention Entropy Collapse Problem

Without intervention, attention logits (Q·K^T / sqrt(d_head)) can grow unboundedly during training, causing attention weights to collapse to one-hot distributions ("attention entropy collapse"). Training fails when max attention logit exceeds ~10^4. This is more likely with high learning rates or long training runs.

### 9.2 QK-Norm

Apply RMSNorm (or LayerNorm) to queries and keys before the scaled dot-product:

```python
q = rms_norm(q)   # per-head normalization
k = rms_norm(k)   # per-head normalization
attn = (q @ k.T) / sqrt(d_head)
```

This bounds the magnitude of Q and K, preventing logit explosion. Gemma 3 switched from logit soft-capping to QK-norm and reports improved accuracy and speed. Qwen3 added QK-norm (removed QKV bias from Qwen2).

QK-norm allows **1.5× higher learning rates** without divergence compared to unnormalized attention ([OpenReview 2024](https://openreview.net/forum?id=RL6R5ryuL5)).

### 9.3 Logit Soft-Capping (Gemma 2, now deprecated)

Gemma 2 applied a tanh-based soft cap to attention logits:
```python
logits = tanh(logits / cap) * cap  # cap = 50.0
```
This prevents extreme values but adds computation. Gemma 3 replaced this with QK-norm. **For new models, use QK-norm instead.**

### 9.4 Should You Use QK-Norm?

For models trained at small scale with standard learning rates, attention collapse is rare. But:
- If you're pushing learning rates (1e-3 and above): **add QK-norm**
- If training is unstable or loss spikes: **add QK-norm**
- For a baseline run at standard LR (~3e-4): optional, but recommended

Implementation is one line per head in your attention forward pass.

---

## 10. Depth vs Width for Tiny Models

### 10.1 The Trade-off

Given a fixed parameter budget, should you be wide (large d_model, few layers) or deep (small d_model, many layers)?

**[MobileLLM (Meta, ICLR 2024)](https://arxiv.org/abs/2402.14905)** systematically studied this by training 19 models near 125M and 350M params with varied depth/width ratios. Key findings:

- **Deeper models consistently outperform wider models** at the same parameter count for small-scale LMs
- A 125M model with 30 layers outperformed one with 12 layers and larger hidden size
- Depth improves generalization (especially compositional/out-of-distribution generalization)
- However: deeper models have **higher training and inference latency** (sequential computation)

**Intuition:** Each layer can learn a different level of abstraction. Shallow-wide models must cram multiple levels of representation into each layer, which is less efficient.

### 10.2 Depth Guidelines for Small Models

Based on MobileLLM, SmolLM2, and Qwen3 data:

| Target params | Recommended layers | Typical d_model |
|---|---|---|
| 30M | 16–20 | 256–384 |
| 125M | 24–32 | 512–640 |
| 350M | 28–36 | 768–960 |

SmolLM2-135M uses 30 layers with d=576 — heavily depth-biased for quality. SmolLM2-360M uses 32 layers with d=960.

### 10.3 Width Rules

**d_model must be divisible by n_heads.** For GQA, n_kv_heads must divide n_heads. Common n_heads: 8, 12, 16, 24 for small models. With d_head=64: d_model = n_heads × 64.

**Intermediate size (SwiGLU):** `int(2/3 * 4 * d_model)` rounded to nearest multiple of 256.

### 10.4 The ETH Zurich Fast-Iteration Angle

For your workflow: start with the **depth-first** design but don't obsess over optimal ratio. The key insight from MobileLLM is that going from 12 to 24+ layers at 125M gives 2–4% absolute accuracy gains on benchmarks. That's worth the depth.

---

## 11. μP: Maximal Update Parametrization

### 11.1 The Problem μP Solves

In standard parametrization (SP), the optimal learning rate changes when you change model width. Doubling d_model typically requires halving the learning rate. This means you can't tune hyperparameters on a small cheap model and expect them to transfer to a large one.

**μP** ([Yang et al., NeurIPS 2022](https://arxiv.org/abs/2203.03466)) re-parametrizes the network so that the **optimal learning rate stays constant as width scales**. This enables **μTransfer**: tune HPs on a tiny 40M proxy model, copy to a 1B model — 7× cheaper than full tuning at scale.

### 11.2 The Core Parameter Changes

For each layer type, μP changes initialization and/or learning rate scaling relative to width multiplier m (where m = d_model / d_model_base):

| Layer type | SP init std | μP init std | μP LR scale |
|---|---|---|---|
| Input embedding | 1/√V | 1/√V (unchanged) | 1/m |
| Hidden linear (QKV, gate, up) | σ | σ/√m | m × base_lr |
| Output (attn out, down_proj) | σ/√(2L) | σ/(m√(2L)) | m × base_lr |
| LM head (untied) | σ | σ/m | 1 (frozen or 1/m) |

The attention logit scaling also changes from 1/√d_head to 1/d_head (when using tied embeddings) or vice versa depending on whether you use μP's special treatment.

### 11.3 Practical μP Workflow

**Step 1:** Choose a proxy model with the same depth as your target but **d_model=256** (small enough to sweep cheaply).

**Step 2:** Apply μP initialization and per-layer LR scaling.

**Step 3:** Sweep over:
- Base learning rate η (log-uniform, e.g., [1e-4, 1e-2])
- Init std σ
- Embedding multiplier β_e
- Output logit multiplier β_o

**Step 4:** Train proxy for ~20 tokens/param (e.g., 5M tokens for a 256K-param proxy).

**Step 5:** Copy optimal η to your full model. No rescaling needed.

**Practical caveat:** Recent work ([arxiv 2510.19093, 2025](https://arxiv.org/abs/2510.19093)) shows weight decay settings may matter as much as μP LR for transfer. Tune both.

### 11.4 Libraries

- **[mup](https://github.com/microsoft/mup)** — Microsoft's reference implementation, works with PyTorch
- **[u-μP](https://proceedings.iclr.cc/paper_files/paper/2025/file/e3130a164f5c724e37271b93bc76dd28-Paper-Conference.pdf)** (ICLR 2025) — unit scaling variant, cleaner and more FP8-stable

**For the ETH Zurich fast-iteration style:** μP is most valuable when you plan to run ≥10 differently-sized training runs. For a single architecture, a manual LR sweep is fine. Add μP when you start scaling.

---

## 12. Reference Architectures in the Wild

Here are ground-truth configurations extracted from official model cards and config.json files:

### SmolLM2-135M ([HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M))

```json
{
  "hidden_size": 576,
  "num_hidden_layers": 30,
  "num_attention_heads": 9,
  "num_key_value_heads": 3,
  "intermediate_size": 1536,
  "max_position_embeddings": 8192,
  "rope_theta": 100000,
  "vocab_size": 49152,
  "tie_word_embeddings": true,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-05,
  "architecture": "LlamaForCausalLM"
}
```

Notes: GQA ratio 3:1, d_head=64, SwiGLU, pre-norm RMSNorm, tied embeddings. ~24.6M non-embedding params.

### SmolLM2-360M ([HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M))

```json
{
  "hidden_size": 960,
  "num_hidden_layers": 32,
  "num_attention_heads": 15,
  "num_key_value_heads": 5,
  "intermediate_size": 2560,
  "max_position_embeddings": 8192,
  "rope_theta": 100000,
  "vocab_size": 49152,
  "tie_word_embeddings": true,
  "rms_norm_eps": 1e-05
}
```

Notes: GQA ratio 3:1, d_head=64, tied embeddings, 32 layers at d=960.

### Qwen3-1.7B ([Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B))

```json
{
  "hidden_size": 2048,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "intermediate_size": 6144,
  "max_position_embeddings": 40960,
  "rope_theta": 1000000,
  "vocab_size": 151936,
  "tie_word_embeddings": true,
  "rms_norm_eps": 1e-06,
  "head_dim": 128
}
```

Notes: QK-norm added (no QKV bias), GQA 2:1, rope_theta=1M, large vocab.

### Llama 3.2-1B ([meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B))

```
hidden_size: 2048
num_hidden_layers: 16
num_attention_heads: 32
num_key_value_heads: 8
intermediate_size: 8192
max_position_embeddings: 131072
rope_theta: 500000
vocab_size: 128256
tie_word_embeddings: false
```

Notes: Only 16 layers (wide and shallow vs SmolLM2), GQA 4:1, huge intermediate (4×d_model), large vocab, rope_theta=500k for long context, **untied** embeddings (1B is large enough).

### Qwen2.5-0.5B ([Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B))

```
hidden_size: 896
num_hidden_layers: 24
num_attention_heads: 14
num_key_value_heads: 2
intermediate_size: 4864
vocab_size: 151936
rope_theta: 1000000
tie_word_embeddings: true
```

Notes: Aggressive MQA-like ratio (7:1 GQA), tied embeddings, large vocab.

### Gemma 3-1B (architectural summary)

- Interleaved local/global attention: 5 local sliding-window layers per 1 global layer
- Local: rope_theta=10,000, window=1024 tokens
- Global: rope_theta=1,000,000, full context (32K for 1B model)
- Replaced Gemma 2's logit soft-capping with QK-norm
- Pre+post norm (Peri-LN style) with RMSNorm
- Technical report: [arxiv 2503.19786](https://arxiv.org/abs/2503.19786)

---

## 13. Concrete Config Specs for Your 8 GB GPU

These specifications are designed to (a) match current SOTA architectural patterns, (b) fit comfortably in 8 GB VRAM for training with bf16 + AdamW + gradient checkpointing, and (c) be runnable in minutes per experiment.

### 13.1 30M Parameter Model

```python
# ~30M total params (including tied embeddings)
config = {
    "d_model": 256,
    "n_layers": 18,
    "n_heads": 8,            # d_head = 256/8 = 32
    "n_kv_heads": 2,         # GQA 4:1
    "d_ffn": 672,            # int(2/3 * 4 * 256) rounded to mult of 64 = 672
    "vocab_size": 32768,     # e.g. GPT-NeoX or custom BPE tokenizer
    "max_seq_len": 2048,
    "rope_theta": 500000,
    "tie_embeddings": True,
    "rms_norm_eps": 1e-5,
    "activation": "swiglu",
}

# Parameter count estimate:
# Embedding: 32768 * 256 = 8.4M (shared with lm_head due to tying)
# Each layer: 4 * d²  (attn) + 3 * d * d_ffn (mlp) ≈ 4*65536 + 3*256*672 ≈ 778K
# 18 layers * 778K = 14M
# Non-embedding total: ~14M
# With shared embedding counted once: ~22M total
```

**Training target:** Fit batch_size=32, seq_len=2048 in <4 GB VRAM without gradient checkpointing. A training run of 500M tokens takes ~2–4 hours.

### 13.2 125M Parameter Model

```python
config = {
    "d_model": 512,
    "n_layers": 24,
    "n_heads": 8,            # d_head = 64
    "n_kv_heads": 2,         # GQA 4:1
    "d_ffn": 1344,           # int(2/3 * 4 * 512) = 1365, round to 1344 (mult 64)
    "vocab_size": 32768,
    "max_seq_len": 2048,
    "rope_theta": 500000,
    "tie_embeddings": True,
    "rms_norm_eps": 1e-5,
    "activation": "swiglu",
}

# Parameter count:
# Embedding: 32768 * 512 = 16.8M
# Each layer: 4*512² (attn) + 3*512*1344 (mlp) = 1.05M + 2.07M = 3.12M
# 24 layers: 74.9M non-embedding
# Total (embedding counted once, tied): ~92M
# Adding lm_head (tied): ~92M
```

**Training target:** Batch_size=16, seq_len=2048 fits in ~5 GB VRAM with bf16. With gradient checkpointing, batch_size=32 fits. A 2B token run takes ~8–12 hours.

### 13.3 350M Parameter Model

```python
config = {
    "d_model": 768,
    "n_layers": 28,
    "n_heads": 12,           # d_head = 64
    "n_kv_heads": 4,         # GQA 3:1
    "d_ffn": 2048,           # int(2/3 * 4 * 768) = 2048 (exactly mult of 256!)
    "vocab_size": 32768,
    "max_seq_len": 2048,
    "rope_theta": 500000,
    "tie_embeddings": True,
    "rms_norm_eps": 1e-5,
    "activation": "swiglu",
}

# Parameter count:
# Embedding: 32768 * 768 = 25.2M
# Each layer: 4*768² (attn, approx for GQA) + 3*768*2048 (mlp)
#           = 2.36M + 4.72M = 7.08M
# 28 layers: 198M non-embedding
# Total with tied embedding: ~223M
# → For 350M, use vocab_size=50257 or 49152, or increase d_model to 896:
```

**Adjusted 350M variant** (matching SmolLM2-360M spirit):
```python
config_350m = {
    "d_model": 896,
    "n_layers": 28,
    "n_heads": 14,           # d_head = 64
    "n_kv_heads": 4,         # GQA 3.5:1 (round to 2 for cleanliness: 7:1)
    "d_ffn": 2368,           # 2/3 * 4 * 896 = 2389, round to 2368
    "vocab_size": 32768,
    "max_seq_len": 2048,
    "rope_theta": 500000,
    "tie_embeddings": True,
}
# Non-embedding params per layer: 4*896² + 3*896*2368 ≈ 3.21M + 6.37M = 9.58M
# 28 layers: 268M + 25.2M embedding = ~293M total
```

**Training target:** With gradient checkpointing, batch_size=8 seq_len=2048 fits in ~7 GB VRAM. A 2B token run takes ~18–24 hours.

---

## 14. VRAM Budget Math

### 14.1 Training Memory Formula

For mixed-precision training with AdamW (the standard):

```
Total VRAM = Parameters + Gradients + Optimizer_states + Activations + Misc

In bytes per parameter:
  BF16 params:      2 bytes
  BF16 gradients:   2 bytes
  FP32 Adam m1:     4 bytes
  FP32 Adam m2:     4 bytes
  FP32 master copy: 4 bytes (optional but common for stability)
  ─────────────────────────
  Subtotal:        16 bytes/param  (without activations)

Activations (per-token, per-layer, for backprop):
  ≈ 12 × d_model × seq_len × batch_size × n_layers bytes  (rough estimate)
  With gradient checkpointing: ≈ sqrt(n_layers) × that
```

### 14.2 Worked Examples for 8 GB VRAM

**30M model** (d=256, L=18):
```
Model+grads+opt: 30M × 16 bytes = 480 MB
Activations (bs=32, seq=2048, no ckpt):
  12 × 256 × 2048 × 32 × 18 = 3.6 GB
Total: ~4.1 GB ✓ (fits with headroom)
```

**125M model** (d=512, L=24):
```
Model+grads+opt: 125M × 16 bytes = 2 GB
Activations (bs=16, seq=2048, no ckpt):
  12 × 512 × 2048 × 16 × 24 = 6.0 GB
Total: ~8 GB — tight! Use gradient checkpointing:
  Activations ≈ 6.0 / sqrt(24) ≈ 1.2 GB → Total: ~3.2 GB ✓
```

**350M model** (d=896, L=28):
```
Model+grads+opt: 350M × 16 bytes = 5.6 GB
Activations (bs=4, seq=2048, with ckpt):
  12 × 896 × 2048 × 4 × sqrt(28) ≈ 0.9 GB
Total: ~6.5 GB ✓ (tight but feasible)
```

### 14.3 Tips for Staying Under 8 GB

1. **Enable gradient checkpointing** (`model.gradient_checkpointing_enable()`) — saves ~40% activation memory at 20% throughput cost
2. **Use BF16 throughout** — halves param/grad memory vs FP32
3. **Use 8-bit AdamW** ([bitsandbytes](https://github.com/TimDettmers/bitsandbytes)) — halves optimizer state memory (8 bytes/param instead of 16)
4. **Reduce sequence length** — activations scale linearly with seq_len; 1024 vs 2048 halves activation memory
5. **Fused optimizer kernels** — `torch.optim.AdamW(fused=True)` reduces peak memory spikes

With 8-bit AdamW: `Model+opt = (2+2+1+1)×N bytes = 6 bytes/param`, leaving much more room for activations.

---

## 15. Learn-by-Doing Experiments

These experiments follow the ETH Zurich / fast-iteration principle: each runs in minutes on your RTX 3060 Ti and builds genuine intuition.

### Experiment 1: Architecture Ablation at 30M (45–90 mins total)

Train four 30M-param models for 200M tokens each, varying only one architectural choice. Use a fixed LR of 3e-4, batch 256 sequences of 512 tokens, cosine decay.

```
Model A (baseline): d=256, L=18, MHA (no GQA), no tied embs
Model B:            d=256, L=18, GQA (4:1), no tied embs
Model C:            d=256, L=18, GQA (4:1), tied embs
Model D:            d=512, L=8,  GQA (4:1), tied embs  ← same ~30M params, wide/shallow
```

Compare: final val loss, training throughput (tokens/sec), VRAM peak.

**Expected finding:** C beats A (GQA is essentially free quality), C beats D (depth wins for small models), B≈C (tying barely changes loss but saves 8M params you can reinvest).

### Experiment 2: RoPE Theta Sensitivity (15 mins)

Train three 30M models for 100M tokens with identical configs except rope_theta: {10000, 100000, 500000}. Then test next-token prediction on sequences of lengths 512, 1024, 2048.

```python
# After training, measure perplexity at each length
for seq_len in [512, 1024, 2048]:
    ppl = evaluate(model, val_tokens[:, :seq_len])
    print(f"theta={theta}, len={seq_len}: ppl={ppl:.2f}")
```

**Expected finding:** Higher theta degrades slightly at short contexts but generalizes dramatically better at long contexts. At seq_len=2048 with theta=10000, performance may collapse.

### Experiment 3: QK-Norm vs Logit Stability (30 mins)

Train a 30M model with an **aggressive learning rate** (LR=1e-2) both with and without QK-norm. Log the max attention logit every 100 steps.

```python
# In your attention forward:
max_logit = (q @ k.transpose(-2,-1) / math.sqrt(d_head)).max().item()
wandb.log({"max_attn_logit": max_logit})
```

**Expected finding:** Without QK-norm, max logit grows ~exponentially and training eventually diverges. With QK-norm, it stays bounded and you can train stably at 3–5× higher LR.

### Experiment 4: μTransfer Width Scaling (60–90 mins)

Implement a μP parametrized 30M model (or use the `mup` library). Then train three versions with d_model ∈ {128, 256, 512} (adjust n_layers to keep total params comparable), all with the **same base_lr=3e-4 and same other HPs**.

In standard parametrization, these will have different optimal LRs. In μP, they should converge to similar validation loss with the same LR.

```
SP model 128-wide: best val loss at lr=1e-3 (example)
SP model 512-wide: best val loss at lr=3e-4 (different!)
μP model 128-wide: best val loss at lr=3e-4
μP model 512-wide: best val loss at lr=3e-4 ← same!
```

**Expected finding:** μP models have consistent optimal LR across widths. This lets you tune on the cheap 128-wide proxy and trust the result at 512-wide scale.

---

## Key Architecture Decision Checklist

For a new small-LM training run in 2026, this is your blueprint:

| Component | Choice | Why |
|---|---|---|
| Architecture | Decoder-only | Standard for generation |
| Attention | GQA, 3:1 or 4:1 Q:KV ratio | KV cache efficiency, near-MHA quality |
| Normalization | RMSNorm, pre-norm | Fast, stable, universal |
| MLP | SwiGLU (3 matrices, 2/3 × 4 × d intermediate) | Better perplexity than ReLU/GELU |
| Positional | RoPE, theta=500000 | Long-context capability |
| Embeddings | Tied input/output for ≤500M | Parameter efficiency |
| FlashAttention | FA2 via `scaled_dot_product_attention` | 2–5× throughput, O(N) memory |
| QK-norm | Optional, add if LR > 5e-4 or training unstable | Prevents attention collapse |
| Init | Normal, std=1/sqrt(d), residual layers ÷ sqrt(2L) | Stable gradient flow |
| Bias | No bias in attention/MLP projections | Fewer params, no downside |
| Depth | More layers > wider hidden for tiny models | Better generalization per param |
| μP | Use if planning ≥5 differently-sized runs | Free HP transfer across scales |

---

## References

- [SmolLM2 paper (2502.02737)](https://arxiv.org/abs/2502.02737) — SmolLM2 architecture and training details
- [SmolLM2-135M config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — Ground truth config
- [Qwen3 Technical Report (2505.09388)](https://arxiv.org/html/2505.09388v1) — Qwen3 architecture
- [MobileLLM (2402.14905)](https://arxiv.org/abs/2402.14905) — Depth vs width for sub-billion models
- [GQA paper (2305.13245)](https://arxiv.org/abs/2305.13245) — Grouped-Query Attention
- [FlashAttention-2 (2307.08691)](https://arxiv.org/abs/2307.08691) — Efficient attention
- [FlashAttention-3 NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf) — FA3 (Hopper-only, not for 3060 Ti)
- [SwiGLU (2002.05202)](https://arxiv.org/abs/2002.05202) — Gated MLP activations
- [YaRN ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf) — Context extension
- [μP / μTransfer (2203.03466)](https://arxiv.org/abs/2203.03466) — Maximal update parametrization
- [U-μP ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/e3130a164f5c724e37271b93bc76dd28-Paper-Conference.pdf) — Unit-scaled μP
- [Cerebras μP Guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization) — Practical μP implementation
- [Gemma 3 Technical Report (2503.19786)](https://arxiv.org/abs/2503.19786) — QK-norm replacing soft-capping
- [Weight Tying (1608.05859)](https://arxiv.org/abs/1608.05859) — Original tied embedding paper
- [Weight Tying Bias (2603.26663)](https://arxiv.org/abs/2603.26663) — Recent analysis of trade-offs
- [DeepSeek-V2 MLA (2405.04434)](https://arxiv.org/abs/2405.04434) — Multi-head Latent Attention
- [RMSNorm (1910.07467)](https://arxiv.org/abs/1910.07467) — Root Mean Square Layer Normalization
- [Peri-LN (2502.02732)](https://arxiv.org/abs/2502.02732) — Peri-LayerNorm for stable training
- [Weight Decay vs μP (2510.19093)](https://arxiv.org/abs/2510.19093) — μP practical limitations
