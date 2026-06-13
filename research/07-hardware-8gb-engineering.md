# Memory & Throughput Engineering for 8 GB VRAM

**TL;DR.** An RTX 3060 Ti (8 GB GDDR6, 448 GB/s bandwidth, ~16 TFLOPS FP32 / ~64 TFLOPS dense BF16 on tensor cores) can train a 125 M-parameter model from scratch in hours to days, and a 350 M model at the edge of its limits with the right knobs turned. The trick is accounting for every byte: weights, gradients, AdamW's two FP32 moment vectors, activation tensors, and framework overhead. This document does that math explicitly, then walks through the optimizer, precision, and checkpointing levers that let you push the ceiling, and closes with realistic throughput and time-to-train estimates for the 30 M / 125 M / 350 M targets that are the focus of this curriculum.

---

## 1. VRAM Budget Decomposition

Training memory has five additive components. All numbers below assume **BF16 weights + BF16 gradients + FP32 optimizer states** (the standard mixed-precision recipe for pre-training stability):

```
VRAM_total = W + G + O + A + F

W  = model weights          = 2 bytes × P   (BF16)
G  = gradients              = 2 bytes × P   (BF16, same shape as W)
O  = AdamW optimizer states = 8 bytes × P   (two FP32 moments)
A  = activation tensors     = depends on B, S, L, d  (see §1.2)
F  = framework overhead     = ~0.5–1 GB     (CUDA context, caching allocator, cuDNN)
```

where `P` = parameter count.

### 1.1 Static footprint (W + G + O)

```
Static = (2 + 2 + 8) × P = 12 bytes × P
```

| Model size | Static VRAM (12 B/param) |
|-----------|--------------------------|
| 30 M      | 0.36 GB                  |
| 125 M     | 1.50 GB                  |
| 350 M     | 4.20 GB                  |
| 700 M     | 8.40 GB  ← exceeds limit without tricks |

This is the same `(2p + 12) × model_size` formula derived in [Understanding GPU Memory Demands for Training LLMs](https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff) (the 8-byte optimizer term accounts for two FP32 states at 4 bytes each; the `p=2` weight + gradient terms total 4 bytes in BF16).

**Key insight:** For a 350 M model, the static footprint is only ~4.2 GB — you have ~3–3.5 GB left for activations + overhead, which is tight but feasible.

### 1.2 Activation memory

Activations are the most architecture-specific and batch-size-sensitive component. A standard transformer layer with hidden dimension `d`, `h` heads, sequence length `S`, and batch size `B` stores (per layer, in BF16, `p=2` bytes):

```
A_layer = p × S × B × d × (16 + 2/p + 2h·S/d + h·S/(p·d))
```

For typical GPT configs at BF16 (`p=2`), this simplifies approximately to:

```
A_layer ≈ 2 × S × B × d × (17 + h·S/d)
```

Summed over all `L` layers:

```
A_total ≈ L × 2 × S × B × d × (17 + h·S/d)  [bytes]
```

**Concrete examples** (GPT-style, head dim = 64 → `h = d/64`):

| Config         | P    | L  | d    | h  | S=512, B=4 | S=1024, B=2 | S=2048, B=1 |
|---------------|------|----|------|----|------------|-------------|-------------|
| 30 M  (small) | 30M  | 6  | 512  | 8  | ~0.18 GB   | ~0.20 GB    | ~0.22 GB    |
| 125 M (GPT-2) | 125M | 12 | 768  | 12 | ~0.54 GB   | ~0.62 GB    | ~0.72 GB    |
| 350 M         | 350M | 24 | 1024 | 16 | ~1.90 GB   | ~2.20 GB    | ~2.60 GB    |

> These activation estimates are **without gradient checkpointing**. With full gradient checkpointing they drop roughly 5-8x (recompute all activations, keep only layer inputs) at a ~30% compute overhead. See §3 for the tradeoff.

### 1.3 Full VRAM budget at a glance

**RTX 3060 Ti usable budget: ~7.2 GB** (8 GB total minus ~0.8 GB CUDA context + allocator overhead).

| Model | Static | Activations (S=1024, B=4) | Activations w/ checkpointing | Total (no ckpt) | Total (ckpt) |
|-------|--------|--------------------------|------------------------------|-----------------|--------------|
| 30 M  | 0.36 GB | ~0.42 GB                 | ~0.06 GB                     | ~1.6 GB        | ~1.2 GB      |
| 125 M | 1.50 GB | ~1.24 GB                 | ~0.17 GB                     | ~3.5 GB        | ~2.5 GB      |
| 350 M | 4.20 GB | ~4.40 GB                 | ~0.60 GB                     | ~9.4 GB ❌     | ~5.6 GB ✓   |

**Conclusions:**
- **30 M and 125 M**: fit comfortably at S=1024 with batch 4, no checkpointing needed.
- **350 M at S=1024**: requires gradient checkpointing. With it, ~5.6 GB, fitting within 7.2 GB.
- **350 M at S=2048**: add ~0.4 GB activations (checkpointed) → ~6 GB. Still feasible, but tight.
- **700 M**: static alone is 8.4 GB — cannot train from scratch with standard AdamW on 8 GB even with checkpointing. Requires 8-bit optimizer (§4) or CPU offload (§5).

---

## 2. Largest Trainable Model on 8 GB

Working backwards from 7.2 GB usable VRAM with all optimizations (BF16 + gradient checkpointing + 8-bit AdamW):

```
VRAM = 2P (weights, BF16) + 2P (gradients, BF16) + 2P (optimizer 8-bit) + A_ckpt + 0.8 GB overhead
     = 6P + A_ckpt + 0.8 GB
```

With 8-bit optimizer, optimizer states drop from 8→2 bytes/param (4x reduction). So static = 6 bytes/param instead of 12.

| Max P | Static (6 B/param) | Overhead | Budget for activations | Feasible? |
|-------|-------------------|----------|------------------------|-----------|
| 1.0 B | 6.0 GB            | 0.8 GB   | 0.4 GB                 | Barely — only S=256, B=1 |
| 700 M | 4.2 GB            | 0.8 GB   | 2.2 GB                 | Yes at S=1024, B=2 |
| 500 M | 3.0 GB            | 0.8 GB   | 3.4 GB                 | Comfortable at S=1024, B=4 |

**Practical recommendation:** **350 M–500 M is the sweet spot** for 8 GB with 8-bit Adam + gradient checkpointing. Do not push to 700 M–1 B unless you also add CPU offload — training will OOM or run at batch size 1 with minimal sequence length, hurting convergence.

---

## 3. Gradient Checkpointing Deep Dive

Gradient checkpointing ([Chen et al. 2016](https://arxiv.org/abs/1604.06174), PyTorch: `torch.utils.checkpoint`) trades recomputation for memory. The original O(√n) strategy saves checkpoints at every √L layers, reducing activation memory from O(L) to O(√L) at the cost of one extra forward pass.

**PyTorch 2.1+ nested checkpointing** ([announced in PyTorch 2.1 release](https://pytorch.org/blog/pytorch-2.1-new-features/)) further reduces this to O(log n) by allowing recursive checkpointing within checkpointed segments.

**Practical numbers for a 12-layer, 768-d model (125 M params):**

| Strategy | Activation Memory | Compute Overhead |
|----------|-------------------|------------------|
| No checkpointing | ~1.24 GB (S=1024, B=4) | 1× |
| Per-layer checkpointing | ~0.17 GB | ~1.33× (one extra forward) |
| Every-2-layers | ~0.28 GB | ~1.15× |
| FlashAttention-2 (no ckpt needed for attn) | saves ~30% within attn | ~1× |

**Rule of thumb:** Gradient checkpointing saves ~5-8x activation memory at ~30% compute cost. For our 8 GB budget, enable it by default for any model ≥ 200 M params or S ≥ 1024.

```python
# PyTorch: enable per-layer gradient checkpointing
from torch.utils.checkpoint import checkpoint

# In your transformer forward pass:
def forward(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x, use_reentrant=False)  # use_reentrant=False is recommended in PyTorch 2.x
    return x
```

---

## 4. 8-Bit Optimizers (bitsandbytes)

The 8-bit Adam paper ([Dettmers et al. 2022, ICLR](https://arxiv.org/abs/2110.02861)) introduced block-wise dynamic quantization for optimizer states. Instead of storing two FP32 tensors (8 bytes/param), it stores two INT8 tensors (2 bytes/param) with per-block scaling factors — achieving **4× reduction in optimizer state memory** with negligible quality loss.

**Memory savings table:**

| Optimizer | Bytes/param (optimizer states only) | Savings |
|-----------|-------------------------------------|---------|
| AdamW FP32 | 8 bytes | baseline |
| AdamW 8-bit | 2 bytes | 4× |
| AdamW 8-bit (paged) | 2 bytes + CPU spill | ~4×+ |

For a 350 M model: standard AdamW uses 2.8 GB in optimizer states; 8-bit Adam uses 0.7 GB — freeing 2.1 GB.

**Installation and usage:**

```bash
pip install bitsandbytes
```

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

Key details from [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/main/en/optimizers):
- Tensors with fewer than 4096 elements remain in FP32 (e.g., biases, LayerNorm params). This is intentional — small tensors don't save much memory and need precision.
- The `paged` variant (`PagedAdamW8bit`) allows optimizer states to spill to CPU RAM during peak usage (useful for sporadic OOM spikes).
- Quality impact: generally negligible for pre-training; some studies note up to 0.5% perplexity increase on small datasets — acceptable for our iteration-first methodology.

**2025 alternative — Muon optimizer:** The [Muon optimizer](https://arxiv.org/abs/2502.16982) (Jordan et al., 2025) applies Nesterov momentum followed by Newton-Schulz orthogonalization (5 iterations). It claims ~10-15% better token efficiency than AdamW and has been used in production at scale (Kimi K2, GLM4.5). Memory overhead is similar to AdamW but with better per-sample generalization. Worth experimenting with once the training pipeline is stable.

---

## 5. CPU Offload

When even 8-bit Adam isn't enough, optimizer states and gradients can be moved to CPU RAM (you have 30 GB — this is a significant reserve).

### 5.1 DeepSpeed ZeRO-Offload

[DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) moves optimizer computation entirely to CPU:
- GPU holds: weights (BF16) + activations only
- CPU RAM holds: FP32 optimizer states + FP32 weights for the update step
- After update: FP32 weights cast to BF16 and sent back to GPU

**Config (minimal `ds_config.json`):**
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "bf16": { "enabled": true },
  "gradient_checkpointing": true
}
```

**Trade-off:** PCIe 4.0 bandwidth is ~16 GB/s bidirectional. Moving optimizer states for a 350 M model requires transferring ~2.8 GB (FP32 states) per update step, adding ~175 ms per step at ideal bandwidth. In practice, with overlap, overhead is 20–50% throughput reduction. For our "train fast, train many" methodology, this is only worth it for models above ~500 M params.

### 5.2 Simple CPU-Adam (no DeepSpeed)

For smaller models, you can manually offload with bitsandbytes' paged optimizer or use PyTorch's built-in offload approach:

```python
# bitsandbytes paged optimizer automatically spills to CPU RAM on OOM
optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=3e-4)
```

This is the lowest-friction option for one-off OOM issues during exploratory training.

---

## 6. RTX 3060 Ti Hardware Facts

| Spec | Value |
|------|-------|
| Architecture | Ampere (GA104) |
| CUDA cores | 4,864 |
| Tensor Cores (3rd gen) | 152 |
| FP32 TFLOPS | ~16.2 |
| BF16 Tensor TFLOPS (dense) | ~64.8 |
| BF16 Tensor TFLOPS (sparse) | ~129.6 |
| Memory | 8 GB GDDR6 |
| Memory Bandwidth | 448 GB/s |
| TDP | 200W |

**BF16 note:** Ampere added native BF16 support on tensor cores. Unlike Volta/Turing which only had FP16, Ampere handles BF16 directly in hardware — critical for training stability (BF16 has the same exponent range as FP32, avoiding the overflow issues common with FP16).

**Bandwidth vs. compute:** The RTX 3060 Ti is almost always **memory-bandwidth-bound during training**, not compute-bound, especially for small models at small batch sizes. The attention operation reads Q, K, V repeatedly from HBM — this is why FlashAttention-2 ([Dao et al. 2023](https://arxiv.org/abs/2307.08691)) provides such large gains on consumer hardware by fusing ops and reducing round-trips to memory.

**Realistic MFU:** Expect 25–45% Model FLOPs Utilization (MFU) for small models on the 3060 Ti without careful optimization, rising to 50–65% with `torch.compile` + FlashAttention-2. (For reference, the RTX 5080 with aggressive optimization reached ~70-85% MFU in a recent [nanoGPT training writeup](https://recsysml.substack.com/p/training-gpt-2-on-a-budget).)

---

## 7. Throughput Estimation and Hours-to-Train

### 7.1 FLOPs per token

For a decoder-only transformer, the standard approximation for **training FLOPs per token** is:

```
FLOP/token ≈ 6 × P     (Chinchilla paper approximation)
```

This covers the forward pass (~2P) plus backward pass (~4P, since it's ~2× forward). More precisely for self-attention with sequence length S:

```
FLOP/token = 6P + 12 × L × h × S²    (attention quadratic term)
```

At S=1024 the quadratic term is usually <5% of 6P for models up to ~350 M, so `6P` is a good working estimate.

### 7.2 Achievable throughput on 3060 Ti

At 30% MFU (conservative, standard BF16 training, no FlashAttention):
```
tokens/sec = (MFU × TFLOPS_BF16) / (6 × P_in_billions × 1e12)
           = (0.30 × 64.8e12) / (6 × P)
```

At 50% MFU (optimized: `torch.compile` + FlashAttention-2):
```
tokens/sec = (0.50 × 64.8e12) / (6 × P)
```

| Model | 30% MFU | 50% MFU |
|-------|---------|---------|
| 30 M  | ~108,000 tok/s | ~180,000 tok/s |
| 125 M | ~26,000 tok/s  | ~43,000 tok/s  |
| 350 M | ~9,300 tok/s   | ~15,400 tok/s  |

> **Important caveat:** These are theoretical estimates from the TFLOPS spec. Actual throughput depends heavily on kernel efficiency, batch size, sequence length, and whether FlashAttention-2 is installed. The RTX 5080 (which is ~2× the BF16 TFLOPS of a 3060 Ti) achieved 92,000 tok/s on a 124 M GPT-2 model in the [training on a budget writeup](https://recsysml.substack.com/p/training-gpt-2-on-a-budget). Scaling that down by ~2× for the 3060 Ti's lower compute and adjusted for architectural differences puts 125 M estimates at roughly 20,000–40,000 tok/s — consistent with the above.

### 7.3 Hours to train: Chinchilla-optimal vs. over-trained

**Chinchilla-optimal token budgets** (20 tokens per parameter):

| Model | Chinchilla tokens | Tokens at 30k tok/s | Hours (30% MFU) | Hours (50% MFU) |
|-------|------------------|---------------------|-----------------|-----------------|
| 30 M  | 600 M             | ~1.5 hrs            | ~0.9 hrs        | **~0.55 hrs**   |
| 125 M | 2.5 B             | ~26.7 hrs           | ~16 hrs         | **~9.6 hrs**    |
| 350 M | 7.0 B             | ~208 hrs            | ~125 hrs        | **~75 hrs**     |

**Over-trained "LLaMA style"** (small model, many tokens — much better inference efficiency):

The LLaMA 1 paper showed that a 7 B model trained on 1 T tokens significantly outperformed a Chinchilla-optimal model of the same compute budget. For small models aimed at being "deployed once, queried forever", over-training makes sense. A 125 M model trained on 10 B tokens (~80× Chinchilla) will be far more capable than the 2.5 B token version.

| Model | Over-trained tokens | Hours (50% MFU) |
|-------|--------------------|--------------------|
| 30 M  | 3 B (100× Chinchilla) | ~4.6 hrs  |
| 125 M | 10 B (4× Chinchilla)  | ~64 hrs   |
| 125 M | 100 B (40× Chinchilla) | ~640 hrs ← use cloud |
| 350 M | 10 B (1.4× Chinchilla) | ~180 hrs  |

**For the ETH Zurich fast-iteration methodology:** Focus on Chinchilla-ish token budgets (20–50× params) for exploratory runs. Use the 30 M model for experiments that complete in under 2 hours. Graduate to 125 M for final architecture validation.

### 7.4 Recommended batch configuration

**Effective batch size** drives learning dynamics (larger = smoother gradients, potentially faster convergence to a point). The standard recipe is ~500K tokens per optimizer step. Since we use gradient accumulation to simulate large batches:

```
tokens_per_step = micro_batch × seq_len × grad_accum_steps
```

| Model | seq_len | micro_batch | grad_accum | tokens/step | Notes |
|-------|---------|-------------|------------|-------------|-------|
| 30 M  | 1024    | 8           | 8          | 65,536      | Easily fits 8 GB |
| 125 M | 1024    | 4           | 16         | 65,536      | With checkpointing |
| 350 M | 1024    | 2           | 32         | 65,536      | Need 8-bit Adam |
| 350 M | 512     | 4           | 16         | 32,768      | No checkpointing needed |

> **2025 research note:** [Arora et al. (2025)](https://arxiv.org/abs/2507.07101) found that small batch sizes train stably and are "more robust to hyperparameter choices" than large effective batches. Don't feel compelled to reach 500K tokens/step — micro_batch=4, grad_accum=4 (16K tokens/step) is a legitimate fast-iteration configuration.

---

## 8. Engineering Checklist for 8 GB

The following settings collectively enable training on the 3060 Ti across all three target scales:

```python
# requirements
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install bitsandbytes flash-attn --no-build-isolation

import torch
import bitsandbytes as bnb

# 1. BF16 everywhere (weights, activations)
torch.set_default_dtype(torch.bfloat16)

# 2. Compile model (20-30% throughput gain on Ampere)
model = torch.compile(model, mode="reduce-overhead")

# 3. 8-bit AdamW (4x less optimizer state VRAM)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4, weight_decay=0.1)

# 4. Gradient checkpointing (in model forward pass)
#    use_reentrant=False is required for compatibility with torch.compile
from torch.utils.checkpoint import checkpoint_sequential
# or per-layer: x = checkpoint(layer, x, use_reentrant=False)

# 5. Flash Attention (memory-bandwidth savings, ~2x speedup on attention layers)
# Use torch.nn.functional.scaled_dot_product_attention (built-in since PyTorch 2.0)
# It automatically dispatches to FlashAttention on supported hardware.
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 6. Memory profiling (run this after first step to catch OOM early)
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
# Note: nvidia-smi shows 'reserved', not 'allocated'. Use torch API.
```

---

## 9. Memory Debugging Toolkit

```bash
# Real-time GPU monitoring
watch -n 0.5 nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv

# One-time snapshot
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# Python: profile peak usage per step
import torch
torch.cuda.reset_peak_memory_stats()
# ... run training step ...
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak allocated this step: {peak:.2f} GB")
```

**Common OOM causes and fixes:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| OOM on first forward pass | Activation tensors too large | Reduce B or S, enable gradient checkpointing |
| OOM on first backward pass | Gradient accumulation of large activations | Already using checkpointing? Check `use_reentrant=False` |
| OOM during optimizer step | FP32 optimizer states + BF16 weights coexist | Switch to 8-bit Adam |
| OOM after N steps (slow leak) | Accumulating tensors in a list | Detach loss: `total_loss += loss.item()` not `+= loss` |
| OOM randomly (spike) | Attention QK^T materializes large matrix | Enable FlashAttention via SDPA |

---

## 10. Summary: Practical Capacity Map

```
┌─────────────────────────────────────────────────────────────┐
│              RTX 3060 Ti (8 GB) Capacity Map                │
├──────────┬────────────────────┬──────────────────┬──────────┤
│ Model    │ Config             │ VRAM usage       │ Tok/s    │
├──────────┼────────────────────┼──────────────────┼──────────┤
│ 30 M     │ S=2048, B=8, no gc│ ~2.5 GB          │ ~150K    │
│ 125 M    │ S=1024, B=4, no gc│ ~3.5 GB          │ ~35K     │
│ 125 M    │ S=2048, B=4, gc   │ ~4.0 GB          │ ~25K     │
│ 350 M    │ S=1024, B=2, gc   │ ~5.6 GB          │ ~12K     │
│ 350 M    │ S=1024, B=4, 8b   │ ~6.5 GB          │ ~10K     │
│ 500 M    │ S=512,  B=2, 8b+gc│ ~6.8 GB          │ ~7K      │
│ 700 M    │ S=512,  B=1, 8b+gc│ ~7.5 GB ⚠        │ ~4K      │
└──────────┴────────────────────┴──────────────────┴──────────┘
gc = gradient checkpointing | 8b = 8-bit AdamW | ⚠ = very tight
```

---

## 11. Learn-by-Doing

### Experiment A: VRAM budgeting in practice (30 min)

Train a series of tiny models, increasing size until OOM. Measure actual VRAM at each size and compare to the formula predictions.

```python
# vram_budget.py
import torch
import bitsandbytes as bnb
from torch import nn

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_peak(model, batch, seq_len=512):
    device = torch.device("cuda")
    model = model.to(device).to(torch.bfloat16)
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4)
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 512, (batch, seq_len), device=device)
    loss = model(x).float().mean()
    loss.backward()
    optim.step()
    return torch.cuda.max_memory_allocated() / 1e9

# Try different model sizes
for d, L in [(256, 4), (512, 8), (768, 12), (1024, 24)]:
    model = build_your_gpt(d=d, L=L, vocab=512)
    P = count_params(model)
    predicted = 12 * P / 1e9  # static formula (no 8-bit yet)
    actual = measure_peak(model, batch=4, seq_len=512)
    print(f"P={P/1e6:.0f}M | predicted={predicted:.2f} GB | actual={actual:.2f} GB")
```

**What to observe:** How close is the formula to reality? Where does the gap come from? (Hint: activation memory, PyTorch allocator fragmentation.)

### Experiment B: Gradient checkpointing vs. throughput tradeoff (20 min)

For a fixed 125 M model, measure tokens/s and peak VRAM with and without gradient checkpointing at batch sizes 1, 2, 4, 8.

**What to observe:** At what batch size does checkpointing break even on throughput? (It often breaks even around B=4–8 on Ampere because the recomputation fits in L2 cache for smaller contexts.)

### Experiment C: 8-bit vs. FP32 optimizer convergence check (1 hour)

Train two identical 30 M runs for 100 M tokens: one with `AdamW`, one with `AdamW8bit`. Plot loss curves.

**Expected result:** Curves nearly identical. If they diverge significantly (>0.5% difference in final validation loss), something is wrong with the 8-bit config (check `min_8bit_size` and ensure embedding layers are excluded).

**Why this matters:** This is your empirical proof that 8-bit Adam is a safe default — you'll use it without guilt for all 350 M+ runs.

### Experiment D: Maximum throughput run (45 min)

Take your best 125 M config and systematically apply each optimization: baseline → +BF16 → +FlashAttention (SDPA) → +torch.compile → +gradient checkpointing (for S=2048). Log tokens/s at each stage.

**Target:** Reach ≥ 20,000 tokens/s on the 3060 Ti for 125 M at S=1024.

**Fill in this table from your results:**

| Config | tok/s | VRAM |
|--------|-------|------|
| FP32 baseline | ? | ? |
| + BF16 | ? | ? |
| + SDPA (FlashAttention) | ? | ? |
| + torch.compile | ? | ? |
| + grad checkpointing (S=2048) | ? | ? |

---

## References

- [Understanding GPU Memory Demands for Training LLMs (Shapp, Medium)](https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff)
- [8-Bit Optimizers via Block-Wise Quantization (Dettmers et al., ICLR 2022)](https://arxiv.org/abs/2110.02861)
- [bitsandbytes optimizers documentation](https://huggingface.co/docs/bitsandbytes/main/en/optimizers)
- [FlashAttention-2: Faster Attention with Better Parallelism (Dao et al. 2023)](https://arxiv.org/abs/2307.08691)
- [Training Compute-Optimal LLMs / Chinchilla (Hoffmann et al. 2022)](https://arxiv.org/abs/2203.15556)
- [Muon is Scalable for LLM Training (Liu et al. 2025)](https://arxiv.org/abs/2502.16982)
- [DeepSpeed ZeRO-Offload Tutorial](https://www.deepspeed.ai/tutorials/zero-offload/)
- [ml-engineering training performance guide (stas00)](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md)
- [Small Batch Size Training for LMs: When Vanilla SGD Works (Arora et al. 2025)](https://arxiv.org/abs/2507.07101)
- [Training GPT-2 on a Budget (RecsysML, Dec 2025)](https://recsysml.substack.com/p/training-gpt-2-on-a-budget)
- [GPU VRAM Calculator (Show HN, Hacker News)](https://news.ycombinator.com/item?id=38774026)
- [Predict PyTorch VRAM Usage: Formulas and Guide (Lyceum Technology)](https://lyceum.technology/magazine/predict-vram-usage-pytorch-model/)
- [PyTorch Gradient Checkpointing Guide](https://python-bloggers.com/2024/09/mastering-gradient-checkpoints-in-pytorch-a-comprehensive-guide/)
- [How to Use 8-Bit Optimizers in PyTorch (W&B)](https://wandb.ai/wandb_fc/tips/reports/How-To-Use-8-Bit-Optimizers-in-PyTorch--VmlldzoyMjg5MTAz)
