# Training Recipes, Optimization & Scaling Laws

> **TL;DR.** Training a small LM well in 2026 means: pick AdamW (or Muon for a potential 2x efficiency gain), use a WSD learning rate schedule so you can checkpoint and continue without restarting, run bf16 everywhere on your RTX 3060 Ti (Ampere supports it natively), use gradient checkpointing to double the sequence length you can fit, and — most importantly — **ignore Chinchilla's 20 tokens/param for small models**. Inference-optimal small models are trained 100–10,000x over Chinchilla's recommendation; SmolLM2-135M used 14,800 tokens/param (2 trillion tokens). Fast iteration matters more than any single hyperparameter: train small, iterate fast, train many models.

---

## 1. Optimizers

### 1.1 AdamW — The Reliable Default

AdamW ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) remains the workhorse of LLM pretraining. It decouples weight decay from the gradient update, which prevents the implicit L2 regularization bias that plagued the original Adam.

**Memory cost of AdamW:** For a model with `P` parameters in bf16, AdamW stores:
- Parameters: `2P` bytes (bf16)
- fp32 master copy: `4P` bytes  
- First moment `m`: `4P` bytes (fp32)
- Second moment `v`: `4P` bytes (fp32)
- **Total: ~14P bytes**

For a 125M model: `14 × 125M × 1 byte ≈ 1.75 GB` (optimizer states alone). On 8 GB VRAM this is very manageable.

**Canonical hyperparameters** (from nanoGPT, Llama, GPT-4 style training):
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,          # peak LR; adjust with sqrt(batch_size) scaling
    betas=(0.9, 0.95),  # beta2=0.95, not 0.999! Shorter memory, more responsive
    weight_decay=0.1,
    eps=1e-8,
)
```

Why `beta2=0.95` and not `0.999`? A smaller beta2 gives the second-moment estimate a shorter effective memory window (~20 steps vs ~1000 steps), allowing the optimizer to respond faster to gradient distribution shifts — important for LM training where the loss landscape shifts throughout training.

### 1.2 Muon — The New Challenger (~2x Efficiency)

[Muon](https://arxiv.org/abs/2502.16982) (MomentUm Orthogonalized by Newton-schulz) applies an orthogonalization step to the momentum gradient before each update. This approximates steepest descent under a spectral-norm trust region. Kimi K2, GLM-4.5, and Moonlight (3B/16B MoE) all adopted Muon in 2025.

**Key result:** Muon achieves ~**2x computational efficiency** compared to AdamW at compute-optimal training — meaning you reach the same loss in half the FLOPs. For a small model this is a real speedup.

**The two tricks that make Muon scalable** (from the [2502.16982 paper](https://arxiv.org/abs/2502.16982)):
1. Add weight decay (same as AdamW)
2. Carefully normalize per-parameter update scale

**Practical note:** Muon only applies to weight matrices (not embeddings, layer norms, or biases). You run AdamW on those. The `distributed_muon` implementation from the paper is memory-optimal and communication-efficient.

```python
# Pseudocode split: Muon for weight matrices, AdamW for everything else
muon_params = [p for name, p in model.named_parameters() if p.ndim >= 2 and 'embed' not in name]
adamw_params = [p for name, p in model.named_parameters() if p not in set(muon_params)]

optimizer = MuonWithAdamW(
    muon_params=muon_params, lr=0.02, weight_decay=0.01,
    adamw_params=adamw_params, adamw_lr=3e-4, adamw_weight_decay=0.1
)
```

**Trade-off:** HTMuon ([arxiv 2603.10067](https://arxiv.org/abs/2603.10067)) extends Muon with heavy-tailed spectral correction, further improving stability. Some studies note gains are *inversely proportional to model scale and training steps* — so Muon's 2x edge may shrink at larger scales but remains real for our 10M–350M targets.

### 1.3 SOAP — Second-Order for the Price of First-Order

[SOAP](https://arxiv.org/abs/2409.11321) (Shampoo with Adam in the Preconditioner's eigenbasis) runs Adam in the eigenbasis of Shampoo's preconditioner. In large-batch regimes, SOAP reduces iterations by **>40%** and wall-clock time by **>35%** vs AdamW.

However, SOAP stores and updates full preconditioner matrices per layer, which is **memory-intensive** — not ideal for 8 GB VRAM. Use SOAP only if you have large enough matrices and spare memory.

### 1.4 Adam-mini — 50% Less Optimizer Memory

[Adam-mini](https://arxiv.org/abs/2406.16793) reduces AdamW's memory footprint by **~50%** by assigning a single learning rate per Hessian-structured parameter block instead of per-element. On a 7B model it achieves 49.6% higher throughput than AdamW.

**For our 8 GB GPU this matters:** Adam-mini frees up optimizer state memory that you can spend on larger batch sizes or longer sequences.

```python
from adam_mini import Adam_mini
optimizer = Adam_mini(
    model, lr=3e-4, weight_decay=0.1,
    betas=(0.9, 0.999),  # note: uses 0.999 by default
    model_sharding=False  # single GPU
)
```

### 1.5 Sophia — Hessian-Aware (Mostly Superseded)

[Sophia](https://arxiv.org/abs/2305.14342) uses a diagonal Hessian preconditioner (clipped to prevent unstable steps). On 540M models it reached the same loss as AdamW on 770M with the same FLOPs — effectively a free 40% model-size boost. However, Sophia requires estimating the Hessian diagonal (Hutchinson estimator, ~every 10 steps), which adds overhead. In 2025, Muon largely supersedes Sophia for this use case with less complexity.

### 1.6 Optimizer Decision Tree for Your 8 GB GPU

```
Are you learning the ropes? → AdamW (simple, well-understood)
Want free 2x efficiency? → Muon + AdamW hybrid (weight matrices / rest)
Memory-constrained (>300M params)? → Adam-mini
Debugging or custom arch? → AdamW (most tooling support)
```

---

## 2. Learning Rate Schedules

### 2.1 Cosine Decay — The Old Standard

Classic cosine decay: warmup linearly to `lr_max` over `warmup_steps`, then decay as:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t/T))
```

**Fatal flaw for fast iteration:** You must pre-commit to total steps `T`. If you want to train for longer (because the loss is still dropping), you have to restart with a new schedule. You lose all the intermediate checkpoints' reusability.

### 2.2 Warmup-Stable-Decay (WSD) — The 2025 Standard

[WSD](https://arxiv.org/abs/2410.05192), popularized by MiniCPM and adopted by DeepSeek-V3 and ERNIE 4.5, splits training into three phases:

```
Phase 1 (Warmup):  Linear ramp from 0 → lr_max over ~1% of total steps
Phase 2 (Stable):  Hold at lr_max for the bulk of training (~90%)
Phase 3 (Decay):   Rapid cosine or sqrt decay to lr_min (~10% of steps)
```

**Why WSD enables fast iteration (the key insight):**

The "river valley" loss landscape framework explains it: during the stable phase, the model rapidly descends the flat manifold (exploiting the long-horizon gradients), but noisy oscillations hide the true loss improvement. When you hit the decay phase, these oscillations collapse and the loss *drops sharply* — revealing accumulated progress. 

**This means:** You can checkpoint at the end of the stable phase and run *multiple different decay anneals* from that checkpoint (e.g., decay on domain-specific data, different decay schedules, continue stable training). No restart needed. This is exactly the ETH Zurich "train many models" ethos.

```python
def wsd_schedule(step, warmup_steps, stable_steps, decay_steps, lr_max, lr_min=1e-5):
    total = warmup_steps + stable_steps + decay_steps
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    elif step < warmup_steps + stable_steps:
        return lr_max
    else:
        decay_step = step - warmup_steps - stable_steps
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * decay_step / decay_steps)
        )
```

**Typical split:** 1% warmup / 90% stable / 10% decay, or for fast experiments: 2% / 88% / 10%.

**2025 extension — WSM** ([arxiv 2507.17634](https://arxiv.org/abs/2507.17634)): instead of online decay, merge recent checkpoints with theoretically derived weights. Consistently outperforms classical WSD.

### 2.3 Warmup — How Many Steps?

Standard: **100–2000 steps** linear warmup. Rule of thumb: ~1–2% of total steps. For a 10-minute run at 50k steps, 500 warmup steps is reasonable. Too-short warmup → instability in early training; too-long → wasted compute.

### 2.4 Learning Rate Magnitude

The optimal LR follows a **power-law with model size and dataset size** ([arxiv 2503.04715](https://arxiv.org/abs/2503.04715)). Rough values from the literature:

| Model size | Typical `lr_max` |
|------------|-----------------|
| 10M–30M    | 1e-3 to 3e-3    |
| 125M       | 3e-4 to 6e-4    |
| 350M       | 2e-4 to 3e-4    |

**The sqrt-scaling rule** (Linear Scaling Rule extended): when doubling effective batch size, multiply LR by `sqrt(2)`. This is empirically more stable than the linear rule for LM training.

**muP transfer trick** ([Maximal Update Parameterization](https://www.emergentmind.com/topics/maximal-update-parameterization-mup)): tune LR on a tiny 1M proxy model, then transfer directly to 125M. Width-scaled learning rates remain stable across model sizes. With [u-muP](https://proceedings.iclr.cc/paper_files/paper/2025/file/e3130a164f5c724e37271b93bc76dd28-Paper-Conference.pdf) this even works with FP8.

---

## 3. Regularization & Stability

### 3.1 Weight Decay

**Standard:** `weight_decay=0.1` for pretraining. Applied only to weight matrices, **not** to biases, embeddings, or layer norm parameters.

Optimal weight decay scales with model and data size ([arxiv 2405.13698](https://arxiv.org/abs/2405.13698)):
- Increases with model size (following muP recommendations)
- Decreases with dataset size

For our targets: `0.1` is a robust default. If you see overfitting at <1B tokens, lower to `0.01`.

```python
# Proper AdamW weight decay: exclude embeddings, LN, biases
no_decay = ['bias', 'norm', 'embedding']
param_groups = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.1},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
]
```

### 3.2 Gradient Clipping

**Standard:** clip by global norm at `max_norm=1.0`. This is the universal choice in GPT-2, LLaMA, nanoGPT.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Gradient clipping prevents the explosive gradient instability that occurs when the model encounters outlier tokens or difficult batches. With bf16, some argue you could be more aggressive (`0.5`), but `1.0` is the safe default.

**Spectral clipping** ([arxiv 2603.14315](https://arxiv.org/abs/2603.14315)): a 2026 technique that clips per-layer spectral norms rather than the global gradient norm. Still experimental; stick to norm clipping for now.

---

## 4. Batch Size & Critical Batch Size

### 4.1 Critical Batch Size (CBS)

The CBS is the batch size above which you get **diminishing returns** from larger batches (the gradient noise becomes too low; you're wasting compute). The CBS scales primarily with **data size**, not model size ([arxiv 2412.01505](https://arxiv.org/abs/2412.01505)).

Below CBS: bigger batch = faster convergence per step. Above CBS: you're overpaying in compute for marginal improvement.

For small LMs during short runs, CBS is typically in the range of **64K–256K tokens** (batch × seq_len).

### 4.2 Effective Batch Size Targets

| Model | Typical target effective batch (tokens) | Physical batch × seq_len → grad accum |
|-------|----------------------------------------|----------------------------------------|
| 30M   | 64K–256K tokens                        | 8 × 512 = 4096 → accum 16 steps = 65K |
| 125M  | 256K–512K tokens                       | 8 × 1024 = 8192 → accum 32 steps = 262K |
| 350M  | 512K–1M tokens                         | 4 × 1024 = 4096 → accum 128 steps = 524K |

Large batches per update = better gradient estimates but slower per-sample learning. For ETH-Zurich-style fast iteration, smaller batches and more frequent updates are often preferable.

### 4.3 Gradient Accumulation

When you can't fit a large batch in VRAM, use gradient accumulation:
```python
for micro_step in range(grad_accum_steps):
    loss = model(batch[micro_step]) / grad_accum_steps
    loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**Caveat:** Gradient accumulation has zero memory benefit (gradients still accumulate in fp32), only batch-size flexibility. It **does** slow you down (extra forward+backward passes). Rule: maximize physical batch size first, use accumulation to hit target effective batch.

---

## 5. Memory Efficiency: Fitting Training in 8 GB VRAM

### 5.1 VRAM Budget for Your 3060 Ti

The RTX 3060 Ti has 8 GB GDDR6. The 3060 Ti is an Ampere card (GA104), which **natively supports bf16 and tf32** on Tensor Cores.

Memory breakdown for training:

```
Total memory = Model weights + Optimizer states + Activations + Gradients + Overhead

Model (bf16):        2P bytes
Optimizer (AdamW):   12P bytes  (fp32 master, m, v)
Gradients (bf16):    2P bytes
Activations:         variable (depends on batch, seq_len, layers)
Overhead:            ~200-500 MB

Total AdamW:  ~16P bytes + activations
Total Adam-mini: ~10P bytes + activations  (saves 6P bytes)
```

For our targets:

| Model | Params | AdamW footprint | Adam-mini | Leaves for activations |
|-------|--------|-----------------|-----------|------------------------|
| 30M   | 30M    | 480 MB          | 300 MB    | ~7.2 GB (plenty)       |
| 125M  | 125M   | 2.0 GB          | 1.25 GB   | ~5.5 GB (comfortable)  |
| 350M  | 350M   | 5.6 GB          | 3.5 GB    | ~2.0 GB (tight!)       |

For 350M with AdamW, you'll likely need gradient checkpointing + Adam-mini to fit comfortably.

### 5.2 Mixed Precision: Always bf16 on Ampere

**Choose bf16 over fp16** on RTX 3060 Ti:
- bf16 has the same dynamic range as fp32 → **no loss scaling required**, no gradient overflow
- fp16 has 5x higher precision but requires loss scaling (`GradScaler`), and overflows easily on large activations
- The RTX 30 series fully supports bf16 Tensor Core operations

```python
# Enable bf16 + tf32 (free on Ampere — uses hardware TF32 for remaining fp32 ops)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# In your training loop:
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(x, y)

# Scaler NOT needed with bf16 (unlike fp16)
loss.backward()
```

**fp8 on RTX 3060 Ti?** fp8 requires Ada/Hopper architecture (RTX 40-series, H100). The RTX 3060 Ti does **not** support fp8 in hardware. TransformerEngine's fp8 mode is not available to you — stick to bf16.

### 5.3 Gradient Checkpointing

Gradient checkpointing (a.k.a. activation checkpointing) trades compute for memory by **recomputing activations during the backward pass** instead of storing them all.

- Memory reduction: up to **sqrt(L)** (L = transformer layers) for activation memory — can cut activation memory by 50-70%
- Speed cost: ~**20% slower** per step (one extra forward pass per layer)
- PyTorch API: `torch.utils.checkpoint.checkpoint(block, x)`

For 350M models or sequences >1024 at your VRAM budget, gradient checkpointing is **strongly recommended**.

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x, use_checkpoint=True):
        if use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
```

### 5.4 torch.compile

`torch.compile` (PyTorch 2.0+) compiles your model to optimized Triton kernels via the `inductor` backend.

**Expected speedup on 8 GB consumer GPU:** 20–40% throughput increase for training, with no memory cost.

```python
model = torch.compile(model, backend='inductor')
```

**Caveats:**
- First run incurs **30–120 second compilation overhead** (JIT compilation)
- Skip for training runs shorter than 5 minutes (overhead not amortized)
- Dynamic shapes (variable seq lengths) hurt compile quality; use fixed sequence lengths
- Incompatible with some custom CUDA kernels (e.g., some FlashAttention versions require `fullgraph=False`)

```python
model = torch.compile(model, backend='inductor', fullgraph=False)  # safer setting
```

### 5.5 Full Optimization Stack for 8 GB

```python
# The complete fast-training setup for RTX 3060 Ti
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = model.to(device='cuda', dtype=torch.bfloat16)
model = torch.compile(model, backend='inductor')

optimizer = torch.optim.AdamW(
    param_groups,  # with weight decay split
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8
)

# Training loop
for step, batch in enumerate(dataloader):
    x, y = batch
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(x, y) / grad_accum_steps
    
    loss.backward()
    
    if (step + 1) % grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
```

---

## 6. Scaling Laws & Token Budgets

### 6.1 Chinchilla (2022): The Baseline

[Chinchilla](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022) established the compute-optimal scaling law: for a fixed compute budget `C` in FLOPs:

```
N_opt ≈ 0.2 * C^0.5   (optimal model params)
D_opt ≈ 20 * N        (optimal training tokens, ~20 tokens/param)
```

This says: if you have budget for 1B FLOPs, split it evenly between model size and token count.

**Critical caveat:** Chinchilla assumes a one-shot training budget where inference cost is free. This is wrong for deployed small models.

### 6.2 Inference-Optimal Scaling: Over-Train Small Models

[Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) (Sardana et al., 2024) corrects for inference cost. Key finding: **model quality continues to improve up to 10,000 tokens/param**, far beyond Chinchilla's 20.

The intuition: if you're going to run inference on a model billions of times, it's worth spending extra training compute (cheap, one-time) to get a smaller model that's cheaper to serve.

**Real-world evidence:**

| Model           | Params  | Training tokens | Tokens/param | Over-training factor |
|-----------------|---------|-----------------|--------------|---------------------|
| Chinchilla-optimal | any | 20 × N       | 20×          | 1×                  |
| LLaMA-3 (8B)   | 8B      | 15T             | 1,875×       | ~94×                |
| SmolLM2-135M   | 135M    | 2T              | 14,815×      | ~741×               |
| SmolLM2-1.7B   | 1.7B    | 11T             | 6,470×       | ~324×               |
| MiniCPM-1.2B   | 1.2B    | 1.1T            | 917×         | ~46×                |

**For your 10M–350M models:** You should plan for **100×–10,000× over-training** relative to Chinchilla. A 30M model trained on 10B tokens (333 tokens/param) is a reasonable starting point. The model won't be "compute-optimal" in the Chinchilla sense — it'll be *inference-optimal*.

### 6.3 Practical Token Budgets for Your GPU

The RTX 3060 Ti runs roughly 50,000–150,000 tokens/second for training small LMs (depending on model size and sequence length). A rough throughput table:

| Model   | Batch × Seq | Tokens/step | Approx tokens/sec | Time for 1B tokens |
|---------|-------------|-------------|-------------------|--------------------|
| 30M     | 16 × 512    | 8,192       | ~120,000 tok/s    | ~2.3 hours         |
| 125M    | 8 × 1024    | 8,192       | ~50,000 tok/s     | ~5.5 hours         |
| 350M    | 4 × 1024    | 4,096       | ~18,000 tok/s     | ~15 hours          |

*These are estimates assuming bf16, torch.compile, no gradient checkpointing. With gradient checkpointing, divide by ~1.2. Numbers are rough — measure your actual setup.*

For "can I train a useful model today?" budgets:
- **1 hour of training:** 30M model on 400M tokens (~13 tokens/param — respectable)
- **8 hours (overnight):** 125M model on 1.4B tokens (~11 tokens/param — modest) OR 30M on 3.2B tokens (~107 tokens/param — actually good!)
- **1 week continuous:** 125M on 24B tokens (~192 tokens/param) — very good small model

The ETH Zurich insight: rather than training one 125M model for a week, train **50 different 30M models** in 2–4 hours each. You'll learn more and find better hyperparameters.

### 6.4 When Does the Loss Still Improve?

The power-law loss curve: `L(N, D) ≈ A/(N^α) + B/(D^β) + C`

For small models, the data term `B/(D^β)` keeps improving for a very long time. There's no sharp "diminishing returns" cliff at 20 tokens/param. The improvements do slow, but they're real, monotonic, and worth pursuing — especially since small models are cheap to train.

---

## 7. Concrete Hyperparameter Tables

### 7.1 Target: 30M Model — Fast Iteration Mode

```yaml
# 30M GPT-style model: < 1 hour training
model:
  n_layers: 6
  d_model: 512
  n_heads: 8
  d_ff: 2048
  seq_len: 512

training:
  optimizer: AdamW
  lr_max: 1e-3
  lr_min: 1e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  grad_clip: 1.0
  
  # WSD schedule
  warmup_steps: 500      # ~1% of 50k total steps
  stable_steps: 45000
  decay_steps: 4500
  
  # Batch
  batch_size: 16         # per GPU (physical)
  seq_len: 512
  grad_accum_steps: 8    # effective batch = 16*8*512 = 65536 tokens
  
  # Memory
  precision: bf16
  torch_compile: true
  gradient_checkpointing: false  # 30M fits without it
  
  # Throughput (estimated)
  # ~120,000 tok/s → 1B tokens in ~2.3h
  # → ~33 tokens/param at 1B tokens
```

### 7.2 Target: 125M Model — Standard Run

```yaml
# 125M GPT-style model: 5-8 hour training
model:
  n_layers: 12
  d_model: 768
  n_heads: 12
  d_ff: 3072
  seq_len: 1024

training:
  optimizer: AdamW         # or Muon+AdamW for ~2x efficiency
  lr_max: 3e-4
  lr_min: 3e-5
  betas: [0.9, 0.95]
  weight_decay: 0.1
  grad_clip: 1.0
  
  # WSD schedule
  warmup_steps: 1000
  stable_steps: 90000
  decay_steps: 9000
  
  # Batch
  batch_size: 8
  seq_len: 1024
  grad_accum_steps: 32    # effective = 8*32*1024 = 262144 tokens (~256K)
  
  # Memory
  precision: bf16
  torch_compile: true
  gradient_checkpointing: false  # ~2GB model + optimizer; fits
  
  # Throughput (estimated)
  # ~50,000 tok/s → 5B tokens in ~28h (overnight: ~1.4B)
```

### 7.3 Target: 350M Model — Weekend Run

```yaml
# 350M model: this is where 8GB gets tight
model:
  n_layers: 24
  d_model: 1024
  n_heads: 16
  d_ff: 4096
  seq_len: 1024

training:
  optimizer: Adam_mini     # saves ~2.1 GB vs AdamW — crucial at this scale
  lr_max: 2e-4
  lr_min: 2e-5
  betas: [0.9, 0.95]
  weight_decay: 0.1
  grad_clip: 1.0
  
  # WSD schedule
  warmup_steps: 2000
  stable_steps: 180000
  decay_steps: 18000
  
  # Batch — must be small due to memory
  batch_size: 4
  seq_len: 1024
  grad_accum_steps: 128   # effective = 4*128*1024 = 524288 tokens (~512K)
  
  # Memory — critical
  precision: bf16
  torch_compile: true
  gradient_checkpointing: true  # essential at this size!
  
  # Throughput (estimated with gradient_checkpointing)
  # ~15,000 tok/s → 1B tokens in ~18.5h
  # Note: the 128-step accumulation adds real overhead; consider
  # smaller grad_accum with Adam-mini's memory savings
```

**Memory verification for 350M with gradient checkpointing + Adam-mini:**
- Model (bf16): 350M × 2 = 700 MB
- Adam-mini states: 350M × 8 ≈ 2.8 GB (vs AdamW's 4.2 GB)
- Gradients (bf16): 700 MB
- Activations (with checkpointing, batch=4, seq=1024): ~1.0–1.5 GB
- CUDA overhead: ~400 MB
- **Total: ~5.6–6.1 GB** — fits in 8 GB with ~1.9 GB headroom.

---

## 8. Putting It All Together: Training Flow Diagram

```
1. Initialize model in bf16
       ↓
2. Compile with torch.compile(backend='inductor')
       ↓
3. WSD schedule: ramp LR → stable at lr_max
       ↓
4. Training loop:
   for each micro-batch:
     a. autocast(bf16) forward pass
     b. loss.backward()
     c. [if grad_accum_steps reached]:
        - clip_grad_norm(1.0)
        - optimizer.step()
        - scheduler.step()
        - zero_grad(set_to_none=True)
       ↓
5. Checkpoint at end of Stable phase
       ↓  ↘
6. Run Decay phase    (or) Continue Stable phase with more data
   (your deployed model)     (free re-use of checkpoint)
```

---

## 9. Learn-by-Doing

### Experiment 1: "The Schedule Showdown" (30 min)

Train the same 30M model to 50M tokens with three schedules: cosine, linear, and WSD. Compare final validation loss and loss curve shape. You should see the WSD loss "step down" sharply at the decay phase.

```bash
# Run A: cosine
python train.py --schedule cosine --model 30m --tokens 50M --tag cosine

# Run B: linear decay
python train.py --schedule linear --model 30m --tokens 50M --tag linear

# Run C: WSD
python train.py --schedule wsd --wsd_decay_frac 0.1 --model 30m --tokens 50M --tag wsd
```

**What to observe:** Plot all three loss curves on the same axis. Note when WSD's loss visibly drops. This is your first encounter with the river-valley phenomenon.

### Experiment 2: "The Optimizer Horse Race" (2 hours)

Train four versions of a 30M model to the same token count (200M tokens). Use: AdamW, Adam-mini, and if available, Muon. Hold everything else constant.

**Measure:** Tokens/second (throughput) and final validation perplexity. Is Muon actually faster to a given loss? Does Adam-mini match AdamW at half the memory?

**Prediction to test:** AdamW and Muon should reach the same loss, but Muon should get there faster (fewer steps). Adam-mini should match AdamW with measurably lower VRAM usage (`torch.cuda.max_memory_allocated()`).

### Experiment 3: "Over-Training Pays Off" (4 hours)

Train three 30M models:
- **A:** 600M tokens (20 tokens/param — Chinchilla-optimal)
- **B:** 3B tokens (100 tokens/param)
- **C:** 9B tokens (300 tokens/param)

Evaluate all three on the same held-out perplexity benchmark.

**Expected result:** A < B < C in quality (lower perplexity). Model C will be visibly better on quality benchmarks. This is the empirical proof that over-training small models works.

### Experiment 4: "WSD Checkpoint Reuse" (1.5 hours)

This is the ETH Zurich trick in action. Train a 30M model to a "stable checkpoint" (e.g., 80% of total tokens, still in WSD's stable phase). Then run **two different decay anneals** from that single checkpoint:

- **Decay A:** Standard cosine decay for 10% more tokens
- **Decay B:** Linear decay with 2x more tokens (longer anneal)

Compare final validation loss. You've now trained **two separate model checkpoints** for the cost of 1.1× training. This is the fast-iteration superpower of WSD.

---

## 10. Key Papers & Resources

| Resource | URL | Why Read It |
|----------|-----|-------------|
| Muon is Scalable for LLM Training (2025) | [arxiv 2502.16982](https://arxiv.org/abs/2502.16982) | 2x efficiency over AdamW, scalability fixes |
| Understanding WSD Schedule (2024) | [arxiv 2410.05192](https://arxiv.org/abs/2410.05192) | River-valley intuition, reusable checkpoints |
| Beyond Chinchilla-Optimal (2024) | [arxiv 2401.00448](https://arxiv.org/abs/2401.00448) | Why you should over-train small models |
| MiniCPM: Scalable Small LM Training (2024) | [arxiv 2404.06395](https://arxiv.org/abs/2404.06395) | WSD in practice, higher compute-optimal data ratio |
| SmolLM2 Paper (2025) | [arxiv 2502.02737](https://arxiv.org/abs/2502.02737) | 14,800 tokens/param for 135M — serious over-training |
| Adam-mini (2024) | [arxiv 2406.16793](https://arxiv.org/abs/2406.16793) | 50% memory reduction, near-AdamW quality |
| SOAP Optimizer (2024) | [arxiv 2409.11321](https://arxiv.org/abs/2409.11321) | Second-order efficiency, large-batch regime |
| Sophia Optimizer (2023) | [arxiv 2305.14342](https://arxiv.org/abs/2305.14342) | Hessian-based; 50% faster to same loss as AdamW |
| HuggingFace Single-GPU Training Guide | [hf.co/docs/transformers/perf_train_gpu_one](https://huggingface.co/docs/transformers/perf_train_gpu_one) | Practical bf16/gradient checkpointing setup |
| Predictable Scale: Optimal HP Scaling Law | [arxiv 2503.04715](https://arxiv.org/abs/2503.04715) | LR and batch size follow power laws with N and D |
| Weight Decay Scaling (2024) | [arxiv 2405.13698](https://arxiv.org/abs/2405.13698) | How weight decay should scale with model/data size |
| WSM: Decay-Free LR via Checkpoint Merging | [arxiv 2507.17634](https://arxiv.org/abs/2507.17634) | WSD extension: merge checkpoints instead of online decay |

---

## Quick Reference Card

```
OPTIMIZER DEFAULTS (for any size, copy-paste starting point):
  AdamW: lr=3e-4, betas=(0.9, 0.95), wd=0.1, clip=1.0
  Adam-mini: same LR, wd=0.1 (for ≥350M where VRAM is tight)
  Muon: lr=0.02 (weight matrices) + AdamW lr=3e-4 (rest)

SCHEDULE (WSD):
  Warmup: 1% of steps (linear 0 → lr_max)
  Stable: 89% of steps (constant lr_max)
  Decay:  10% of steps (cosine lr_max → lr_min, where lr_min = 0.1 × lr_max)

TOKENS/PARAM TARGETS:
  Minimum useful:    100×   (better than random at tasks)
  Good small model:  500×   (SmolLM/MiniCPM territory)
  Excellent:        3000×+  (SmolLM2-135M at 14,800×)

MEMORY TRICKS (8 GB VRAM, ordered by impact):
  1. bf16 everywhere (2× over fp32, free on RTX 30-series)
  2. Adam-mini instead of AdamW (2× optimizer savings)
  3. gradient_checkpointing (50-70% activation savings, 20% speed cost)
  4. gradient accumulation (no memory saving, but larger effective batch)
  5. torch.compile (20-40% throughput boost, no memory cost)
  6. set_to_none=True in zero_grad (tiny but free)
```
