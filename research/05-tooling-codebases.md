# Tooling, Codebases & 8GB Practical Setup

**TL;DR:** You do not need 16 nodes of A100s to train a language model from scratch. The ecosystem in 2025-2026 gives a single RTX 3060 Ti owner a complete, modern stack: reference architectures (nanoGPT/nanochat), optimizer research (modded-nanoGPT/Muon), production-grade frameworks (litGPT, TRL), and low-level control (llm.c). This document maps each codebase to what you should *learn* from it vs. what you should *borrow*, then gives you the exact installation recipe for your 8 GB Ampere machine and a minimal-but-modern scratch repo layout.

---

## Part 1: Reference Codebases

### 1. nanoGPT — Karpathy (Deprecated but Required Reading)

**What it is:** [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — ~600 lines of PyTorch, two files (`model.py` + `train.py`), that implement a GPT-2-style decoder-only transformer and a training loop. Karpathy also released [build-nanogpt](https://github.com/karpathy/build-nanogpt), a video + code walkthrough building it from scratch.

**Status (2025):** Karpathy has declared nanoGPT "deprecated" in favor of nanochat (see below), but it remains the single best starting point for internalizing transformer training from first principles.

**What to learn from it:**
- The minimal training loop: data loading, forward pass, loss, backward, clip-gradients, step, log — in ~150 lines with no framework magic hiding anything.
- How `torch.compile` and `ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)` fit into real code.
- The difference between GPT-2's position embedding and modern RoPE-based alternatives.
- Gradient accumulation as a substitute for large batch sizes when VRAM is limited.

**Build from scratch vs. reuse:** Read and retype it once. Do not build on top of it for new projects — the architecture is pre-2023. But the training loop pattern transfers directly.

**8 GB note:** nanoGPT's GPT-2 (124M) fits fine with bf16 + gradient checkpointing. A 10M–50M parameter model trains comfortably at batch size 32–64.

---

### 2. nanochat — Karpathy (Current, October 2025)

**What it is:** [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) — Released October 13, 2025. Described by Karpathy as "among the most unhinged I've written." ~8,000 lines of PyTorch. The full end-to-end pipeline from byte-pair tokenization through pretraining, mid-training (SmolTalk dataset), supervised fine-tuning (SFT on GSM8K), optional GRPO reinforcement learning, to a web chat UI.

**Pipeline stages:**
1. **Tokenizer:** Custom Rust BPE with 65,536-token vocabulary, trained on FineWeb-EDU shards.
2. **Pretraining:** Transformer on FineWeb; evaluates a CORE score.
3. **Mid-training:** SmolTalk dataset for conversational ability.
4. **SFT:** Instruction following + math reasoning (GSM8K).
5. **RL (optional):** GRPO on GSM8K.
6. **Inference:** CLI + web UI.

The recommended hardware is an 8×H100 node (~$100 at spot prices for a 4-hour run). On your 3060 Ti, you will adapt this to much smaller model sizes, but the pipeline architecture is the canonical reference for what "complete training" looks like in 2025.

**What to learn from it:**
- How all the post-training stages chain together.
- What a `--depth` parameter-based architecture dial looks like in practice.
- How to write a minimal BPE tokenizer.
- GRPO implementation without a separate critic model (memory-friendly vs. PPO).

**Build from scratch vs. reuse:** Study the architecture and pipeline design. For your 8 GB machine, extract the model definition and training loop; skip the 8×H100 assumptions in the launch scripts.

---

### 3. modded-nanoGPT — Keller Jordan (Speedrun Benchmark, Muon)

**What it is:** [github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — A competitive benchmark where researchers minimize the time to reach ≤3.28 validation loss on FineWeb with a 124M-parameter model on 8×H100 GPUs. The baseline was 45 minutes (from llm.c); as of May 2026, the record stands at **1.328 minutes** — a 34× speedup in about 18 months.

**Key innovations catalogued by this repo:**
- **Muon optimizer** ([Keller Jordan's post](https://kellerjordan.github.io/posts/muon/)): Takes momentum gradients for 2D weight matrices and orthogonalizes them via Newton-Schulz iteration (5 steps, polynomial coefficients `3.4445, -4.7750, 2.0315`). The result is the nearest semi-orthogonal matrix to the update. On a 1.5B transformer, Muon achieves GPT-2-XL HellaSwag performance in 10 hours vs. 13.3 hours with AdamW — a 1.35× speedup with ~0.5% overhead. Muon held the record for 12 consecutive speedrun iterations.
- **Architectural improvements that made the leaderboard:** RoPE (rotary position embeddings), QK-Norm, ReLU² activation, value embeddings, skip connections with gating, multi-token prediction heads, sliding window Flash Attention.
- **Systems-level:** Fused Triton kernels, FP8 matrix multiplications, batch size scheduling, async data loading.

**What to learn from it:**
- Muon is the optimizer to try for any 2D weight matrix (attention projections, MLP layers). Use AdamW only for embeddings and scalars.
- QK-Norm (LayerNorm on Q and K before attention) is cheap and stabilizes training at low learning rates.
- ReLU² (squared ReLU) consistently outperforms GeLU and SiLU in the small model regime.
- The speedrun discussion thread ([#23](https://github.com/KellerJordan/modded-nanogpt/discussions/23)) is a gold mine of ablations.

**Build from scratch vs. reuse:** Read every commit message in the leaderboard history. Borrow the Muon implementation directly — it is ~50 lines of PyTorch and plugs into any training loop.

---

### 4. litGPT — Lightning AI

**What it is:** [github.com/Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) — Production-ready implementations of 20+ LLM architectures (Llama 3, Mistral, Phi-3, Falcon, Gemma 2, etc.) built on Lightning Fabric. Supports pretraining, continued pretraining, SFT, LoRA/QLoRA finetuning, evaluation, and deployment. Was the official starter kit for the NeurIPS 2023 LLM Efficiency Challenge (finetune one model, 24 hours, one GPU) and powered the original TinyLlama project.

**What to learn from it:**
- How a properly abstracted training script handles distributed training (DDP, FSDP) without locking you into a single path.
- Production-quality data loading with `PackedDataset` for token-packed sequences.
- LoRA implementation that can be surgically applied to any weight matrix.
- YAML-based config system for sweeping hyperparameters.

**Build from scratch vs. reuse:** For your 8 GB setup, litGPT is the right framework if you want to finetune or run continued pretraining on an existing checkpoint (e.g., SmolLM2-135M). For pure from-scratch pretraining of a tiny model, it adds more abstraction than necessary, but the architecture files (`litgpt/model.py`) are the cleanest modern Llama implementation available.

**VRAM math for 3060 Ti with litGPT:**
- SmolLM2-135M (Hugging Face) in bf16: ~270 MB weights. With optimizer states (AdamW, 2× param count in fp32): ~1 GB total. Batch 64, seq 2048: activations ~500 MB. Full run comfortably under 4 GB.
- 350M model in bf16: ~700 MB weights, optimizer ~2.8 GB, activations ~1 GB = ~4.5 GB, fits with gradient checkpointing.

---

### 5. TinyLlama

**What it is:** [github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) — The training codebase for the 1.1B parameter Llama-2-architecture model trained on 3T tokens. Built on litGPT + FSDP on 16 nodes of 4×A100-40G GPUs. Paper: [TinyLlama: An Open-Source Small Language Model (2024)](https://arxiv.org/abs/2401.02385).

**Architecture (directly reusable for smaller variants):**
- 22 transformer layers, hidden size 2048, 16 attention heads, GQA (4 KV heads), FFN dim 5632.
- RoPE positional embeddings, RMSNorm (no bias), SwiGLU activation, 32K vocabulary.
- Flash Attention 2 throughout.

**What to learn from it:**
- This is the canonical reference architecture for a sub-2B model. If you want to build a 50M or 150M model, shrink TinyLlama's architecture (fewer layers, smaller hidden dim) rather than inventing a new architecture.
- The paper's analysis of training instabilities at low learning rates and high token counts is practically useful.
- GQA (Grouped Query Attention) is worth using even at 50M params: it reduces KV cache size proportionally to the group ratio, improving inference speed on limited VRAM.

**Build from scratch vs. reuse:** Study the architecture config. Do not try to run the training code directly (it assumes FSDP multi-node). Use litGPT with TinyLlama's architecture instead.

---

### 6. llm.c — Karpathy

**What it is:** [github.com/karpathy/llm.c](https://github.com/karpathy/llm.c) — LLM training in raw C and CUDA, no PyTorch dependency. The reference CPU fp32 implementation is ~1,000 lines. Reproduces GPT-2 (124M to 1.558B). Currently ~7% faster than PyTorch Nightly for the same model. Full reproduction of GPT-2 1.558B took 24 hours on 8×H100 at $672 (circa 2024). As of June 2025, still actively maintained.

**What to learn from it:**
- Concrete understanding of what happens inside `loss.backward()` — each kernel, each memory layout, each synchronization point.
- How attention, LayerNorm, softmax, and cross-entropy are implemented in raw CUDA.
- Memory bandwidth vs. compute analysis: why attention is memory-bound at small batch sizes.
- The encoding of a training loop in ~1,000 lines forces you to understand the exact computation graph.

**Build from scratch vs. reuse:** This is a *learning* codebase for you, not a daily-driver. Spend one afternoon reading `train_gpt2.c` carefully. You will understand PyTorch's behavior more deeply afterward. Do not build your experiments on top of it.

---

### 7. Hugging Face Ecosystem (transformers / TRL / nanotron)

**transformers:** [github.com/huggingface/transformers](https://github.com/huggingface/transformers) — The standard library for loading, evaluating, and finetuning models. For your purposes: use `AutoModelForCausalLM` + `AutoTokenizer` for evaluation and SFT, but avoid using its `Trainer` for pretraining from scratch (too opaque).

**TRL (Transformer Reinforcement Learning):** [github.com/huggingface/trl](https://github.com/huggingface/trl) — As of TRL v1.0 (April 2026), the unified post-training stack for SFT, reward modeling, DPO, and GRPO. Supports small models down to Qwen2.5-0.5B-Instruct. Key trainers:
- `SFTTrainer`: For instruction tuning after pretraining. Handles data packing, LoRA injection.
- `GRPOTrainer`: GRPO removes the separate critic (unlike PPO), halving memory requirements. Crucial for 8 GB VRAM.
- `DPOTrainer`: Direct Preference Optimization; used in Llama 3's post-training.

**nanotron:** [github.com/huggingface/nanotron](https://github.com/huggingface/nanotron) — HuggingFace's distributed pretraining framework with 3D parallelism (data, tensor, pipeline). Designed for multi-node scale. For a single 3060 Ti, this is overkill for training but worth reading for understanding how production-scale pretraining is organized. The [ultrascale-playbook](https://github.com/huggingface/nanotron/tree/main/docs) documents detailed benchmarks.

**datatrove:** [github.com/huggingface/datatrove](https://github.com/huggingface/datatrove) — Platform-agnostic data processing pipelines for LLM training data (reading, filtering, deduplication, tokenization). Used to process FineWeb and similar datasets. For your 8 GB machine: use datatrove locally to tokenize a text dataset into binary shard files once, then train from those shards. Low memory usage, runs fine on CPU.

---

### 8. PufferLib (Methodology Link)

**What it is:** [github.com/PufferAI/PufferLib](https://github.com/PufferAI/PufferLib) — A fast reinforcement learning library by Joseph Suarez that trains at 300K–1.2M RL steps per second on a single GPU. Its relevance here is *methodological*, not architectural.

**The "PufferLib / ETH Zurich" methodology for LM training:**
The philosophy — "train many small models fast rather than one big model slow" — is most associated with Joseph Suarez (@jsuarez5341) and popularized for LM work by @yacineMTB and @itsreallyvivek. The core ideas:
1. Keep training runs under 5 minutes on your local GPU.
2. Each run tests exactly one variable.
3. Run 10–20 experiments per day rather than 1 per week.
4. Build the experimental harness before the model.
5. "The more models you train, the more you learn."

This is the same epistemology as Y-combinator's "make something people want" applied to model research: tight feedback loops over grand plans. nanoGPT speedrunning operationalizes this for the broader community.

---

## Part 2: The 8 GB Setup — Exact Install Recipe

### Hardware Profile

Your GPU: **NVIDIA RTX 3060 Ti** — Ampere architecture, compute capability **sm_86**, 8 GB GDDR6, 448 GB/s memory bandwidth, 16.2 TFLOPS bf16 tensor core throughput. This is a real research GPU for sub-500M models.

Key architectural facts:
- Ampere supports **bf16 Tensor Core** operations natively (no fp16/fp32 accuracy tradeoffs).
- Supports **Flash Attention 2** (requires sm_80+, which Ampere satisfies).
- Does **not** support Flash Attention 3 (requires sm_90, i.e., Hopper H100).
- `torch.compile` works fully on sm_86 with CUDA 12.x.

### Step 0: System Prerequisites

```bash
# Check your CUDA driver version (must be >= 525 for CUDA 12.x)
nvidia-smi

# Install CUDA Toolkit 12.4 system-wide (if not present)
# Download from: https://developer.nvidia.com/cuda-12-4-0-download-archive
# Or use the runfile for Linux x86_64. Alternatively, PyTorch ships its own CUDA runtime.
```

### Step 1: Install uv (Python env manager)

[uv](https://github.com/astral-sh/uv) by Astral — a Rust-based replacement for pip + virtualenv + pyenv. 10–100× faster than pip. As of 2026, version ~0.9.x.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # or restart shell

# Create a project-local venv with Python 3.11 (recommended over 3.13 for ML due to ecosystem maturity)
cd /home/ricardo/dev/small-tank
uv venv .venv --python 3.11
source .venv/bin/activate

# Verify
python --version  # Python 3.11.x
```

Note on Python 3.13: The ecosystem (flash-attn, bitsandbytes) is better tested on 3.11 or 3.12 as of mid-2026. Use 3.11 to avoid friction.

### Step 2: PyTorch with CUDA 12.4

```bash
# PyTorch 2.7.x with CUDA 12.4 (cu124 wheels bundle CUDA runtime — no system CUDA needed)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify GPU is visible and bf16 works
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'bf16 support: {torch.cuda.is_bf16_supported()}')
t = torch.ones(1, dtype=torch.bfloat16, device=\"cuda\")
print(f'bf16 tensor on GPU: OK')
"
```

Expected output:
```
PyTorch: 2.7.x+cu124
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Ti
VRAM: 8.0 GB
bf16 support: True
bf16 tensor on GPU: OK
```

### Step 3: Flash Attention 2

Flash Attention compiles from source (15–45 min) or installs from prebuilt wheels in seconds. Use prebuilt wheels:

```bash
# Option A: Use flashattn.dev prebuilt wheel finder
# Visit https://flashattn.dev/ and select:
#   Python: 3.11, PyTorch: 2.7, CUDA: 12.4, OS: Linux
# Then pip install the URL it gives. Example:
uv pip install flash-attn --no-build-isolation \
  --find-links https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/

# Option B: Install from the official pre-built wheels repo
# Check https://github.com/Dao-AILab/flash-attention/releases for latest
# Example (replace version + Python/PyTorch/CUDA suffix to match your stack):
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4+cu124torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Verify
python -c "from flash_attn import flash_attn_func; print('Flash Attention 2: OK')"
```

Note: Flash Attention 3 requires sm_90 (H100). On your 3060 Ti (sm_86), Flash Attention 2 is what you use.

### Step 4: Core Training Libraries

```bash
# Essential ML tools
uv pip install \
  numpy \
  einops \
  tiktoken \
  sentencepiece \
  transformers \
  datasets \
  accelerate \
  wandb \
  tensorboard

# bitsandbytes — 8-bit optimizers + quantization
# Note: for *training* small models from scratch, bitsandbytes' main value is
# the 8-bit AdamW optimizer (saves ~50% optimizer memory vs. fp32 Adam states)
uv pip install bitsandbytes

# datatrove — data pipeline for tokenizing training corpora
uv pip install datatrove[processing]

# TRL — for SFT and GRPO post-training stages
uv pip install trl

# Triton — for custom CUDA kernels (used by Muon, Flash Attn internals)
uv pip install triton
```

### Step 5: Development Tools

```bash
uv pip install \
  ipython \
  rich \                    # better terminal output
  typer \                   # CLI for training scripts
  pydantic \                # config validation
  pytest \                  # tests
  ruff                      # fast linter

# Weights & Biases setup (free tier sufficient for individual research)
wandb login  # enter your API key from wandb.ai
```

### VRAM Budget Math

Mental model for 8 GB training:

| Component | Formula | Example: 50M model, seq=1024, batch=8 |
|-----------|---------|----------------------------------------|
| Model weights | params × 2 bytes (bf16) | 50M × 2 = 100 MB |
| Optimizer states (AdamW) | params × 8 bytes (fp32 m+v+master) | 50M × 8 = 400 MB |
| Gradients | params × 2 bytes (bf16) | 50M × 2 = 100 MB |
| Activations (no grad ckpt) | ~batch × seq × layers × hidden × 2 | 8 × 1024 × 12 × 512 × 2 ≈ 100 MB |
| **Total** | | **~700 MB** — fits easily |

For a 150M model, seq=2048, batch=16: approximately 3–4 GB total. Comfortable.

For a 350M model, seq=2048, batch=8: approximately 6–7 GB. Use `torch.compile` + gradient checkpointing.

Key lever: **bitsandbytes 8-bit AdamW** (`bnb.optim.AdamW8bit`) cuts optimizer states from 400 MB to 100 MB for a 50M model. For larger runs, this is the first optimization to apply.

---

## Part 3: Recommended Repo Layout

Build your scratch repo with this structure. It is minimal but supports the full training → evaluation → post-training cycle:

```
small-tank/
├── model/
│   ├── __init__.py
│   ├── config.py          # ModelConfig dataclass: n_layers, n_heads, d_model, etc.
│   ├── transformer.py     # Pure PyTorch: Attention, MLP, Block, Transformer
│   ├── rope.py            # Rotary Position Embeddings
│   └── muon.py            # Muon optimizer (~50 lines, borrow from modded-nanoGPT)
│
├── data/
│   ├── tokenize.py        # Runs datatrove pipeline → .bin shard files
│   ├── loader.py          # MemoryMappedDataset: reads shards, returns token tensors
│   └── shards/            # Output: train_000.bin, train_001.bin, val.bin
│
├── train.py               # Main training script (~300 lines)
│                          # - argparse / typer config
│                          # - wandb.init()
│                          # - DataLoader → model → Muon/AdamW → loss loop
│                          # - torch.compile, bf16 autocast
│                          # - checkpoint save/load
│
├── eval.py                # Perplexity on val set; optional lm-eval-harness call
│
├── sample.py              # Greedy / top-p sampling from a checkpoint
│
├── finetune/
│   ├── sft.py             # SFT via TRL SFTTrainer on instruction data
│   └── grpo.py            # GRPO via TRL GRPOTrainer
│
├── configs/
│   ├── 10m.yaml           # 10M param config: layers=6, d_model=256, heads=8
│   ├── 50m.yaml           # 50M param config: layers=12, d_model=512, heads=16
│   └── 150m.yaml          # 150M param config: layers=18, d_model=768, heads=12
│
├── experiments/           # One folder per run: config copy + wandb run id + notes
│   └── 2026-06-13-muon-vs-adamw/
│
├── pyproject.toml         # uv-managed deps
├── .python-version        # 3.11
└── research/              # This folder: reference docs
```

**`train.py` minimal skeleton (the critical parts):**

```python
import torch
import wandb
from model.transformer import Transformer
from model.config import ModelConfig
from model.muon import Muon
from data.loader import MemoryMappedDataset

# ---- Config ----
cfg = ModelConfig(n_layers=12, d_model=512, n_heads=16, vocab_size=50257, seq_len=1024)
model = Transformer(cfg).cuda()
model = torch.compile(model)  # ~20-40% speedup on Ampere

# ---- Optimizer: Muon for weight matrices, AdamW for everything else ----
muon_params = [p for name, p in model.named_parameters() if p.ndim == 2]
adam_params  = [p for name, p in model.named_parameters() if p.ndim != 2]
optimizer = [Muon(muon_params, lr=0.02), torch.optim.AdamW(adam_params, lr=3e-4)]

# ---- Training loop ----
ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
for step, (x, y) in enumerate(loader):
    x, y = x.cuda(), y.cuda()
    with ctx:
        logits, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for opt in optimizer: opt.step(); opt.zero_grad()
    wandb.log({'loss': loss.item(), 'step': step})
```

**`pyproject.toml` (uv-managed):**

```toml
[project]
name = "small-tank"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.7.0",
    "flash-attn>=2.7.0",
    "transformers>=4.47.0",
    "datasets>=3.2.0",
    "bitsandbytes>=0.45.0",
    "datatrove[processing]>=0.4.0",
    "trl>=1.0.0",
    "wandb>=0.19.0",
    "einops>=0.8.0",
    "tiktoken>=0.8.0",
    "rich>=13.9.0",
    "typer>=0.15.0",
]
```

---

## 4. Experiment Tracking: wandb vs. Alternatives

**Weights & Biases (wandb):** The standard for ML research. Free tier supports unlimited runs, 100 GB storage. Call `wandb.init(project="small-tank", config=cfg_dict)` at the top of `train.py` and `wandb.log({'loss': loss.item(), 'tokens_seen': tokens})` inside the loop. The dashboard gives you loss curves, GPU utilization, hyperparameter sweeps (W&B Sweeps), and reproducibility via run configs. Recommended.

**TensorBoard:** Zero friction, no account needed. `from torch.utils.tensorboard import SummaryWriter`. Good for offline use. Less feature-rich than wandb for comparison across many runs.

**For the ETH Zurich fast-iteration method:** Log every run, no matter how short. When you train 20 models in a day, wandb's comparison tables become invaluable. Name runs descriptively: `muon-lr0.02-depth12` vs. `adamw-lr3e-4-depth12`.

---

## 5. Learn-by-Doing

### Experiment 1: Clone and Time nanoGPT on Your Machine

Clone [build-nanogpt](https://github.com/karpathy/build-nanogpt), prepare the Shakespeare dataset (~1 MB), train a 10M parameter GPT on it. Target: loss < 1.5 in under 2 minutes on your 3060 Ti. Measure tokens/second and peak VRAM. This is your hardware baseline.

```bash
git clone https://github.com/karpathy/build-nanogpt
cd build-nanogpt
python data/shakespeare_char/prepare.py
# Edit train_gpt2.py: set n_layer=6, n_embd=256, n_head=8
# Add torch.compile
python train_gpt2.py
```

Log: tokens/second, VRAM used, loss at 1K steps. This number is your machine's capability anchor.

### Experiment 2: Drop in Muon, Compare to AdamW

Take the nanoGPT training loop. Copy the Muon implementation from [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt). Run the same 10M model for 2000 steps with AdamW (lr=3e-4), then with Muon (lr=0.02 for 2D params) + AdamW for embeddings/scalars. Plot validation loss vs. wall-clock time. Does Muon converge faster on your GPU? Record the speedup ratio.

### Experiment 3: VRAM Budget Profiling

Train a model, and profile its actual VRAM use vs. your theoretical estimate using `torch.cuda.max_memory_allocated()`. Try: (a) no optimization, (b) bf16 autocast, (c) `torch.compile`, (d) gradient checkpointing, (e) bitsandbytes 8-bit AdamW. Make a table:

| Setting | Peak VRAM | Tokens/sec |
|---------|-----------|-----------|
| fp32 + no compile | ? | ? |
| bf16 | ? | ? |
| bf16 + compile | ? | ? |
| bf16 + compile + grad ckpt | ? | ? |
| bf16 + compile + grad ckpt + bnb 8bit | ? | ? |

This experiment will permanently calibrate your intuition for what fits in 8 GB.

### Experiment 4: End-to-End Pipeline in One Afternoon

Use datatrove to tokenize 50 MB of text (e.g., the first 10 FineWeb shards or a Wikipedia dump). Train a 50M model for 30 minutes with wandb logging. Then run TRL SFTTrainer on a 1000-example instruction dataset (e.g., a small subset of UltraChat). Sample from the base model and the SFT model and compare outputs. This closes the loop from raw text to a chat-capable model on your single GPU.

---

## Quick Reference: Decision Matrix

| Goal | Use This |
|------|----------|
| Learn transformer training from scratch | nanoGPT + build-nanogpt video |
| Learn the full chat pipeline (pretraining → GRPO) | nanochat |
| Get the best optimizer for your training runs | Muon from modded-nanoGPT |
| Finetune an existing checkpoint with LoRA | litGPT or TRL + transformers |
| Understand what PyTorch is doing internally | llm.c (read, don't run) |
| Process a large text dataset into training shards | datatrove |
| Run SFT / DPO / GRPO post-training | TRL v1.0 |
| Track 20+ experiments per day | wandb |
| Manage Python environments | uv |

---

## Sources

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [karpathy/nanochat](https://github.com/karpathy/nanochat)
- [Karpathy on nanochat (X post)](https://x.com/karpathy/status/1977755427569111362)
- [Nanochat: The best ChatGPT $100 can buy (discussion)](https://github.com/karpathy/nanochat/discussions/1)
- [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [Muon optimizer (Keller Jordan blog)](https://kellerjordan.github.io/posts/muon/)
- [Speedrunning ideas discussion #23](https://github.com/KellerJordan/modded-nanogpt/discussions/23)
- [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt)
- [TinyLlama paper (2024)](https://arxiv.org/abs/2401.02385)
- [karpathy/llm.c](https://github.com/karpathy/llm.c)
- [huggingface/trl](https://github.com/huggingface/trl)
- [TRL v1.0 release (April 2026)](https://www.marktechpost.com/2026/04/01/hugging-face-releases-trl-v1-0-a-unified-post-training-stack-for-sft-reward-modeling-dpo-and-grpo-workflows/)
- [huggingface/nanotron](https://github.com/huggingface/nanotron)
- [huggingface/datatrove](https://github.com/huggingface/datatrove)
- [PufferAI/PufferLib](https://github.com/pufferai/pufferlib)
- [Dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention)
- [flash-attn prebuilt wheels](https://flashattn.dev/)
- [mjun0812 prebuilt flash-attention wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
- [astral-sh/uv](https://github.com/astral-sh/uv)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [HuggingFace single GPU training guide](https://huggingface.co/docs/transformers/main/perf_train_gpu_one)
- [Squeezing 1-2% from Muon (Cesista et al.)](https://leloykun.github.io/ponder/muon-opt-coeffs/)
- [The Automated LLM Speedrunning Benchmark (2025)](https://arxiv.org/pdf/2506.22419)
