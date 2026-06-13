# Lesson 1 — Foundations

> **You already built a working language model.** In ~7 minutes you trained a 5M-param transformer from
> scratch that writes coherent stories (RUN 002). This lesson makes you *understand* it — paper by paper,
> mapped to the exact code you wrote and the runs you logged. Two tracks: **A — AI Research** (the ideas /
> *why* it works) and **B — AI Engineering** (the machinery / *how* to train it). Budget ~1 week. Write every
> prediction in [`research/log.md`](../research/log.md) **before** you run.

---

## Track A — AI Research (the ideas)

### A1. Attention Is All You Need (Vaswani et al., 2017) — https://arxiv.org/abs/1706.03762
- **Extract:** self-attention is content-based routing (every token looks at every other and pulls a weighted
  mix of values); multi-head = several routing tables in parallel; the `1/√d_k` scaling keeps softmax from
  saturating; attention is **permutation-invariant**, so positional information *must* be injected.
- **Map:** [`src/model.py`](../src/model.py) → `Attention` (your `wq/wk/wv/wo`, the SDPA call) and `Block`
  (attn → residual → MLP → residual). The `1/√d_k` lives inside `F.scaled_dot_product_attention`. You already
  felt *why position magnitude matters*: the RUN 003a baseline bug was a positional signal swamping the token
  signal.
- **Do:**
  1. 3-pass read. On paper, argue why attention with no positional info can't distinguish "dog bites man"
     from "man bites dog."
  2. **Predict, then verify:** what happens if you remove the causal mask in a language model? Edit
     `Attention.forward` to `is_causal=False`, train 200 steps, watch val loss. *Prediction prompt:* will loss
     go down faster or slower, and will generation be better or worse? (Hint: think about what the model can
     now "see.") Record the result and the explanation in `log.md`, then revert.

### A2. GPT-2 — *Language Models are Unsupervised Multitask Learners* (Radford et al., 2019)
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **Extract:** drop the encoder → a **decoder-only** stack trained on pure next-token prediction is a general
  learner; learned absolute position embeddings; tied input/output embeddings.
- **Map:** your entire `TinyLM` *is* a GPT. The Stage-2 **baseline rung** uses learned PE + LayerNorm + GELU
  (RUN 003) — that *is* the GPT-2 block. Tied embeddings: `self.head.weight = self.tok.weight`.
- **Do:** Predict the embedding parameter fraction for our 5M model (you measured **20%** in RUN 002), then
  compute what it would be with a 50k vocab at d=256 (`50000*256 / total`). This is the tiny-model embedding
  trap (`research/03`, DECISIONS.md D3) — see it numerically.

### A3. RoPE — *RoFormer* (Su et al., 2021) — https://arxiv.org/abs/2104.09864
- **Extract:** rotate the query/key vectors by position-dependent angles so the dot product depends only on
  **relative** position; extrapolates to longer contexts; adds **zero** parameters.
- **Map:** [`src/model.py`](../src/model.py) → `apply_rope` / `rotate_half`. You **just ablated this** in Stage 2.
- **Do:** Before looking, **predict**: against a *fair* learned-PE baseline, is RoPE a big quality win or
  roughly neutral on short sequences? Then read the corrected ablation row in
  [`research/_assets/ablation_5m.json`](../research/_assets/ablation_5m.json) (RUN 003). **The lesson:** on
  TinyStories at seq 512, RoPE's headline benefit (length extrapolation + relative bias) is barely exercised,
  so its quality delta is small — *a SOTA component's benefit can be invisible in your regime.* That is exactly
  why you ablate instead of cargo-culting. (RoPE still earns its place: it's free and unlocks long-context
  later — `rope_theta=500000` is "insurance" you pay nothing for.)

### A4. TinyStories (Eldan & Li, 2023) — https://arxiv.org/abs/2305.07759
- **Extract:** a tiny model becomes **coherent** when the data distribution is simple enough for it to fit.
  At small scale, **data compressibility beats parameter count.** A ~1M model on TinyStories out-writes a 125M
  model on web text.
- **Map:** this is *why* RUN 002 produced real stories. The whole `RECOMMENDATION.md` rests on this single result.
- **Do:** Read §3–4. **Predict** the param count where coherence first appears, then run the **coherence ladder**
  (1M / 3M / 10M / 28M) — this is [Experiment 1 in `research/08`](../research/08-open-questions-next-experiments.md).
  It is simultaneously your next experiment *and* this lesson's centerpiece.

### A5. Chinchilla — *Training Compute-Optimal LLMs* (Hoffmann et al., 2022) — https://arxiv.org/abs/2203.15556
- **Extract:** for a fixed compute budget, the optimum is ~**20 tokens per parameter**; final loss is a
  predictable power law of params (N) and data (D).
- **Map:** DECISIONS.md **D4** — why we *don't* blindly follow 20 tok/param for a small deployed model; RUN 002
  trained at ~15 tok/param.
- **Do:** Read §3. **Predict** the slope of loss-vs-params on a log-log plot, then run the scaling-law mini-study
  ([Experiment 12 in `research/08`](../research/08-open-questions-next-experiments.md)): 2M/5M/10M/25M at 20
  tok/param. A clean straight line means your pipeline is healthy.

---

## Track B — AI Engineering (the machinery)
The craft of actually training. Here the best resources are often **code and blogs**, not papers.

### B1. The training loop — Karpathy: nanoGPT + "Zero to Hero"
- **Resource:** https://github.com/karpathy/nanoGPT · the *Let's build GPT* video (youtube/Karpathy).
- **Extract:** the canonical loop = batch → forward → loss → backward → clip → optimizer step → schedule;
  periodic eval; checkpoint the best.
- **Map:** [`src/train.py`](../src/train.py) is a nanoGPT-shaped loop (AdamW, cosine schedule, grad clip, bf16
  autocast, val eval).
- **Do:** Read nanoGPT's `model.py` side-by-side with ours and list the differences (ours: RoPE, RMSNorm,
  SwiGLU, GQA, tied embeds; nanoGPT: learned PE, LayerNorm, GELU, MHA — i.e. ours = nanoGPT + the Stage-2 ladder).

### B2. Mixed precision — PyTorch AMP docs + the bf16-vs-fp16 distinction
- **Resource:** https://pytorch.org/docs/stable/amp.html
- **Extract:** **bf16** keeps fp32's exponent range, so (unlike fp16) it needs **no loss scaling**; Tensor Cores
  do the matmuls in low precision while master weights stay fp32.
- **Map:** `torch.autocast("cuda", dtype=torch.bfloat16)` in `train.py`; RUN 000 verified `bf16_supported: True`.
- **Do:** Explain in one sentence why fp16 needs a GradScaler and bf16 doesn't. Predict the VRAM/speed effect of
  bf16 vs fp32, then reason it against your RUN 001 numbers.

### B3. FlashAttention / SDPA — Dao et al. 2022 (https://arxiv.org/abs/2205.14135) + PyTorch SDPA docs
- **Extract:** attention is **memory-bandwidth-bound**; FlashAttention fuses it and never materializes the
  `T×T` score matrix → less memory *and* faster. It's a systems win, not a FLOPs win.
- **Map:** `F.scaled_dot_product_attention` in `model.py`; you **measured** it in Experiment 0 — RUN 001:
  SDPA vs eager = **3.2× faster AND 3.2× less memory**, and eager *OOM'd* at seq 1024 where SDPA fit.
- **Do:** Re-read your RUN 001 SDPA-vs-eager numbers, then read the FlashAttention abstract + Figure 1. Write
  why the win is memory traffic, not arithmetic.

### B4. Tokenization — Karpathy "Let's build the GPT Tokenizer" + HF `tokenizers`
- **Resource:** the tokenizer video; https://github.com/huggingface/tokenizers
- **Extract:** BPE merges frequent byte pairs; for a *tiny* model the **vocab size is a parameter-budget
  decision**, because the embedding table is a big fraction of the model.
- **Map:** [`src/data.py`](../src/data.py) trains a byte-level BPE at **vocab 4096**; DECISIONS.md D3; you
  measured embedding = 20% in RUN 002.
- **Do:** The vocab-budget exercise ([Experiment 3 in `research/08`](../research/08-open-questions-next-experiments.md)):
  train tokenizers at vocab 4k/8k/16k, compute the embedding fraction of the 5M for each, and find where it
  blows the ~14% budget.

### B5. Memory & throughput engineering — `research/07` + your own probe
- **Extract:** training VRAM ≈ weights + optimizer states (AdamW = 2× params in fp32 moments) + gradients +
  activations (cut by checkpointing); ~**12 bytes/param** static, ~6 with 8-bit AdamW.
- **Map:** [`scripts/probe.py`](../scripts/probe.py); RUN 001 (the 125M-class fits 8 GB only with 8-bit AdamW +
  gradient checkpointing).
- **Do:** **Predict** the peak VRAM of the (true) 30M config before running it, then check against `probe.py`.
  Within 15%? Good — that's the mastery bar from CURRICULUM Week 2.

---

## This week's concrete plan
1. **Day 1–2:** 3-pass read **A1** (Attention) + watch **B1** (Karpathy *Let's build GPT*). Do the causal-mask
   exercise (A1.Do.2).
2. **Day 3:** Read **A3** (RoPE) and inspect your RUN 003 ablation row. Read **B3** (FlashAttention) against your
   RUN 001 numbers.
3. **Day 4:** Read **A4** (TinyStories). Run the **coherence ladder** (08 Exp 1) — the week's flagship experiment.
4. **Day 5:** Skim **A5** (Chinchilla) + **B4/B5**. Do the vocab-budget and VRAM-prediction exercises.
5. **End of week:** re-read your `log.md` predictions and score your calibration (right ~50% to start; aim for
   70–80% over time). That score, not the checkmarks, is the real progress metric.

## Where this goes next
Lesson 2 deepens the modern stack (RoPE/GQA/SwiGLU/RMSNorm internals + tokenizer scaling laws), mapped to the
true-30M build (Experiment 2, depth-vs-width). The full arc is in [`research/CURRICULUM.md`](../research/CURRICULUM.md).

---

### Lesson 1 reading list (all verified-real links)
| Item | Source |
|---|---|
| A1 Attention | https://arxiv.org/abs/1706.03762 |
| A2 GPT-2 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| A3 RoPE | https://arxiv.org/abs/2104.09864 |
| A4 TinyStories | https://arxiv.org/abs/2305.07759 |
| A5 Chinchilla | https://arxiv.org/abs/2203.15556 |
| B1 nanoGPT | https://github.com/karpathy/nanoGPT |
| B2 PyTorch AMP | https://pytorch.org/docs/stable/amp.html |
| B3 FlashAttention | https://arxiv.org/abs/2205.14135 |
| B4 HF tokenizers | https://github.com/huggingface/tokenizers |
| B5 Memory eng. | [`research/07-hardware-8gb-engineering.md`](../research/07-hardware-8gb-engineering.md) |
