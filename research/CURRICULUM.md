# CURRICULUM.md — From RTX 3060 Ti to World-Class Model Researcher

**TL;DR:** This is a 10-week, hands-on curriculum that turns the `small-tank` project into a deliberate-practice path toward research mastery. It is built on one thesis — proven by the ETH Zurich method and the modded-nanoGPT speedrun — that **research taste is a trainable muscle, and you train it by running many small experiments fast, predicting outcomes before you look, and writing down what surprised you.** Each week pairs canonical papers (Attention, GPT-2/3, Chinchilla, RoPE, GQA, FlashAttention, TinyStories, phi, SmolLM, Muon, μP) with concrete ablations runnable in minutes-to-hours on your 8 GB GPU, an explicit "mastery bar," and a meta-skill (reading papers, ablation discipline, lab notebook, sharing). Everything ties back to the eight research docs (`00`–`07`) already in this folder. The goal is not to train one good model — it is to become the kind of person who can train any model and know *why* it works.

---

## How to use this document

- **It is a curriculum, not a checklist.** Each week has: *Read* (papers), *Build/Run* (experiments tied to "train many models fast"), *Mastery bar* (how you know you've internalized it), and *Meta-skill* (the research habit you're growing that week).
- **The spine is the fast-iteration loop** from `00-eth-zurich-method.md`: hypothesis → prediction → change one thing → run a short proxy → verdict → update your intuition. Your standard proxy is a **5M-param model on a 50M-token budget (~5–8 min/run)**, giving 7–12 hypothesis tests per hour.
- **Before every run, write your prediction** — not what you hope, but what you *expect* and why. Track your calibration. Beginners are right ~50% of the time; experts hit ~70–80% (`00`).
- **Keep a flat `research/log.md`** with one entry per run: `hypothesis · change · prediction · result · verdict · intuition-update`. This is the single highest-leverage habit in the whole curriculum.
- **One git branch per experiment.** Merge only validated wins. This keeps your research history auditable and prevents confounded results (`00`).
- **Mastery is measured by predictions, not by green checkmarks.** When you can predict the *shape* of a loss curve or an ablation delta before running it, you have the skill.

### The canonical reading list (bookmark these)

| Topic | Paper | URL |
|---|---|---|
| Transformer | Attention Is All You Need (Vaswani 2017) | https://arxiv.org/abs/1706.03762 |
| Scaling / few-shot | GPT-3 (Brown 2020) | https://arxiv.org/abs/2005.14165 |
| GPT-2 (report) | Language Models are Unsupervised Multitask Learners | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| Compute-optimal | Chinchilla (Hoffmann 2022) | https://arxiv.org/abs/2203.15556 |
| Original scaling laws | Kaplan 2020 | https://arxiv.org/abs/2001.08361 |
| Positional | RoPE / RoFormer (Su 2021) | https://arxiv.org/abs/2104.09864 |
| Attention efficiency | GQA (Ainslie 2023) | https://arxiv.org/abs/2305.13245 |
| Attention efficiency | Multi-Query Attention (Shazeer 2019) | https://arxiv.org/abs/1911.02150 |
| Kernel | FlashAttention (Dao 2022) | https://arxiv.org/abs/2205.14135 |
| Kernel | FlashAttention-2 (Dao 2023) | https://arxiv.org/abs/2307.08691 |
| Norm | RMSNorm (Zhang & Sennrich 2019) | https://arxiv.org/abs/1910.07467 |
| Activation | GLU Variants / SwiGLU (Shazeer 2020) | https://arxiv.org/abs/2002.05202 |
| Data quality | TinyStories (Eldan & Li 2023) | https://arxiv.org/abs/2305.07759 |
| Data quality | Textbooks Are All You Need / phi-1 (Gunasekar 2023) | https://arxiv.org/abs/2306.11644 |
| Data quality | phi-1.5 (Li 2023) | https://arxiv.org/abs/2309.05463 |
| Web data | FineWeb / FineWeb-Edu (Penedo 2024) | https://arxiv.org/abs/2406.17557 |
| Tiny models | SmolLM2 (Allal 2025) | https://arxiv.org/abs/2502.02737 |
| Depth>width tiny | MobileLLM (Liu 2024) | https://arxiv.org/abs/2402.14905 |
| Optimizer | Muon / Moonshot scaling (2025) | https://arxiv.org/abs/2502.16982 |
| Optimizer (orig) | Muon writeup (Keller Jordan) | https://kellerjordan.github.io/posts/muon/ |
| HP transfer | μP / Tensor Programs V (Yang 2022) | https://arxiv.org/abs/2203.03466 |
| Vocab scaling | Scaling Laws with Vocabulary (Tao 2024) | https://arxiv.org/abs/2407.13623 |
| Tokenizer-free | Byte Latent Transformer (2024) | https://arxiv.org/abs/2412.09871 |
| LR schedule | MiniCPM / WSD (Hu 2024) | https://arxiv.org/abs/2404.06395 |
| Speedrun | modded-nanoGPT (Keller Jordan repo) | https://github.com/KellerJordan/modded-nanogpt |
| RL post-train | DPO (Rafailov 2023) | https://arxiv.org/abs/2305.18290 |
| RL post-train | GRPO / DeepSeekMath (Shao 2024) | https://arxiv.org/abs/2402.03300 |

*(All URLs verified as real arXiv/blog/repo locations. If a link 404s, search the title — do not trust a guessed ID.)*

---

## The four phases at a glance

| Phase | Weeks | Theme | Outcome |
|---|---|---|---|
| **I. Foundations** | 1–2 | The transformer, the loop, the hardware | A working train loop; you can read a config and predict its VRAM |
| **II. The modern stack** | 3–5 | RoPE, GQA, RMSNorm/SwiGLU, FlashAttn, tokenizer | You can build a 2026-SOTA tiny model from scratch and justify every block |
| **III. Training science** | 6–8 | Scaling laws, schedules, optimizers, data quality | You run ablations that *predict* large-model behavior from small proxies |
| **IV. Frontier + post-train** | 9–10 | μP, Muon mastery, SFT/DPO/GRPO, your capstone | A "minimally useful" model you trained and a public writeup |

---

## PHASE I — Foundations (Weeks 1–2)

### Week 1 — The transformer, end to end, and the iteration loop

**Read.** *Attention Is All You Need* (https://arxiv.org/abs/1706.03762) — slowly, with paper and pen. Then the GPT-2 report (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) for the decoder-only simplification. Skim Kaplan scaling laws (https://arxiv.org/abs/2001.08361) for the *idea* that loss is a smooth power law of compute.

**Build / Run.** Set up the environment (`05-tooling-codebases.md`): `uv` → PyTorch + cu124 → SDPA-based attention. Get a tiny char-level model training on TinyStories (`04-datasets.md`, Experiment 1). Reproduce the coherence ladder: train **1M / 3M / 10M / 28M** models and watch where text becomes grammatical. This is your first "train many models fast" session and it costs under 1 GPU-hour.

- **Prediction to write first:** At what param count does TinyStories output become coherent? (The paper says sub-10M — predict before you read the number.)

**Mastery bar.** You can draw the decoder block from memory (attention → residual → norm → MLP → residual), explain why it's causal-masked, and you have a generation loop that produces text. You logged 4 runs in `research/log.md`.

**Meta-skill — Reading a paper in three passes.** Pass 1: title/abstract/figures/conclusion (5 min, decide if relevant). Pass 2: intro + method, skip proofs (30 min, get the mechanism). Pass 3: reproduce one equation or result by hand. Do all three on *Attention Is All You Need* this week.

### Week 2 — Hardware intuition and the VRAM budget

**Read.** `07-hardware-8gb-engineering.md` cover to cover. FlashAttention paper intro (https://arxiv.org/abs/2205.14135) for *why* attention is memory-bandwidth-bound.

**Build / Run.** Do `07`'s Experiment D (VRAM Budget Profiling) and `05`'s Experiment 3. Build the table: model size × seq len × batch → peak VRAM (via `torch.cuda.max_memory_allocated()`, **not** nvidia-smi) and tokens/sec. Confirm the **12 bytes/param** static rule (BF16 weights+grads + FP32 AdamW) and watch it drop to ~6 bytes/param with 8-bit AdamW + gradient checkpointing.

- **Prediction to write first:** What's the largest model you can train at S=1024, B=8, no checkpointing? Predict, then measure.

**Mastery bar.** Given any config (d_model, layers, vocab, seq, batch, optimizer), you can compute peak VRAM within ~15% *before* running, and you know which knob (8-bit Adam, grad checkpointing, batch, seq) to turn when you OOM. You have answered the open questions in `07` with **real numbers from your card** (throughput for 30M/125M, actual MFU).

**Meta-skill — The lab notebook.** Commit to `research/log.md`. Adopt the template from `00`. Every run, every session, no exceptions. The notebook is the compounding asset.

---

## PHASE II — The modern stack (Weeks 3–5)

### Week 3 — Positional encoding and normalization

**Read.** RoPE (https://arxiv.org/abs/2104.09864), RMSNorm (https://arxiv.org/abs/1910.07467). Then `01-architecture-sota.md` §RoPE/§norm.

**Build / Run.** Three ablations on the 5M proxy (one change each, short runs):
1. Learned absolute pos-emb vs sinusoidal vs **RoPE** (theta=10k vs 500k).
2. LayerNorm vs **RMSNorm** (watch tokens/sec *and* loss).
3. Post-norm vs **pre-norm** (pre-norm should train more stably at higher LR).

- **Prediction:** RoPE at theta=500k costs nothing now but buys later context-extension. Pre-norm lets you raise LR. Write your expected loss deltas first.

**Mastery bar.** You can implement RoPE from the rotation-matrix equations without copying, explain why theta=500k is "free insurance" for later YaRN extension (`01`), and you've *seen* pre-norm stabilize a high-LR run that post-norm couldn't.

**Meta-skill — Change one thing.** This week enforces ablation hygiene: never change two variables in one run. A confounded experiment teaches nothing.

### Week 4 — Attention efficiency and the MLP

**Read.** GQA (https://arxiv.org/abs/2305.13245) and the original MQA (https://arxiv.org/abs/1911.02150). SwiGLU (https://arxiv.org/abs/2002.05202). FlashAttention-2 (https://arxiv.org/abs/2307.08691). Then `01` §GQA/§SwiGLU.

**Build / Run.**
1. MHA vs **GQA** (Q:KV = 4:1) vs MQA at the proxy scale — measure KV-cache size, throughput, and loss. At tiny scale GQA's quality cost should be near-zero while KV memory drops.
2. 4×d MLP vs **SwiGLU at int(2/3·4·d)** — confirm SwiGLU matches quality at ~same param count (`02-training-recipes.md`).
3. Swap naive attention for `F.scaled_dot_product_attention(is_causal=True)` and measure the speedup (FlashAttention-2 dispatch on Ampere — no separate install, `07`).

- **Prediction:** SDPA gives you 20–30% throughput for one line of code. Write the number you expect.

**Mastery bar.** You can explain GQA as "share K/V heads across query-head groups to shrink the KV cache" and quantify the trade-off. You default to SDPA everywhere. You've reproduced `01`'s claim that depth beats width (run MobileLLM's ablation: 12 vs 24 layers at fixed 30M params — read https://arxiv.org/abs/2402.14905).

**Meta-skill — Throughput as a first-class metric.** Log tokens/sec and MFU on every run, not just loss. A 2% quality gain that halves throughput is usually a loss in the fast-iteration regime.

### Week 5 — Tokenization and the embedding budget

**Read.** `03-tokenization.md` fully. Scaling Laws with Vocabulary (https://arxiv.org/abs/2407.13623). Skim BLT (https://arxiv.org/abs/2412.09871) for the frontier, but don't implement it.

**Build / Run.** Run `03`'s `vocab_budget.py` before touching architecture. Train byte-level BPE tokenizers at **vocab = 4k / 8k / 16k / 32k** on your corpus (HF Tokenizers, ~30s each on CPU). For a fixed 5M-param budget, train a model with each and compare val loss *and* embedding-param fraction. Verify tokenizer round-trip before any GPU hours.

- **Prediction:** For a 5M model, a 32k vocab will eat most of your params and *hurt* loss vs 8k. Write the crossover you expect.

**Mastery bar.** You internalize that for tiny models, **the embedding table is a parameter-allocation decision, not a free choice** (`03`: a naive 50k vocab burns 86% of a 30M model). You can pick vocab from model size on sight and you always tie embeddings under 500M params.

**Meta-skill — Sanity checks before compute.** Round-trip the tokenizer, eyeball a data sample, confirm special-token IDs. Cheap pre-flight checks save GPU-hours.

---

## PHASE III — Training science (Weeks 6–8)

### Week 6 — Scaling laws you can feel

**Read.** Chinchilla (https://arxiv.org/abs/2203.15556) in full — this is the single most important training paper. Re-read `00` §scaling and `04` §tokens-per-param.

**Build / Run.** The **scaling-law mini-study** (`00` Experiment): train **2M / 5M / 10M / 25M** models at Chinchilla-optimal (~20 tok/param), plot loss vs params on log-log. A clean straight line means your pipeline is healthy and proxies are trustworthy.

- **Prediction:** Write the slope you expect and whether your 25M point will fall on the line extrapolated from 2M/5M/10M.

**Mastery bar.** You can sketch the compute-optimal frontier from memory and explain *why* Chinchilla's 20 tok/param is **wrong for your goal** — it's optimal for one-shot compute, not for a deployed model you'll over-train (`02`, `04`). A broken (non-linear) plot makes you suspect a data or LR bug *before* you trust any result.

**Meta-skill — Calibration review.** Two weeks of predictions are in your log. Tally your hit rate. Where were you systematically wrong (too optimistic about a trick? always under-predicting throughput?)? Naming your biases is how taste forms (`00`, `itsreallyvivek`).

### Week 7 — Schedules, over-training, and reusable checkpoints

**Read.** MiniCPM/WSD (https://arxiv.org/abs/2404.06395). `02-training-recipes.md` §WSD and §over-train.

**Build / Run.**
1. **WSD schedule:** train through warmup+stable, checkpoint, then branch **two decay anneals** (e.g., to different final LRs / data mixes) from that one checkpoint — "two models for 1.1× cost" (`02`). Internalize why this is the iteration superpower.
2. **Over-training pays off** (`02` Experiment 3): three 5M models at **20 / 100 / 300 tok/param**. The 300× model should be visibly better despite being "past optimal."

- **Prediction:** Plot expected val loss at 20/100/300 tok/param before running. Where do *you* think diminishing returns kick in?

**Mastery bar.** You default to WSD for every run, you reuse stable-phase checkpoints reflexively, and you can argue from data why SmolLM2-135M used ~14,800 tok/param (`02`, `04`). You set token budgets by *intent* (exploration vs deployable artifact), not by Chinchilla reflex.

**Meta-skill — Branch-per-experiment, merge-validated-wins.** Your WSD decay-branches map naturally onto git branches. Practice keeping a clean, auditable experiment tree.

### Week 8 — Optimizers and data quality (the two biggest levers)

**Read.** Muon writeup (https://kellerjordan.github.io/posts/muon/) + the scaling paper (https://arxiv.org/abs/2502.16982). TinyStories (https://arxiv.org/abs/2305.07759), phi-1 "Textbooks Are All You Need" (https://arxiv.org/abs/2306.11644), and SmolLM2 (https://arxiv.org/abs/2502.02737). `02` §optimizers, `04` §data-quality, `06-evaluation.md`.

**Build / Run.**
1. **AdamW vs Muon** at 5M and 25M (Muon on 2D weight matrices, AdamW on embeddings/norms). Does the ~2×-efficiency claim hold at *your* small scale, or is it a multi-GPU/large-scale effect? (`05`, `02` open questions — answer it empirically.)
2. **Data-quality ablation** — the most important experiment in the project: train the *same* 25M architecture on (a) raw web text vs (b) **FineWeb-Edu** vs (c) a TinyStories+Cosmopedia mix. Evaluate with `06`'s four-tier pipeline (loss/BPB, PIQA+ARC-easy, GPT-judge, human read).

- **Prediction:** Per `06`, data curation moved a 135M model +15 pts on HellaSwag with *zero* architecture change. Predict your PIQA delta from data alone, and predict whether Muon beats AdamW at 5M.

**Mastery bar.** You've *felt* that **data quality dominates architecture at tiny scale** (`04`, `06`) — and you can quantify it. You can configure Muon's weight-matrix/rest split correctly and decide when it's worth the config complexity. You run `06`'s eval automatically at the end of every job.

**Meta-skill — Cheap-to-specify verification.** You don't re-derive a model's quality by hand; you author a *contract* (the `06` four-tier scorecard, a fixed probe set) once and read pass/fail. Verification cost should scale with the spec, not the solution (the project's "iron rule").

---

## PHASE IV — Frontier techniques + post-training + capstone (Weeks 9–10)

### Week 9 — μP, Muon mastery, and reading the speedrun

**Read.** μP / Tensor Programs V (https://arxiv.org/abs/2203.03466) — hard but worth it. Read the modded-nanoGPT repo (https://github.com/KellerJordan/modded-nanogpt) and `00` §speedrun: identify which of its tricks (Muon, RoPE, QK-norm, ReLU², FP8) are single-GPU-portable vs 8×H100-specific.

**Build / Run.**
1. **μP transfer:** tune LR on a d=256 proxy, then *transfer* it to a wider d=512 model without re-tuning (`01`). Verify the optimal LR is stable across widths — this is the payoff that lets you tune cheap and scale confident.
2. **QK-norm ablation:** add QK-norm and push LR 1.5× higher; confirm it suppresses loss spikes (`01`). Note: FP8 is *not* available on your Ampere card (`02`) — write down why, so you never waste time chasing it.

- **Prediction:** Will the d=256-optimal LR be within 2× of the d=512-optimal LR under μP? Predict, then test.

**Mastery bar.** You can explain μP as "parameterize so the optimal LR is width-invariant," and you know exactly which speedrun techniques transfer to a single 8 GB Ampere card and which don't. You can read a frontier training repo and separate the portable ideas from the hardware-specific hacks — a senior-researcher skill.

### Week 10 — Post-training and the capstone

**Read.** DPO (https://arxiv.org/abs/2305.18290), GRPO/DeepSeekMath (https://arxiv.org/abs/2402.03300). `05` §TRL, `06` §minimally-useful bar.

**Build / Run — the capstone.** Pick ONE narrow, evaluable target (story completion, simple Python, or simple Q&A — `06`'s open question demands you choose the domain). Then run the full pipeline you've now learned end to end:
1. Custom BPE tokenizer sized to your model (`03`).
2. Pretrain a **30–125M** model with the 2026 stack: RoPE θ=500k, RMSNorm pre-norm, GQA, SwiGLU, SDPA/FlashAttn-2, WSD schedule, 8-bit Adam (and Muon if Week 8 justified it), over-trained to 100–300 tok/param (`01`,`02`,`07`).
3. **SFT** with TRL `SFTTrainer` on a small instruction set; optionally **DPO/GRPO** for preference/RL (`05`).
4. Evaluate against `06`'s "minimally useful" bar: **val perplexity < 12, PIQA > 58% (or ARC-easy > 35%), GPT-judge grammar > 7.0/10, <3/20 completions degenerate.**

- **Prediction:** Write your expected scorecard *before* the final eval. The gap between prediction and result is your taste, measured.

**Mastery bar.** You shipped a model that clears 3 of 4 "minimally useful" groups, *and* you can explain every architecture and training choice with a citation and an ablation you personally ran. You over-trained on purpose, not by accident.

**Meta-skill — Sharing results.** Write a public-quality report: the config, the loss curves, the ablation table, what surprised you, what you'd do next. Per `00`/`itsreallyvivek`, research taste compounds when it's externalized and critiqued. A clean writeup + reproducible repo *is* the artifact that marks a researcher.

---

## After Week 10 — Becoming world-class (the ongoing loop)

The curriculum ends; the practice doesn't. The researchers you admire (Karpathy, Keller Jordan, the PufferLib/SmolLM teams) all run the same loop forever, just on harder problems:

1. **Pick an open question** — start with the dozens already listed across `00`–`07` (e.g., "does Muon help below 100M?", "TinyStories-style closed-domain advantage for code?", "optimal data mixture below 100M?"). Each is a real, unanswered question you are now equipped to attack.
2. **Run the smallest experiment that could answer it.** Proxy first, always.
3. **Predict, run, journal, share.** Forever.
4. **Raise the bar:** reproduce a SOTA result from scratch; then beat it on one axis; then write something the field didn't already know.

**What world-class looks like, concretely:** you can take a new paper, predict whether its result will replicate at small scale, design the one ablation that would falsify it, run it before lunch, and be right ~75% of the time — and when you're wrong, you learn faster than anyone because your notebook tells you exactly which prior was off. That is not talent. It is this loop, run hundreds of times. **The more models you train, the more you learn.**

---

### Quick map: curriculum week → research doc

| Week | Primary docs |
|---|---|
| 1 | `00`, `04`, `05` |
| 2 | `07`, `05` |
| 3 | `01` |
| 4 | `01`, `02`, `07` |
| 5 | `03` |
| 6 | `00`, `02`, `04` |
| 7 | `02` |
| 8 | `02`, `04`, `05`, `06` |
| 9 | `01`, `00` |
| 10 | `01`–`07` (full pipeline), `06` (bar) |
