# RECOMMENDATION — The North Star Decision

> ⚠️ **Partially superseded — read [`DECISIONS.md`](./DECISIONS.md) first.** The critic ([`08`](./08-open-questions-next-experiments.md) §2) found this doc conflicts with `01`/`BUILD-PLAN` on two points, now resolved in `DECISIONS.md`: **(a)** the 30M architecture below (d=512/**8 layers**) is **superseded as the default** by the deep-narrow d=256 family pending Experiment 2 — don't launch the main run until that ablation picks the winner; **(b)** the §6 success-bar *thresholds* are **provisional/`unverified`** until anchored by Experiments 1 + 9, and `bits-per-byte` (not val loss or the judge) is the primary gate. The *strategy* (TinyStories 30M first, FineWeb-Edu 125M second) stands.

> **TL;DR** Build a **~30M-parameter decoder-only transformer that generates coherent short English stories**, trained on **TinyStories (~475M tokens, ~15 tokens/param over ~3 passes → ~1.5B tokens seen)** on the RTX 3060 Ti. This is the single highest-probability path to a *visibly* "useful" model on 8 GB hardware: TinyStories empirically makes sub-50M models coherent, runs complete in **~2–4 hours** (with sub-1-hour proxy runs for iteration), and quality is gut-checkable by reading the output. The success bar: **val loss ≤ 1.5 (perplexity ≤ 4.5), GPT-judge grammar ≥ 8.0/10, and ≥ 17/20 human-read completions coherent with no token loops.** Once this pipeline is proven end-to-end, the *runner-up* — a **125M FineWeb-Edu general-text model** targeting PIQA ≥ 60% — becomes the natural "level 2" stretch goal. This doc decides the target, the size, the data, the budget, and the bar, and justifies each against the alternatives. Everything downstream (build plan, configs, eval harness) keys off this file.

---

## 1. The decision in one table

| Axis | Decision | One-line justification |
|---|---|---|
| **Primary capability** | Coherent short-story completion (TinyStories domain) | Only capability that is *visibly* useful at sub-50M scale; closed vocabulary is compressible by a tiny model. |
| **Model size** | **~30M params** (d=512, 8 layers, 8/2 GQA) | Sweet spot: coherent on TinyStories, sub-1h proxy runs, fits 8 GB with batch 32 and **no** gradient checkpointing. |
| **Architecture** | Decoder-only, RoPE θ=500k, RMSNorm pre-norm, SwiGLU, GQA 4:1, tied embeddings, SDPA/FA2 | 2026 SOTA small-LM stack; verified against SmolLM2 / Qwen3 / Llama-3.2 configs. |
| **Tokenizer** | Byte-level BPE, **vocab 8192** | Keeps embedding table to ~14% of params (4.2M / 30M); zero `<UNK>`; trainable in ~30s. |
| **Dataset** | **TinyStories** (~7.6 GB, ~475M tokens) | The one corpus *proven* to make <10M models coherent. Closed domain = compressible signal. |
| **Token budget** | ~1.5B tokens seen (~3 epochs, ~50 tok/param) | Over-train past Chinchilla (20 tok/param); narrow domain saturates fast. |
| **Optimizer** | AdamW (lr 6e-4, β=(0.9,0.95), wd 0.1 on matrices only) + bf16 + tf32 | Simplest robust default; switch to Muon as a *deliberate experiment* later, not on run 1. |
| **Schedule** | WSD (1% warmup / 89% stable / 10% decay) | Checkpoint at end of stable phase → branch multiple decay anneals cheaply. Fast-iteration friendly. |
| **Train time** | **~2–4 h** full run; **~20–45 min** proxy runs (5M model / 50M tokens) | Enables 5+ experiments per session — the ETH Zurich loop. |
| **Success bar** | val loss ≤ 1.5, GPT-judge grammar ≥ 8.0, ≥ 17/20 human-read coherent | All gut-checkable in < 15 min eval, < $0.01/run. |

---

## 2. Why "coherent short stories" is the right PRIMARY target

The project deferred the capability choice to research. There are four serious candidates. The decision is **story completion (TinyStories)**, and here is the reasoning, with the runners-up ranked.

### 2.1 The deciding principle
At sub-50M scale, **data distribution compressibility dominates everything** (`research/06`, `research/04`). A tiny model can only learn a distribution it can actually fit. The single most important empirical fact in the whole research corpus:

- A **1M-param** model trained on TinyStories generates *more coherent* text than a **125M** GPT-2 trained on heterogeneous web data ([TinyStories, Eldan & Li 2023](https://arxiv.org/abs/2305.07759)).
- A **28M** TinyStories model reaches **val loss ~1.3 / perplexity ~3.7** — a regime where output is genuinely readable.

That means TinyStories is the *only* candidate where the 8 GB / 30M constraint produces something a human will look at and say "this works." Every other target either needs more params than fit comfortably, or produces output that only a benchmark can distinguish from noise.

### 2.2 Candidate comparison

| Rank | Capability | Min params to be "useful" | Visible quality at 30M? | Iteration speed | Verdict |
|---|---|---|---|---|---|
| **1 (PICK)** | **Short-story completion** | ~5–30M | **Yes — readable prose** | Sub-1h runs | **Primary.** Compressible domain; visible win; fastest loop. |
| 2 | General web-text (FineWeb-Edu) | ~100–135M | Marginal — PIQA ~60% is the only signal | 10–16h Chinchilla | **Runner-up / level 2.** Best "real LM" feel, but needs 125M and benchmarks to judge. |
| 3 | Python code completion | ~50M (phi-style) | Partial — short snippets | ~hours | **Stretch.** "TinyStories for code" is plausible but data curation is harder; defer. |
| 4 | Simple Q&A / instruct | ~50–135M | Needs SFT stage | 2-stage pipeline | **Later.** Requires a solid base model first; not a starting target. |

### 2.3 Why not start at the runner-up (125M FineWeb-Edu)?
It is the more *impressive* artifact (a small general LM), but it is the wrong *first* target:
- 125M needs gradient checkpointing + careful VRAM management to fit 8 GB (`research/07`), adding friction before the pipeline is proven.
- A Chinchilla-optimal run is 10–16 h; an over-trained one is ~64 h — too slow for the "train many models" learning loop on run 1.
- Its quality is only legible through benchmarks (PIQA/ARC), not by reading — slower, less motivating feedback for a learner.

**Strategy: prove the entire pipeline (tokenizer → data → train loop → checkpoint → eval → generate) on the 30M TinyStories model first, then graduate to the 125M FineWeb-Edu model as level 2.** TinyStories de-risks every piece of infrastructure cheaply.

---

## 3. The recommended MODEL (concrete spec)

A **~30M-parameter** decoder-only transformer. This is the consensus "fast-iteration vehicle" across `research/00`, `01`, `02`, and `07`.

```yaml
# configs/30m_tinystories.yaml
name: 30m_tinystories
# --- architecture ---
d_model: 512
n_layers: 8            # depth-biased but modest; TinyStories doesn't need 30 layers
n_heads: 8             # Q heads
n_kv_heads: 2          # GQA 4:1
head_dim: 64
d_ffn: 1376            # SwiGLU: round(2/3 * 4 * 512) -> nearest 64
activation: swiglu
norm: rmsnorm          # pre-norm
pos: rope
rope_theta: 500000     # future-proofs context extension at zero cost
qk_norm: false         # add only if loss spikes at higher LR
tie_embeddings: true   # halves embedding cost
vocab_size: 8192       # byte-level BPE
max_seq_len: 1024      # TinyStories are short; 512-1024 is plenty
attn_impl: sdpa        # PyTorch scaled_dot_product_attention -> FlashAttention-2 on Ampere

# --- param accounting (approx) ---
# embedding (tied):  8192 * 512            = 4.2M   (~14%)
# 8 transformer blocks (attn+swiglu+norms) ~ 25-26M
# TOTAL                                    ~ 29-30M
```

**Param budget check (the #1 tiny-model trap):** at vocab 8192 / d 512, embeddings are 4.2M (~14%) — comfortably under the 15% ceiling (`research/03`). A naive 50k vocab here would burn ~86% of params on the embedding table and leave nothing for layers. **Do not raise vocab without re-running the budget math.**

**VRAM check (8 GB):** 30M params in bf16 + AdamW = ~480 MB weights+opt state; at batch 32 / seq 1024 the run sits well under 4 GB with **no gradient checkpointing** (`research/01`, `07`). Gradient checkpointing and 8-bit AdamW are reserved for the 125M+ level-2 model.

**Architecture provenance:** every component is taken from verified 2025–2026 configs — SmolLM2-135M, Qwen3, Llama-3.2 (`research/01`). RoPE θ=500k, pre-norm RMSNorm, SwiGLU at 2/3·4·d, GQA, tied embeddings, and SDPA-backed FlashAttention-2 are the converged small-LM stack. FlashAttention-3 is **not** usable (Hopper-only); the Ampere GA104 gets FA2 automatically through `F.scaled_dot_product_attention`.

---

## 4. The DATASET + TOKEN BUDGET

**Dataset: TinyStories** (~7.6 GB on disk, ~475M tokens of GPT-3.5/4-generated simple English stories), with **TinyStoriesInstruct** held in reserve for a later SFT experiment at zero extra download (`research/04`).

**Tokenizer:** train a **byte-level BPE, vocab 8192**, on the TinyStories corpus itself (HuggingFace `tokenizers`, ~30s on CPU). A domain-trained tokenizer gives ~20–30% better fertility than reusing GPT-2's (`research/03`). Verify encode→decode round-trip and that `id 0 = <|endoftext|>` before spending GPU time.

**Token budget:** TinyStories has ~475M unique tokens. Train for **~3 epochs → ~1.5B tokens seen (~50 tokens/param)**. This deliberately over-trains past Chinchilla's 20 tok/param (`research/02`), which is correct for a deployed/over-trained small model on a *narrow* domain — but the closed vocabulary saturates quickly, so there is no need for the 1,000×+ ratios used on web data. Watch the val-loss curve: stop the stable phase when it flattens, then run the WSD decay.

**Why not FineWeb-Edu for run 1?** It's the right corpus for the 125M level-2 model (10B-token `sample-10BT`, ~27 GB), but its open-domain distribution is *not* compressible at 30M — the model would be incoherent. Keep it for level 2 (`research/04`).

---

## 5. EXPECTED TRAINING TIME

Estimates from `research/07` throughput modeling (~108–180K tok/s for a 30M model on the 3060 Ti at 25–45% MFU; calibrate empirically on the first run):

| Run type | Model | Tokens | Wall time | Purpose |
|---|---|---|---|---|
| **Proxy / ablation** | 5M | ~50M | **~20–45 min** | Architecture & HP sweeps; 5+ per session. |
| **Main run** | 30M | ~1.5B (3 epochs) | **~2–4 h** | The deliverable TinyStories model. |
| Level-2 (later) | 125M FineWeb-Edu | ~2.5B (Chinchilla-ish) | ~10–16 h | Stretch: a "real" small LM. |

This keeps the project firmly inside the ETH Zurich loop: many cheap proxy runs, occasional multi-hour commitments, calibrated predictions logged before each run (`research/00`).

---

## 6. The SUCCESS BAR — how we know it's "useful"

A four-tier eval, all runnable in < 15 min for < $0.01/run (`research/06`). The 30M TinyStories model is **"minimally useful" when it clears at least 3 of 4 groups**, and the target is to clear all four:

| # | Metric | Bar | Tool |
|---|---|---|---|
| 1 | **Validation loss** | **≤ 1.5** (perplexity ≤ ~4.5) | Train loop / W&B. Reference: 28M TinyStories model hits ~1.3. |
| 2 | **GPT-judge grammar** | **≥ 8.0 / 10** (consistency ≥ 7.0, creativity ≥ 5.0) | GPT-4-mini judge on 20 fixed story-prompt probes. |
| 3 | **Human reading pass** | **≥ 17 / 20** completions coherent, **0** token loops / severe drift | 5-min manual read. |
| 4 | **Bits-per-byte** | **≤ 1.0** on held-out TinyStories | Archival, tokenizer-agnostic metric. |

Note: standard reasoning benchmarks (HellaSwag, MMLU, ARC-challenge) are **deliberately skipped** at 30M — they sit at chance and only mislead (`research/06`). They re-enter at the 125M level-2 model, where the bar becomes **PIQA ≥ 60%** (SmolLM2-135M reference: **68.4% PIQA, 42.1% HellaSwag** per [SmolLM2, COLM 2025](https://arxiv.org/abs/2502.02737)).

**Write `evaluate.py` first** — a standalone script that takes a checkpoint and emits all four tiers as JSON. Frictionless eval is the precondition for the fast-iteration method; build it before the first real training run.

---

## 7. What this commits us to (build-plan north star)

1. **Install stack:** uv → PyTorch 2.x cu124 → SDPA (no separate flash-attn needed) → HF tokenizers → W&B. (Note: research suggests Python 3.11 for ecosystem stability; the env is 3.13 — verify flash-attn/bitsandbytes wheels, but SDPA path avoids the flash-attn dependency entirely.)
2. **Tokenizer:** train byte-level BPE vocab 8192 on TinyStories; verify round-trip.
3. **Model:** implement the 30M spec above (single `model.py`, swappable config).
4. **Data:** tokenize TinyStories to binary shards once (datatrove or a simple script).
5. **Eval:** write `evaluate.py` (4 tiers → JSON) **before** the first full run.
6. **Proxy loop:** validate the pipeline on a 5M / 50M-token run (< 45 min); confirm loss falls and generation is non-degenerate.
7. **Main run:** 30M, ~1.5B tokens, WSD schedule, AdamW, bf16; checkpoint at end of stable phase, branch the decay anneal.
8. **Gate:** evaluate against Section 6 bar. If cleared → ship the artifact and graduate to the 125M FineWeb-Edu level-2 target.

**Sources:** [TinyStories (Eldan & Li, 2023)](https://arxiv.org/abs/2305.07759) · [SmolLM2 (COLM 2025)](https://arxiv.org/abs/2502.02737) · internal research docs `00`–`07` in `research/`.
