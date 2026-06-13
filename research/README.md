# Small-Tank Research — Source of Truth

> **Mission.** Build a *minimally useful* sub-0.5B-parameter language model **from scratch** on a single **RTX 3060 Ti (8 GB VRAM)**, using **current (2024–2026) SOTA architecture and training techniques**, by following the **ETH-Zurich / PufferLib fast-iteration method**: keep runs short (minutes, not days), and *train many models* — because the more models you train, the more you learn. This folder serves a **threefold objective**: **(1) AI research** — learn the bases of training models and research taste; **(2) AI engineering** — learn the nuts and bolts of actually training models; **(3) an auto-improving model** — end with a model that continuously self-improves by ingesting new SOTA from arXiv, Hugging Face, and X / x.ai (see [`09-autoimprovement-loop.md`](./09-autoimprovement-loop.md)). It is both an authoritative technical reference *and* a personal training ground / curriculum for becoming a world-class model researcher. Every doc explains the *why*, the trade-offs, and includes hands-on experiments.

---

## ⭐ Headline recommendation

**Build a ~30M-parameter decoder-only transformer that generates coherent short English stories, trained on TinyStories (~475M tokens, ~3 epochs → ~1.5B tokens seen, ~50 tok/param) on the RTX 3060 Ti.**

This is the single highest-probability path to a *visibly* useful model on 8 GB hardware: TinyStories empirically makes sub-50M models coherent, a full run completes in **~2–4 hours** (with sub-1-hour proxy runs for iteration), and quality is gut-checkable *by reading the output*.

**Success bar:** val loss ≤ 1.5 (perplexity ≤ ~4.5) · GPT-judge grammar ≥ 8.0/10 · ≥ 17/20 human-read completions coherent with no token loops · bits-per-byte ≤ 1.0.

**Level 2 (stretch):** a **125M FineWeb-Edu** general-text model targeting **PIQA ≥ 60%**, once the entire pipeline (tokenizer → data → train loop → checkpoint → eval → generate) is proven end-to-end on the 30M model.

Full reasoning, the concrete config, and the alternatives considered are in **[RECOMMENDATION.md](./RECOMMENDATION.md)**.

---

## 🧰 Hardware constraints (hard limits)

```
GPU         NVIDIA RTX 3060 Ti — Ampere GA104, sm_86, 8 GB VRAM, ~448 GB/s
            bf16 + TF32 Tensor Cores ✓ · FlashAttention-2 (via SDPA) ✓ · FlashAttention-3 ✗ (Hopper-only) · FP8 ✗ (Ada/Hopper-only)
CPU         Intel i5-12400F — 12 threads
RAM         30 GB
Disk        ~660 GB free
Software    Python 3.13, Linux, torch not yet installed
            (research suggests Python 3.11 for ML-ecosystem wheel stability; the SDPA path avoids the flash-attn dependency entirely)
```

Practical ceiling: **~350–500M params** trains comfortably with gradient checkpointing + 8-bit AdamW; **~30M** is the fast-iteration vehicle (sub-1h Chinchilla runs).

---

## 📚 Table of contents

| # | Doc | One-line description |
|---|---|---|
| 00 | [00-eth-zurich-method.md](./00-eth-zurich-method.md) | The fast-iteration philosophy: train many small models on one GPU; calibration journaling; ablation discipline. |
| 01 | [01-architecture-sota.md](./01-architecture-sota.md) | The 2026 SOTA small-LM stack: decoder-only, RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, QK-norm, muP. |
| 02 | [02-training-recipes.md](./02-training-recipes.md) | Optimizers (AdamW/Muon/Adam-mini), WSD schedule, over-training past Chinchilla, bf16, hyperparameter tables. |
| 03 | [03-tokenization.md](./03-tokenization.md) | Tokenization as a param-budget problem; byte-level BPE at vocab 8k–32k; weight tying; special tokens. |
| 04 | [04-datasets.md](./04-datasets.md) | The data ladder: TinyStories → FineWeb-Edu → Cosmopedia → SmolTalk/code; quality > quantity at small scale. |
| 05 | [05-tooling-codebases.md](./05-tooling-codebases.md) | The codebases & tools: nanoGPT/nanochat, modded-nanoGPT/Muon, TRL, datatrove, litGPT, llm.c, uv, install order. |
| 06 | [06-evaluation.md](./06-evaluation.md) | Four-tier eval: loss/BPB → PIQA+ARC-easy → GPT-judge → human read; what to skip at <150M; the "useful" bar. |
| 07 | [07-hardware-8gb-engineering.md](./07-hardware-8gb-engineering.md) | The 8 GB VRAM budget equation, throughput/MFU estimates, time-to-train, what fits at which seq length. |
| 08 | [08-open-questions-next-experiments.md](./08-open-questions-next-experiments.md) | The skeptic's pass: gaps, the 4 contradictions, and the prioritized 12-experiment backlog (start with Exp 0). |
| 09 | [09-autoimprovement-loop.md](./09-autoimprovement-loop.md) | **Objective 3**: the standing loop that ingests SOTA from arXiv + Hugging Face + x.ai and turns it into experiments. |
| ‼️ | [DECISIONS.md](./DECISIONS.md) | **Authoritative.** Resolves the 4 contradictions into one provisional decision each — wins over any conflicting doc. |
| ★ | [RECOMMENDATION.md](./RECOMMENDATION.md) | **The north-star decision**: target, size, data, budget, and success bar (partially superseded by DECISIONS.md). |
| → | [BUILD-PLAN.md](./BUILD-PLAN.md) | The concrete step-by-step plan to go from empty repo to a trained, evaluated 30M model. |
| 🎓 | [CURRICULUM.md](./CURRICULUM.md) | The learning path: what to read/build in what order to internalize the research and grow research taste. |
| 🔗 | [sources.md](./sources.md) | Aggregated, de-duped list of every external source cited across the docs. |

---

## 🚀 Start here (reading order)

1. **[RECOMMENDATION.md](./RECOMMENDATION.md)** → then **[DECISIONS.md](./DECISIONS.md)** — what we're building and why, then the authoritative resolutions that override any conflict. Read these two first.
2. **[00-eth-zurich-method.md](./00-eth-zurich-method.md)** — the *method* you'll work by. Internalize the fast-iteration loop before touching code.
3. **[BUILD-PLAN.md](./BUILD-PLAN.md)** — the ordered steps from empty repo to trained model.
4. **[07-hardware-8gb-engineering.md](./07-hardware-8gb-engineering.md)** — know exactly what fits in 8 GB before you write the model.
5. **[03](./03-tokenization.md) → [01](./01-architecture-sota.md) → [02](./02-training-recipes.md) → [04](./04-datasets.md) → [06](./06-evaluation.md)** — tokenizer, model, training, data, eval, in the order you'll build them.
6. **[05-tooling-codebases.md](./05-tooling-codebases.md)** — reach for this when wiring up the actual install and scratch repo.
7. **[CURRICULUM.md](./CURRICULUM.md)** — use as the meta-track for deliberate skill-building.

---

## 📓 How to use this folder as a living research journal

This is **not** a write-once reference — it is a governed, living log that compounds knowledge run over run.

- **Predict before you run.** Before every experiment, write down what you *actually expect* (not what you hope) and why. Track your calibration rate over time — experts hit ~70–80%; beginners ~50%.
- **Keep a flat `log.md`** with one entry per run: *hypothesis → change → result → verdict → intuition-update*. After two weeks, review for systematic biases in your predictions.
- **One branch per experiment.** Merge only validated wins, so the git history is an auditable record of what actually worked.
- **Close the loop every session** (per the global workflow): update `docs/changelog.md` with progress, `docs/known_issues.md` with mistakes + lessons, `docs/decisions.md` with process decisions.
- **Governed memory, not append-only.** When a finding is superseded by a real measurement, *replace* the claim — don't stack contradictions. Stamp findings with a date and re-verify stale ones.
- **Resolve the open questions empirically.** Each doc ends with explicit open questions (throughput on *this* GPU, Muon at small scale, vocab sweet spots, etc.). These are your experiment backlog — the docs hold estimates; your runs replace them with ground truth. When you measure one, update the doc in place and note it in the journal.

The discipline is the product: *implement → verify → register the lesson*, on repeat.
