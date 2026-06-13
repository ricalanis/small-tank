# Lessons — the guided learning track

This is the **active learning track** for small-tank: a sequence of lessons that turn the project into
a path toward becoming a world-class model researcher (objectives 1 & 2). It is the *doing* companion to
[`research/CURRICULUM.md`](../research/CURRICULUM.md), which holds the full 10-week map.

## The method: build first, then read
You are not reading papers cold. By the time a lesson asks you to read *Attention Is All You Need*, you have
**already implemented** the decoder block (`src/model.py`); by the time you read *RoPE*, you have **already
ablated** it (RUN 003). Reading a paper after you've built and measured its idea is far more durable than
reading it first. This is the ETH-Zurich loop applied to learning itself.

Every lesson item has four parts:
- **Read** — the source (paper / blog / video).
- **Extract** — the single idea to take away.
- **Map** — the exact file or run in *this repo* where that idea already lives.
- **Do** — a `read → predict → verify` exercise. Write your prediction in [`research/log.md`](../research/log.md) **before** you run it.

## Lessons
| # | Lesson | Covers | Maps to CURRICULUM |
|---|---|---|---|
| 01 | [01-foundations.md](01-foundations.md) | The transformer, the LM, RoPE, TinyStories, scaling laws; the training loop, bf16, FlashAttention/SDPA, tokenization, VRAM | Weeks 1–2 |
| 02 | *(next)* The modern stack in depth — RoPE/GQA/SwiGLU/RMSNorm + tokenizer budget | Weeks 3–5 |
| 03 | *(next)* Training science — scaling laws, WSD, optimizers, data quality | Weeks 6–8 |

## The meta-skill (use it every lesson): reading a paper in 3 passes
1. **Pass 1 (5 min):** title, abstract, figures, conclusion — decide if it's relevant.
2. **Pass 2 (30 min):** intro + method, skip the proofs — get the mechanism.
3. **Pass 3:** reproduce one equation or result by hand.

Mastery is measured by **predictions, not checkmarks**: when you can predict the shape of a loss curve or the
sign of an ablation delta before running it, you have the skill. Track your calibration in `research/log.md`.
