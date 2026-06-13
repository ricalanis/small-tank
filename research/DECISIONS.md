# DECISIONS.md — Resolved Single-Source-of-Truth

> **Status: authoritative.** When any doc (`00`–`08`, `RECOMMENDATION`, `BUILD-PLAN`, `CURRICULUM`) conflicts with this file, **this file wins**. It resolves the contradictions the critic found in [`08-open-questions-next-experiments.md`](./08-open-questions-next-experiments.md) §2 into a single provisional decision each, so we don't carry two live numbers into the build. Most thresholds here are **provisional** and explicitly tagged `unverified` — they get replaced by ground truth as the experiment backlog (`08` §3) runs. Governed, not append-only: when an experiment settles one of these, **edit the losing doc in place**, update this file, and stamp `last_verified`.
>
> `created: 2026-06-13` · `last_verified: 2026-06-13 (pre-measurement — all numbers provisional)`

---

## D0. Governing process rule (resolves `08` §1.1)
The entire corpus (`00`–`07`) is **modeled, not measured on the actual RTX 3060 Ti**. Treat every throughput / VRAM / wall-clock number as a **hypothesis to falsify**, not a fact. **No planning number is trusted until [`08` Experiment 0](./08-open-questions-next-experiments.md#3-prioritized-backlog) (the ~10-min throughput + VRAM probe) has run.** Run Experiment 0 *first*, in the first session.

---

## D1. The 30M architecture is UNRESOLVED until Experiment 2 (resolves `08` §2.2)
Two specs exist and they disagree on the depth/width axis:
- `RECOMMENDATION` §3 / `configs/30m_tinystories.yaml`: **wide-shallow** — d=512, **8 layers**, 8/2 GQA.
- `01-architecture-sota.md` / `BUILD-PLAN` Stage 3: **deep-narrow** — d=256, **18 layers**, 8/2 GQA.

`01`'s own thesis (MobileLLM: *depth beats width for tiny models*) argues against the wide-shallow choice. **Decision:**
- **Do NOT launch the 30M "main run" until [Experiment 2](./08-open-questions-next-experiments.md#3-prioritized-backlog) (depth-vs-width at fixed ~30M) picks the winner from data.** The candidates are: 8L·d512, 18L·d256, 12L·d≈360 — equal params, equal tokens, equal seed.
- **Provisional default for all pre-Exp-2 work:** the **deep-narrow d=256 family** (consistent with the depth>width thesis). The wide-shallow d=512/8L spec is **superseded as the default** but retained as one of the three Exp-2 candidates.
- Re-verify embedding fraction stays ≤15% for whichever spec wins (see D3).

---

## D2. Success bar — metric SET fixed, thresholds provisional (resolves `08` §2.3)
The 30M model is a **story model**, so benchmarks (PIQA/ARC/HellaSwag) are **out** at this scale (`06`: near-chance, misleading). Adopt the `RECOMMENDATION` §6 **metric set**, with `bits-per-byte` (deterministic, offline) as the **primary gate** and the GPT-judge as **secondary** (the judge needs a paid API + is non-deterministic — `08` §1.8):

| Tier | Metric | Provisional bar | Role |
|---|---|---|---|
| 1 (primary) | **bits-per-byte** (held-out TinyStories) | ≤ 1.0 `unverified` | Deterministic, offline, CI-able gate |
| 2 (primary) | **validation loss / perplexity** | ≤ 1.5 / ppl ≤ 4.5 `unverified` | Anchored only by the ~1.3-loss 28M reference |
| 3 (secondary) | **GPT-judge grammar** | ≥ 8.0/10 `unverified` | Run judge 3× to measure variance before gating (Exp 9) |
| 4 (secondary) | **human reading pass** | ≥ 17/20 coherent, 0 loops `unverified` | 5-min manual read |

- **"Minimally useful" = clears ≥ 3 of 4.**
- Thresholds 8.0 and 17/20 are **unsourced** — anchor them with [Experiment 1](./08-open-questions-next-experiments.md#3-prioritized-backlog) (coherence ladder) + [Experiment 9](./08-open-questions-next-experiments.md#3-prioritized-backlog) (calibrate `evaluate.py` on a public TinyStories checkpoint) before treating them as real.
- **`BUILD-PLAN` Stage 4's PIQA/ARC bar applies ONLY to the 125M FineWeb-Edu level-2 model**, never to the 30M story model.

---

## D3. Tokenizer & proxy parity (resolves `08` §1.2)
- **Target model (30M):** byte-level BPE, **vocab 8192** — embedding ≈ 14% of params. ✅
- **Micro-proxy (5M @ d=256):** vocab 8192 would make embeddings **~42%** of the model → embedding-dominated → ablations won't transfer to the 30M target. **Decision: the micro-proxy uses vocab 4096 (or char-level), chosen so its embedding fraction matches the target's ~14%.** Confirm parity in [Experiment 3](./08-open-questions-next-experiments.md#3-prioritized-backlog) before trusting any 5M ablation.

---

## D4. Token budget — flatten-driven, not fixed tok/param (resolves `08` §2.4)
The "over-train to 100–500 tok/param" rule (`02`/`04`/`BUILD-PLAN`) **assumes a corpus much larger than the token budget.** TinyStories has only ~475M unique tokens, so 100 tok/param = ~6 epochs and 500 = ~30 epochs (memorization risk). **Decision for the 30M TinyStories run: train until validation loss flattens — expect ~2–4 epochs (~50 tok/param) — NOT a fixed tok/param target.** The 100–500 tok/param over-training rule applies to the **125M FineWeb-Edu** level-2 run, where the corpus (10B tokens) dwarfs the budget.

---

## D5. Default optimizer = AdamW (resolves `08` §2.5)
**AdamW is the default for runs 1–N** (lr 6e-4 for 30M, betas (0.9,0.95), wd 0.1 on matrices only, bf16). **Muon is [Experiment 8](./08-open-questions-next-experiments.md#3-prioritized-backlog), not a default** — its ~2× efficiency claim is unvalidated below 100M and may not pay off on a bandwidth-bound card with small 2D matrices. `05`'s "Muon is the optimizer to use" is **superseded** by this.

---

## D6. Throughput planning figure = pessimistic until measured (resolves `08` §2.1)
Docs disagree 2–4× (`04`: ~8–12K tok/s for 50–150M; `07`: 26–43K for 125M). **Until Experiment 0 measures the truth, plan with the pessimistic `04` figure** and label every wall-clock estimate `unverified`. Update both docs to the measured number afterward and delete the loser.

---

## D7. Minor clarifications
- **Dataset size (`08` §2.6):** TinyStories is **~2 GB** for the V1 `train` split alone; **~7.6 GB** if you pull all splits + V2 + Instruct. **Run 1 uses the V1 train split.** (Disk is not a constraint either way.)
- **Proxy naming (`08` §2.7):** **micro-proxy** = 5M / ~50M-token (~5–8 min, HP sweeps & ablations); **target-proxy** = 30M (pre-commit fit checks). Ablate at micro, validate the winner at target.
- **Python env (`08` §1.7):** verify `torch`, `bitsandbytes`, `datatrove` wheels import on **3.13 as the literal first Stage-0 action**; fall back to a uv-pinned **3.11** venv if any fails. SDPA avoids the `flash-attn` dependency entirely.

---

### How to retire a decision
When an experiment from `08` §3 settles one of these: (1) edit the **losing** doc in place (don't leave two numbers live), (2) update the row here with the measured value + drop the `unverified` tag, (3) stamp `last_verified` in both, (4) log the change in `research/log.md`. The corpus is **governed, not append-only**.
