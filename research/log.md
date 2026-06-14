# Research Log — small-tank

The single highest-leverage habit in this project (`00-eth-zurich-method.md`): one entry per run,
**prediction written before the run**, one variable changed per run. Track your calibration over time
(beginners ~50%, experts ~70–80%). The notebook is the compounding asset.

## Entry template
```
## RUN <id> — <YYYY-MM-DD> — <one-line title>
HYPOTHESIS:   <what I expect and WHY, written BEFORE running>
CHANGE:       <the single thing that differs from the last run>
CONFIG:       <configs/<name>.yaml + git commit hash>
RESULT:       <val loss / bpb / PIQA / tok/s / peak VRAM>
VERDICT:      <win / loss / neutral — vs. my prediction>
INTUITION:    <what this updates in my mental model>
UNVERIFIED→:  <which DECISIONS.md / 08-§4 claim this retires, if any>
```

---

## RUN 000 — 2026-06-13 — Stage 0: environment up, toolchain verified on Python 3.13
HYPOTHESIS:   torch+CUDA installs clean for sm_86; the real risk is bitsandbytes/datatrove
              wheels on Python 3.13 (DECISIONS.md D7) — expected ~50/50 they'd need a 3.11 fallback.
CHANGE:       fresh repo; uv venv (.venv, Python 3.13.3); installed torch + core tooling.
CONFIG:       torch 2.6.0+cu124 · driver 580.95.05 (CUDA 13 capable) · commit 329e934
RESULT:       cuda available ✓ · device RTX 3060 Ti · sm_86 ✓ · bf16 supported ✓ ·
              SDPA/FlashAttn-2 path ✓ · VRAM 8.2 GB · numpy/datasets/tokenizers/wandb/yaml/tqdm/
              datatrove ✓ · bitsandbytes 0.49.2 + AdamW8bit importable ✓
VERDICT:      WIN — better than predicted. ALL wheels (incl. bitsandbytes, datatrove) install &
              import on 3.13. No 3.11 fallback required.
INTUITION:    The CUDA-13 driver is backward-compatible with cu124 wheels; bitsandbytes ships a
              working sm_86 build for cp313. Stage-0 risk was overestimated by the docs.
UNVERIFIED→:  RETIRES DECISIONS.md D7 (Python 3.13 ecosystem risk). Still pending: lm-eval install
              (deferred to Stage 4). NOT yet measured: throughput/VRAM (08 Experiment 0 — next).

## RUN 001 — 2026-06-13 — 08 Experiment 0: throughput + VRAM probe (scripts/probe.py)
HYPOTHESIS:   Real tok/s would land between doc 04 (8–12K for 50–150M) and doc 07 (30M 108–180K,
              125M 26–43K), which conflict 2–4x. Predicted ~30M ≈ 60–90K; 125M needs 8-bit+ckpt to fit.
CHANGE:       First measurement on real hardware. bf16, SDPA, seq 1024, AdamW, no grad-ckpt (unless noted).
CONFIG:       scripts/probe.py @ commit 3b42cb0; RTX 3060 Ti; torch 2.6.0+cu124.
RESULT (measured, NOT estimated):
              5.3M  : 217K tok/s @ b32 (4.34GB) · MFU ~34%
              14.8M : 75K tok/s @ b16 (5.85GB) · OOM @ b32
              26.7M : 70K tok/s @ b16 (5.59GB) · MFU ~45% · OOM @ b32
              88.1M : OOM @ b8 with plain AdamW;  FITS w/ 8-bit AdamW + grad-ckpt → 22K tok/s @ b24 (5.7GB)
              SDPA vs eager (5M b8): 194K vs 61K tok/s (3.2x faster) AND 1.18 vs 3.83 GB (3.2x less mem)
              eager 30m-deep b16: OOM (eager can't even fit where SDPA uses 5.85GB)
VERDICT:      WIN. Conflict settled: doc 07 ~2x optimistic for 30M (real ~70K, not 108–180K);
              doc 04 ~6–9x too pessimistic (refuted). doc 07's 125M range (26–43K) ≈ right (~22K w/ ckpt).
              125M-class FITS 8GB only with 8-bit AdamW + grad-ckpt. SDPA is mandatory (memory, not just speed).
INTUITION:    At seq 1024 eager attention materializes [B,nh,T,T] and dominates memory → SDPA/FlashAttn-2
              is the enabler, not an optimization. ~30M trains at ~70K tok/s → 1.5B tokens ≈ ~6h (RECOMMENDATION's
              "2–4h" was optimistic). CAVEAT: probe configs undershoot labels (30m-deep=14.8M, 125m=88.1M);
              true 30M/125M will be a bit slower/heavier — tune configs/*.yaml to hit exact param targets in Stage 1.
UNVERIFIED→:  RESOLVES DECISIONS.md D6 + 08 §2.1 (throughput conflict). Strong preview of Exp 4 (SDPA speedup).
              Still open: true-30M throughput (config undershoots); 8-bit AdamW *quality* impact (Exp 5).

## RUN 002 — 2026-06-13 — BUILD-PLAN Stage 1: full pipeline end-to-end (5M on TinyStories)
HYPOTHESIS:   A 5M model trained ~7 min on TinyStories would become grammatical & on-topic but
              wobble on long-range consistency (the canonical TinyStories result at this scale).
CHANGE:       First real training run. Built src/{model,data,train}.py + scripts/generate.py from scratch
              (2026 stack: RoPE/RMSNorm/SwiGLU/GQA/SDPA, tied embeds). 68M train tokens, vocab 4096.
CONFIG:       configs/5m.yaml, 5000 steps, bs32 seq512, AdamW lr6e-4 cosine, bf16. commit 2018891+.
RESULT:       5.28M params (embedding 20%) | val loss 8.32 → 1.844 (smooth, no NaN/divergence) |
              206K tok/s (confirms Exp 0's 5M ~217K) | 2.4 GB peak | 82M tokens seen (~15 tok/param) |
              GENERATION: coherent toy stories — named chars, dialogue, arcs, ZERO token loops.
              Minor long-range slips (a bear "Max" greeted as "Timmy") = expected 5M signature.
VERDICT:      WIN — gate cleared. The whole loop works: data→tokenizer→train→checkpoint→generate.
              Project's core thesis (TinyStories → coherent tiny models) CONFIRMED on this hardware.
INTUITION:    At 5M, grammar is fully learned but entity tracking isn't — capacity goes to local
              fluency first. val 1.84 vs the ~1.3 of a 28M ref ⇒ headroom from scale + more tokens.
UNVERIFIED→:  Note: embedding fraction came out 20% (vocab 4096 @ d256), not the ~14% D3 targets —
              to hit 14% the proxy needs vocab ~2900 OR slightly more params. Refine in Exp 3.
              Next: Stage 2 (ablate RoPE/RMSNorm/SwiGLU/GQA vs baseline) OR size true-30M for Exp 2.

## RUN 003 — 2026-06-13 — BUILD-PLAN Stage 2: additive architecture ablation (5M proxy)
HYPOTHESIS:   Each modern component beats the GPT-2-era baseline; expected RoPE small win, RMSNorm ~neutral,
              SwiGLU small win, GQA ~neutral on loss but lighter. All at 2500 steps, same seed/tokens.
CHANGE:       Refactored src/model.py to make pos/norm/mlp/attn swappable behind config; scripts/ablate.py
              runs the cumulative ladder. One variable added per rung.
RESULT (003a, BUGGED baseline = sinusoidal PE): baseline val 3.745, "+RoPE" -1.685 (implausible).
        ROOT CAUSE: unit-magnitude sinusoidal PE added to 0.02-init token embeddings → PE swamps token signal.
        FIX: baseline uses learned absolute PE (GPT-2 canonical); sinusoidal path now scales token emb by sqrt(d).
RESULT (003b, CORRECTED):
              baseline (learned PE/LN/GELU/MHA) 5.90M  val 2.194   304K tok/s
              + RoPE                            5.77M  val 2.059  (-0.135)  241K
              + RMSNorm                         5.77M  val 2.064  (+0.005)  203K
              + SwiGLU                          5.87M  val 2.040  (-0.024)  193K
              + GQA (full modern)               5.28M  val 2.054  (+0.015)  205K
              Full modern vs baseline: -0.140 val, mostly RoPE + a bit of SwiGLU; RMSNorm/GQA quality-neutral.
VERDICT:      WIN (and a great lesson). Matches literature: RoPE real-but-modest at seq512, SwiGLU small win,
              RMSNorm neutral (value = simplicity/scale), GQA neutral-on-loss + cuts params 5.87M→5.28M (KV cache).
INTUITION:    The SOTA stack is NOT a free lunch at toy scale: full-modern is SLOWER (205K vs 304K tok/s) for
              -0.14 loss. RoPE's length-extrap, GQA's inference KV savings, RMSNorm's at-scale stability pay off
              at LARGER scale / longer context / inference — not visibly at 5M/seq512. Also: my RMSNorm (float32
              upcast, non-fused) is slower than fused F.layer_norm; torch.compile would likely close that.
              The bug itself is the deepest lesson: an unfair baseline made RoPE look 12x better than real.
UNVERIFIED→:  Confirms each component's role qualitatively. Open: re-measure norm speed under torch.compile;
              GQA/RoPE benefits at the true-30M / longer-seq regime. Next: Lesson 1 + Exp 2 (true-30M depth-vs-width).

## RUN 004 — 2026-06-14 — Lesson A1.Do.2: remove the causal mask (label-leakage demo)
HYPOTHESIS:   (written BEFORE running) Flipping is_causal=True→False lets position i attend to position
              i+1, which IS the prediction target → label leakage. So VAL LOSS will DROP much faster /
              far lower than the causal baseline (the number looks GREAT). BUT GENERATION will be WORSE:
              the model learns to depend on future tokens that don't exist at autoregressive inference
              time (train/inference mismatch). Net lesson predicted: a better-looking metric, a broken model.
CHANGE:       src/model.py Attention.forward: F.scaled_dot_product_attention(..., is_causal=False).
              ~200 steps, otherwise configs/5m.yaml unchanged. (Revert after.)
CONFIG:       configs/5m.yaml, 250 steps, eval every 50. Controlled pair (a1_causal vs a1_nomask),
              one variable = is_causal. Run on ricardoubuntu, commit 493ec82 (edits reverted, not committed).
RESULT:       control  is_causal=True : val 8.32 → 3.44  (normal descent, matches RUN 002 trajectory)
              no-mask  is_causal=False: val 8.30 → 0.29  (COLLAPSE — ~12x lower, still plunging)
              0.29 is impossible for an honest 5M TinyStories LM (RUN 002 = 1.84 @5000 steps; 28M ref ~1.3)
              ⇒ unambiguous label leakage. (Generation not sampled; checkpoints deleted to protect 5m.pt.)
VERDICT:      WIN — both predictions correct. Val loss drops far faster/lower (pred #1 ✓); the cause is
              leakage, so generation would be broken (pred #2 ✓ by inference). Calibration: 2/2 this run.
INTUITION:    The causal mask is what makes val loss an HONEST proxy for the autoregressive task. Remove it
              and you optimize bidirectional fill-in — a task you can never play at inference (no future
              tokens exist when generating left-to-right). A metric and a model can point opposite ways:
              a spectacular-looking loss can mean a broken model. Generalizes: any train/inference input
              mismatch (leakage, teacher-forcing-only signals) inflates the metric while degrading deployment.
UNVERIFIED→:  Lesson A1 (Attention Is All You Need) closed. Confirms: attention is permutation-equivariant
              (order must be injected); causal masking is what defines a *language* model vs a fill-in model.


## RUN 005 — 2026-06-14 — Lesson A2 / Exp 3: pure vocab-allocation sweep (embedding vs body)
HYPOTHESIS:   (Claude's prediction; Ricardo delegated this round.) Hold total params ~5.3M and sweep
              vocab V∈{512,1k,2k,4k,8k,16k}, solving d_ff so only the embedding/body split moves.
              Compare in BPB (per-token loss would rig it toward small V), equal-BYTES budget (100MB).
              Predict a shallow U: BPB-minimum around V=2k–4k. High V (8k,16k) loses to body starvation
              + Zipfian-tail undertraining (16k is degenerate: d_ff=26, basically no MLP). Low V (512–1k)
              is body-rich (d_ff up to 908) so it stays close, but pays at fixed seq_len=512 — finer
              tokens → longer text per story → less narrative per window + capacity spent on byte-assembly,
              so it ticks slightly above the optimum rather than winning. Net: interior optimum, gentle
              left shoulder, steep right cliff. Secondary watch: does the BPB-optimal V land near D3's
              4096 (chosen for proxy fidelity, not loss) — a coincidence worth interrogating?
CHANGE:       New rig: model.py d_ff override (body-resize knob); data.py per-vocab artifacts
              (prepare_vocab + SMALLTANK_DATA_DIR); scripts/vocab_alloc.py. One variable swept = vocab.
CONFIG:       configs/5m.yaml base, d=256/6L fixed, batch32 seq512, ~5.3M params/rung, 100MB byte budget.
RESULT:       <pending — run on ricardoubuntu>
VERDICT:      <pending>
INTUITION:    <pending>
UNVERIFIED→:  <pending>  Tests DECISIONS.md D3 (vocab budget) as a *loss* question, not just fidelity.
