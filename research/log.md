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

