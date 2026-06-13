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

