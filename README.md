# small-tank

Building a *minimally useful* sub-0.5B-parameter language model **from scratch** on a single
**RTX 3060 Ti (8 GB VRAM)**, using current (2024–2026) SOTA architecture and training, by following the
**ETH-Zurich / PufferLib fast-iteration method**: keep runs short, train *many* models — because the more
models you train, the more you learn.

This is an **open project** with a threefold objective:

1. **AI research** — learn the bases of training models and research taste.
2. **AI engineering** — learn the nuts and bolts of actually training models.
3. **An auto-improving model** — end with a model that continuously self-improves by ingesting new SOTA
   from arXiv, Hugging Face, and X / x.ai (see [`research/09-autoimprovement-loop.md`](research/09-autoimprovement-loop.md)).

## Status
| Stage | What | Result |
|---|---|---|
| 0 | Env on Python 3.13 | torch 2.6.0+cu124, sm_86, bf16, SDPA ✓ (RUN 000) |
| Exp 0 | Throughput + VRAM probe | ~30M ≈ 70K tok/s; 125M fits w/ 8-bit AdamW+ckpt (RUN 001) |
| 1 | Full pipeline end-to-end | **coherent 5M TinyStories model**, val 1.84, ~7 min (RUN 002) |
| 2 | Architecture ablations | each modern component measured on our own model (RUN 003) |

Live progress is journaled in [`research/log.md`](research/log.md) (predict → run → verify).

## Start here
- **Learn:** the guided learning track is [`lessons/`](lessons/) — start with
  [`lessons/01-foundations.md`](lessons/01-foundations.md). It pairs the foundational AI-research and
  AI-engineering papers/resources with the code you've already built and the runs you've already done.
- **Understand the project:** [`research/`](research/) is the source of truth — read
  [`research/README.md`](research/README.md), then [`research/RECOMMENDATION.md`](research/RECOMMENDATION.md)
  + [`research/DECISIONS.md`](research/DECISIONS.md).
- **The headline:** a **~30M-param decoder-only transformer trained on TinyStories**, then a 125M
  FineWeb-Edu stretch model. Build path: [`research/BUILD-PLAN.md`](research/BUILD-PLAN.md). Full
  curriculum: [`research/CURRICULUM.md`](research/CURRICULUM.md). Next experiments:
  [`research/08-open-questions-next-experiments.md`](research/08-open-questions-next-experiments.md).

## Quickstart
```bash
source .venv/bin/activate
python -m src.data prepare --train-stories 300000 --vocab 4096   # download + tokenize TinyStories
python -m src.train --config configs/5m.yaml                     # train the 5M micro-proxy (~7 min)
python scripts/generate.py --ckpt research/checkpoints/5m.pt     # read its stories
python scripts/probe.py                                          # Experiment 0: measure your card
python scripts/ablate.py                                         # Stage 2: ablate the architecture
```

## Hardware
RTX 3060 Ti (Ampere sm_86, 8 GB) · i5-12400F (12 threads) · 30 GB RAM · Linux · Python 3.13.

## Layout
```
research/    the source-of-truth docs + the research log (research/log.md)
lessons/     the guided learning track (start: lessons/01-foundations.md)
src/         model.py (TinyLM: RoPE·RMSNorm·SwiGLU·GQA·SDPA) · train.py · data.py
configs/     5m.yaml (built) · 30m.yaml · 125m.yaml (Stage 3)
scripts/     probe.py (Exp 0) · ablate.py (Stage 2) · generate.py · scan.py (arXiv/HF intake)
xsearch.py   xAI/Grok web+X search (the x.ai pipe for objective 3)
```

## Tools
`xsearch.py "<query>" --days 14 --sources x,web` — reads `XAI_API_KEY` from env or `~/.config/claudemaxxing/.env`.
