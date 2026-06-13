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

## Start here
- **[`research/`](research/)** is the source of truth. Read [`research/README.md`](research/README.md), then
  [`research/RECOMMENDATION.md`](research/RECOMMENDATION.md) + [`research/DECISIONS.md`](research/DECISIONS.md).
- The headline: a **~30M-param decoder-only transformer trained on TinyStories**, then a 125M FineWeb-Edu
  stretch model. Full reasoning in `research/RECOMMENDATION.md` (resolutions in `research/DECISIONS.md`).
- The build path: [`research/BUILD-PLAN.md`](research/BUILD-PLAN.md). The learning path:
  [`research/CURRICULUM.md`](research/CURRICULUM.md). The next experiments:
  [`research/08-open-questions-next-experiments.md`](research/08-open-questions-next-experiments.md).

## Hardware
RTX 3060 Ti (Ampere sm_86, 8 GB) · i5-12400F (12 threads) · 30 GB RAM · Linux · Python 3.13.

## Layout
```
research/    the source-of-truth docs + the research log (research/log.md)
src/         model.py · train.py · data.py · evaluate.py · optim.py   (to be built — BUILD-PLAN Stage 1)
configs/     5m.yaml · 30m.yaml · 125m.yaml
scripts/     scan.py (arXiv/HF intake) · generate.py
xsearch.py   xAI/Grok web+X search (the x.ai pipe for objective 3)
```

## Tools
`xsearch.py "<query>" --days 14 --sources x,web` — reads `XAI_API_KEY` from env or `~/.config/claudemaxxing/.env`.
