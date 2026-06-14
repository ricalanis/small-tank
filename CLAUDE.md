# CLAUDE.md — small-tank project context

This file makes Claude Code boot with full project context on **any machine** (it travels in git, unlike
machine-local auto-memory). Read it first, then `research/README.md`, `research/DECISIONS.md`, and the tail of
`research/log.md`.

## What this is
An **open project** to build a *minimally useful* sub-0.5B language model **from scratch** on a single
**RTX 3060 Ti (8 GB)**, using 2026 SOTA architecture/training, via the **ETH-Zurich fast-iteration method**
(short runs, train many models). Threefold objective:
1. **AI research** — learn the bases of training models + research taste.
2. **AI engineering** — learn the nuts and bolts of training models.
3. **Auto-improving model** — a standing loop that ingests new SOTA from arXiv + Hugging Face + x.ai
   (`research/09-autoimprovement-loop.md`; `xsearch.py` is the x.ai pipe).

It is also Ricardo's personal training ground to become a world-class model researcher — so docs and plans are
pedagogical, and there's a guided learning track in `lessons/`.

## Conventions (follow these)
- **Language: English only** (open project), even though Ricardo often writes in Spanish.
- **Research-log discipline:** every training run gets a `RUN NNN` entry in `research/log.md` with a
  **prediction written before the run**, one variable changed per run. See the template at the top of that file.
- **Governed docs, not append-only:** `research/DECISIONS.md` is authoritative and **wins over any conflicting
  doc**. When an experiment settles something, edit the losing doc *in place* and stamp it — don't stack
  contradictions. `research/08-open-questions-next-experiments.md` is the experiment backlog.
- **Commit style:** atomic commits, end messages with the Co-Authored-By trailer.

## Where things are
```
research/    source-of-truth docs (00–09, RECOMMENDATION, BUILD-PLAN, CURRICULUM, DECISIONS) + log.md
lessons/     guided learning track (start: lessons/01-foundations.md)
src/         model.py (TinyLM: RoPE·RMSNorm·SwiGLU·GQA·SDPA, swappable comps) · train.py · data.py
configs/     5m.yaml (built) · 30m.yaml · 125m.yaml (Stage 3)
scripts/     probe.py (Exp 0) · ablate.py (Stage 2) · generate.py · scan.py (arXiv/HF intake)
xsearch.py   xAI/Grok web+X search
```

## Current state (update this as you go)
- **Done:** Stage 0 (env), Exp 0 (throughput/VRAM probe), Stage 1 (coherent 5M TinyStories model, val 1.84),
  Stage 2 (architecture ablations). Runs 000–003 logged.
- **Key measured facts:** ~30M ≈ 70K tok/s on the 3060 Ti; 125M-class fits 8 GB only with 8-bit AdamW +
  grad-checkpointing; SDPA is mandatory (eager OOMs at seq 1024).
- **Next options:** Lesson 1 (active tutoring, in progress), Exp 1 (coherence ladder), Exp 2 (true-30M
  depth-vs-width — gates the main run, `DECISIONS.md` D1).
- To resume: read the **last RUN entry** in `research/log.md` — it always ends with a "Next:" pointer.

## Working across machines (Tailscale)
The repo lives on GitHub (`git@github.com:ricalanis/small-tank.git`) and is reachable from any tailnet machine.
**Only `ricardoubuntu` (100.70.90.85) has the GPU — it is the only machine that can train.** Other machines
(`fleet-ralanis` Mac, `ricalaniscloud` linux) are for reading, editing, and learning.

Typical loop:
```bash
# on any machine: edit code / read lessons / write predictions, then
git add -A && git commit -m "..." && git push

# to TRAIN: changes must run on the GPU box. From a laptop, SSH over tailscale:
ssh ricardo@ricardoubuntu      # or @100.70.90.85
cd ~/dev/small-tank && git pull && source .venv/bin/activate
python -m src.train --config configs/5m.yaml
```
**SSH user is `ricardo`** (not `ricalanis`). Credentials for password login live in the gitignored `.env`
(`RICARDOUBUNTU_SSH_USER` / `RICARDOUBUNTU_SSH_PASS`) — never commit them. Prefer key auth (`ssh-copy-id`) when
set up.
**Does NOT travel via git** (rebuild per machine): `.venv/` (run the Quickstart below), `data/` (re-run
`src.data prepare`), `research/checkpoints/*.pt` (retrain, or `rsync`/`scp` over tailscale if you want a trained
artifact on another box). Machine-local Claude **auto-memory** doesn't travel either — *this file* is its
portable substitute, so keep "Current state" above honest.

## Quickstart on a fresh machine
```bash
git clone git@github.com:ricalanis/small-tank.git && cd small-tank
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU box only
uv pip install numpy datasets tokenizers wandb tqdm pyyaml bitsandbytes datatrove
python -m src.data prepare --train-stories 300000 --vocab 4096            # rebuild data
python -m src.train --config configs/5m.yaml                              # GPU box only
```
The global orchestrator harness (oll/xsearch key/etc.) is bootstrapped separately per machine via
`~/dev/claudemaxxing` — this project's context is self-contained in this file.
