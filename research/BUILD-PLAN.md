# BUILD-PLAN.md — Staged Roadmap to a From-Scratch Small LM on 8 GB

**TL;DR:** This is the concrete engineering roadmap that turns research docs `00`–`07` into a trained model on a single RTX 3060 Ti (8 GB, sm_86, Python 3.13). It is organized into 6 stages (0–5), each with an **objective**, **exact commands/steps**, a **deliverable**, a **smoke test** that runs in minutes, and a **what-you-learn checkpoint**. The spine is the ETH-Zurich ethos (`00-eth-zurich-method.md`): keep runs short (minutes for proxies), train *many* models, write a prediction before every run, and only graduate to a bigger model once the small one has taught you what you need. The critical-path arc is: **env → reproduce a tiny TinyStories model end-to-end (<1 hr) → modernize the architecture (RoPE/RMSNorm/SwiGLU/GQA) → scale to a 30M→125M target → eval + iterate forever.** Do not skip Stage 1; a working end-to-end loop on a trivial model is worth more than a perfect architecture you can't train.

---

## How to use this plan

- **One stage at a time, gated by its smoke test.** A stage is "done" only when its smoke test passes *and* you have written the learning-checkpoint note into `research/log.md`.
- **The fast-iteration loop is the product.** Every stage is designed so the inner loop (edit → run → verdict) is under a few minutes for the proxy model. If a change makes the loop slower than ~5 min for the 30M proxy, that is a bug in your setup, not a fact of life.
- **Write your prediction first.** Before any run that tests a hypothesis, write in `log.md` what you *expect* to happen and why (not what you hope). Track your calibration. This is the single highest-leverage habit in `00-eth-zurich-method.md`.
- **Git discipline:** one branch per experiment, merge only validated wins (Stage 0 sets this up). This keeps your research history auditable and prevents confounded results.
- **Reference docs by filename** are cited inline so you can go deep when a stage gets hard.

---

## Experiment-tracking habit (set up once, use every run)

This is not optional and not a separate stage — it threads through all of them. From `00-eth-zurich-method.md` and `06-evaluation.md`:

1. **`research/log.md`** — a flat, append-only journal. One entry per run, using this template:
   ```
   ## RUN <id> — <date> — <one-line title>
   HYPOTHESIS:   <what I expect and WHY, written BEFORE running>
   CHANGE:       <the single thing that differs from the last run>
   CONFIG:       <link to configs/<name>.yaml + git commit hash>
   RESULT:       <val loss / bpb / PIQA / tok/s / peak VRAM>
   VERDICT:      <win / loss / neutral — vs. my prediction>
   INTUITION:    <what this updates in my mental model>
   ```
2. **Weights & Biases** — log *every* run, x-axis = `tokens_seen` (not steps), so runs with different batch sizes are comparable (`06-evaluation.md`). `pip install wandb` in Stage 0; `wandb login` once.
3. **One change per run.** If you change two things, you've learned nothing. Confounded runs are worse than no runs.
4. **Calibration review:** after ~2 weeks of entries, re-read `log.md` and score how often your HYPOTHESIS matched your VERDICT. Beginners ~50%; the goal is 70–80%.

---

## Stage 0 — Environment & repo scaffold

**Objective:** A reproducible PyTorch+CUDA environment that runs on the RTX 3060 Ti (Ampere sm_86, bf16 + FlashAttention-2 via SDPA), plus a clean scratch repo wired for fast iteration. (Refs: `05-tooling-codebases.md`, `07-hardware-8gb-engineering.md`.)

**Steps / commands:**
```bash
# 0.1 Env manager (uv is fast and reproducible). Research doc recommends Python 3.11
#     for best ML-ecosystem compat, but this box has 3.13. Try 3.13 first; if a wheel
#     (flash-attn source build, bitsandbytes) is missing, pin a 3.11 venv with uv.
cd /home/ricardo/dev/small-tank
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not already present
uv venv --python 3.13 .venv && source .venv/bin/activate
# Fallback if a dep won't install on 3.13:  uv venv --python 3.11 .venv

# 0.2 PyTorch for CUDA (sm_86 is fully supported by modern wheels). Use the current
#     stable cu12x index. Verify the exact index URL at pytorch.org before running.
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# 0.3 Core tooling
uv pip install numpy datasets tokenizers wandb tqdm pyyaml lm-eval
# Optional now, needed later: 8-bit optimizer states (huge on 8 GB — see 07)
uv pip install bitsandbytes
# Note: SDPA gives FlashAttention-2 on Ampere with NO separate flash-attn install (07).
```
```bash
# 0.4 Sanity check the GPU stack
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
torch.backends.cuda.matmul.allow_tf32 = True            # free on Ampere (02)
# FlashAttention-2 path check via SDPA:
from torch.nn.functional import scaled_dot_product_attention as sdpa
q=k=v=torch.randn(2,4,128,64,device="cuda",dtype=torch.bfloat16)
print("sdpa ok:", sdpa(q,k,v,is_causal=True).shape)
PY
```
```bash
# 0.5 Repo scaffold (from-scratch layout per 05-tooling-codebases.md)
git init
mkdir -p configs data scripts src research/checkpoints
touch src/{model.py,train.py,data.py,evaluate.py,optim.py} \
      configs/{5m.yaml,30m.yaml,125m.yaml} research/log.md
printf ".venv/\ndata/\nresearch/checkpoints/\nwandb/\n__pycache__/\n*.bin\n" > .gitignore
git add -A && git commit -m "Stage 0: env + scaffold"
```

**Deliverable:** Activated `.venv` with a working torch+CUDA stack; the 0.4 sanity script prints `bf16 supported: True` and `sdpa ok`; a committed repo skeleton (`src/`, `configs/`, `research/log.md`).

**Smoke test (<2 min):** Run the 0.4 script. It must print the GPU name, `bf16 supported: True`, and a successful SDPA shape. If `bitsandbytes` or any wheel fails to import on 3.13, fall back to the 3.11 venv now (cheaper than discovering it mid-Stage-3).

**Learn-checkpoint:** Confirm the hardware reality from `07-hardware-8gb-engineering.md`: bf16 (no loss scaling, unlike fp16), TF32 on, SDPA = free FlashAttention-2, and **fp8 is NOT available on GA104** (don't chase it). Write one line in `log.md`: torch version, CUDA version, bf16 status.

---

## Stage 1 — Reproduce a tiny TinyStories model end-to-end (<1 hr)

**Objective:** Get a *complete* loop working — data → tokenize → train → checkpoint → generate — on a trivially small model, so every downstream change is a one-variable edit on a known-good baseline. (Refs: `04-datasets.md` for TinyStories, `02-training-recipes.md` for the loop, `06-evaluation.md` for the reading pass.) **This is the most important stage. A coherent 5–15M model beats a perfect untrained architecture.**

**Why TinyStories:** `04-datasets.md` — sub-10M-param models become *coherent* on TinyStories because the closed vocabulary + repetitive narrative structure is compressible by a tiny model. It is the canonical "does my whole pipeline work" dataset.

**Steps / commands:**
```bash
# 1.1 Get TinyStories (~few GB) and a small slice for the smoke test
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
import itertools, pathlib
out = pathlib.Path("data/tinystories_sample.txt")
with out.open("w") as f:
    for ex in itertools.islice(ds, 200_000):     # ~ tens of MB, enough to train a toy
        f.write(ex["text"].replace("\n"," ") + "\n")
print("wrote", out)
PY

# 1.2 Train a byte-level BPE tokenizer, vocab=8192 (03-tokenization.md: tiny models
#     must keep the embedding table small; 8k @ d=256 is ~14% of a 30M budget).
python - <<'PY'
from tokenizers import ByteLevelBPETokenizer
tok = ByteLevelBPETokenizer()
tok.train(["data/tinystories_sample.txt"], vocab_size=8192,
          special_tokens=["<|endoftext|>"])
tok.save_model("data")          # writes vocab.json + merges.txt
# round-trip check (03 mandates this before spending GPU hours)
s="Once upon a time, a small fox found a red ball."
assert tok.decode(tok.encode(s).ids)==s, "round-trip FAILED"
print("tokenizer ok, vocab=8192")
PY
```

**Implementation work:** Write a *minimal* GPT (`src/model.py` + `src/train.py`) — retype nanoGPT-style from `05-tooling-codebases.md` §1 (read it once, type it once; ~250 lines). Use `torch.amp.autocast('cuda', dtype=torch.bfloat16)`, AdamW (lr=3e-4, betas=(0.9,0.95), wd=0.1 on weight matrices only — `02-training-recipes.md`), grad-clip 1.0, and a 5M-param config (`configs/5m.yaml`: d=256, n_layer=8, n_head=8 — a proxy size from `00`/`01`).

**Deliverable:** `src/model.py`, `src/train.py`, a saved 5M checkpoint in `research/checkpoints/`, and a `scripts/generate.py` that samples text from a prompt.

**Smoke test (~10–20 min):** Train the 5M model on the sample slice for a few hundred steps. Two gates: (1) **loss goes down** smoothly (no NaNs, no divergence); (2) `generate.py "Once upon a time"` produces *grammatical, on-topic* toy-story text (it will be simple, that's correct). Per `06-evaluation.md`, do a 5-minute human reading pass: fewer than ~3 of 20 completions should loop or drift.

**Learn-checkpoint:** You now own the full loop. Record in `log.md`: measured **tokens/sec** for the 5M model (this calibrates every time estimate in `07`), peak VRAM (`torch.cuda.max_memory_allocated()`), and one sample completion. The open question "what is real throughput on *this* card" (`00`, `04`, `07`) now has a first data point.

---

## Stage 2 — Modernize the architecture (RoPE / RMSNorm / SwiGLU / GQA)

**Objective:** Upgrade the nanoGPT-era model to the 2026 SOTA small-LM block and *prove each component earns its place* via short ablations. (Ref: `01-architecture-sota.md` is the spec; `00` is the ablation discipline.)

**Target block (from `01-architecture-sota.md`):** decoder-only, **RoPE** (theta=500000), **pre-norm RMSNorm**, **SwiGLU** MLP with intermediate = `int(2/3 · 4 · d)` rounded to a multiple of 256, **GQA** (e.g. 8 query / 2 KV heads), **tied embeddings** (sub-500M), attention via SDPA (`is_causal=True`). Add **QK-norm** only if you train at lr>5e-4 or see loss spikes (`01`).

**Steps:** Implement each component as a *swappable* piece behind the YAML config, then run a one-variable ablation per change against the Stage-1 baseline, same data/tokens/seed:
```bash
# Each run changes exactly ONE thing; same 5M proxy, same token budget, same seed.
python src/train.py --config configs/5m.yaml --pos sinusoidal   # baseline (Stage 1)
python src/train.py --config configs/5m.yaml --pos rope         # +RoPE
python src/train.py --config configs/5m.yaml --pos rope --norm rmsnorm    # +RMSNorm
python src/train.py --config configs/5m.yaml --pos rope --norm rmsnorm --mlp swiglu  # +SwiGLU
python src/train.py --config configs/5m.yaml --pos rope --norm rmsnorm --mlp swiglu --attn gqa  # +GQA
```

**Deliverable:** A modern `src/model.py` where RoPE/RMSNorm/SwiGLU/GQA are config-selectable, plus an **ablation table in `log.md`** (component → val loss → Δ vs. previous → tok/s cost).

**Smoke test (~5–8 min each, run as a batch):** Each proxy run finishes in minutes; the cumulative ablation is one working session. Gate: the modern stack should match-or-beat the baseline val loss at equal tokens, and GQA should *cut KV memory* with negligible loss change (verify with `max_memory_allocated()`).

**Learn-checkpoint:** This is where you *feel* why each SOTA choice exists. Note in `log.md`: which component gave the biggest loss drop, which was nearly free, and whether QK-norm was needed. Run the `01` depth-vs-width mini-ablation (e.g., 8-layer-d256 vs 16-layer-d181 at equal params) to viscerally confirm **depth beats width for tiny models** (MobileLLM finding, `01`).

---

## Stage 3 — Scale toward the target (30M → 125M), with the modern recipe

**Objective:** Take the validated modern block to a "this is a real model" size, using the full training recipe (WSD schedule, over-training, VRAM-fit techniques) on real data (FineWeb-Edu). (Refs: `01` configs, `02-training-recipes.md` recipe, `04-datasets.md` data, `07` VRAM math.)

**Target configs (from `01-architecture-sota.md`):**
- **30M** (primary fast-iteration vehicle): d=256, 18 layers, 8/2 Q/KV heads, d_ffn≈672, rope_theta=500000, tied, vocab=8192–32768. Fits <4 GB at batch=32, seq≈1024–2048. A Chinchilla-ish run is <1 hr (`07`).
- **125M** (the "useful" graduation target): d=512, 24 layers, vocab=16384 @ d=768 per `03`. Needs **gradient checkpointing** + consider **8-bit AdamW / Adam-mini** to fit batch=16 on 8 GB (`02`, `07`).

**Steps / recipe (from `02-training-recipes.md`):**
1. **Data:** stream/download **FineWeb-Edu sample-10BT** (~27 GB) — best web-scale set for tiny models (`04`); optionally mix Cosmopedia v2. Tokenize once into binary shards (datatrove or a simple pack script) so the loop is I/O-free.
2. **Schedule: WSD** (Warmup-Stable-Decay), split ~1% / 89% / 10%. Checkpoint at the *end of the stable phase*, then branch multiple short decay anneals from that one checkpoint — train two models for ~1.1× the cost (`02`). This is the fast-iteration multiplier at scale.
3. **Over-train:** target **100–500 tokens/param** (not Chinchilla's 20) — `02`/`04` show monotonic gains far past compute-optimal for deployed small models. 100 tok/param on 30M is <1 hr on this card.
4. **VRAM fit at 125M:** `model.gradient_checkpointing_enable()`, `bitsandbytes.optim.AdamW8bit` (or Adam-mini), bf16, TF32. Verify peak with `torch.cuda.max_memory_allocated()` — target 5.5–6.5 GB with ~1.5 GB headroom (`07`).
5. **`torch.compile(model)`** for any run >10 min (20–40% throughput; skip during rapid sweeps because of ~60–90 s warmup — `02`/`07`).

**Deliverable:** A trained **30M** checkpoint at ≥100 tokens/param on FineWeb-Edu, and a **125M** checkpoint that fits and trains stably on 8 GB; both logged to W&B with `tokens_seen` x-axis.

**Smoke test (~10–15 min):** Before any multi-hour run, do a *fit-and-flow* check: 200 steps of the 125M config with checkpointing + 8-bit AdamW. Gates: (1) it does NOT OOM (read `max_memory_allocated`, not nvidia-smi — `07`); (2) loss decreases; (3) measured tok/s is within ~2× of the `07` estimate (30M ~108–180K, 125M ~26–43K tok/s). If VRAM is tight, drop seq length or batch before reducing model quality.

**Learn-checkpoint:** Record the **real MFU / tok/s / peak VRAM** for 30M and 125M — this closes the biggest open question across `00/04/07`. Run the `02` over-training experiment (same 30M model at 20 vs 100 vs 300 tok/param) to *prove to yourself* Chinchilla is wrong for this use case. Note whether 8-bit AdamW changed the loss curve at this small scale (`07` flags this as uncertain).

---

## Stage 4 — Evaluate, define "minimally useful", and iterate

**Objective:** Build a frictionless 4-tier eval that runs on every checkpoint, and judge models against a pre-committed "minimally useful" bar. (Ref: `06-evaluation.md`.)

**Steps (`evaluate.py` runs all four tiers, outputs JSON — `06`):**
1. **Loss / bits-per-byte** to W&B every run. `bpb = ce_nats / ln(2) / avg_bytes_per_token`; <1.0 means it learned something, <0.8 on domain text is respectable.
2. **Curated lm-eval-harness suite: PIQA + ARC-easy + WinoGrande only.** **Skip HellaSwag/MMLU below ~150M** — they're near-chance and will mislead you (`06`).
3. **LLM-as-judge** (grammar/consistency/creativity) on ~20 fixed probe prompts — the highest-signal qualitative metric for tiny models, ~$0.003/run (`06`).
4. **5-minute human reading pass** — 20 completions; flag loops/drift.

**Pre-committed "minimally useful" bar (`06-evaluation.md` §7.4) — write this down BEFORE training:** val loss < 2.5; **PIQA > 58% OR ARC-easy > 35%**; judge grammar > 7.0/10; human pass ≥ 10/20 plausible. Meeting **3 of 4 groups** = minimally useful. Ceiling = SmolLM2-135M (42% HellaSwag, 68% PIQA); floor = Cerebras-GPT-111M.

```bash
python src/evaluate.py --ckpt research/checkpoints/30m_best.pt --out eval_30m.json
# emits {val_loss, bpb, piqa, arc_easy, winogrande, judge_grammar, human_pass}
```

**Deliverable:** `src/evaluate.py` (one command → structured JSON, all four tiers), an `eval_*.json` per checkpoint, and a scorecard in `log.md` vs. the 4-group bar.

**Smoke test (~10–15 min):** Run `evaluate.py` on the Stage-1 5M checkpoint *and* a Stage-3 30M checkpoint. Gate: the script completes end-to-end and the 30M clearly beats the 5M on PIQA/judge/bpb. If a benchmark sits exactly at chance, that's expected at this scale — it confirms why `06` says to skip it, not a bug.

**Learn-checkpoint:** You now have a closed loop: hypothesis → train → eval-JSON → verdict → `log.md`. Note which metric is the most *actionable* at your scale (usually judge-grammar + bpb, per `06`). From here, **iterate forever** in the ETH-Zurich spirit: each session, run ≥5 proxy experiments before touching a bigger model.

---

## Stage 5 — (Optional) Post-training: SFT and beyond

**Objective:** Turn a base model into something instruction-following, once the base is solid. (Refs: `04-datasets.md` for SmolTalk/TinyStoriesInstruct, `05-tooling-codebases.md` for TRL.)

**Steps:**
- Start with **TinyStoriesInstruct** (same download as Stage 1, zero extra cost) to prove the SFT loop, then graduate to **SmolTalk** (≤100–200K samples for a sub-100M model — it can't absorb the full 1M; `04`).
- Use **TRL `SFTTrainer`** for instruction tuning; later **GRPOTrainer** (no separate critic, half the memory of PPO) for RL on a narrow task like GSM8K (`05`). Add the ChatML template (`<|im_start|>/<|im_end|>`) only at SFT, never at pretraining (`03`).

**Deliverable:** An SFT checkpoint that follows simple instructions; judge-eval before/after on the same probes.

**Smoke test (~15 min):** SFT the 30M base on a few thousand TinyStoriesInstruct samples; gate = it follows a held-out instruction prompt better than the base model on the LLM-judge.

**Learn-checkpoint:** Note the smallest SFT set that produced a visible behavior change, and whether GQA/tied-embeddings choices from Stage 2 caused any inference quirks. This closes the pretraining→chat arc from `nanochat` (`05`) at a scale that fits 8 GB.

---

## Critical-path summary & guardrails

| Stage | Gate to pass before moving on | Typical inner-loop time |
|---|---|---|
| 0 Env | bf16=True, SDPA ok, repo committed | minutes |
| 1 Repro | 5M loss↓ + coherent toy stories | ~10–20 min |
| 2 Modernize | modern block ≥ baseline, ablation table | ~5–8 min/run |
| 3 Scale | 125M fits 8 GB (no OOM), 30M ≥100 tok/param | proxy <1 hr; 125M hrs |
| 4 Eval | `evaluate.py` JSON; 3/4 "useful" groups | ~10–15 min |
| 5 SFT | follows instructions better than base | ~15 min |

**Guardrails (from the docs):** never run without a written prediction (`00`); one variable per run (`00`); measure VRAM with `max_memory_allocated`, not nvidia-smi (`07`); don't chase fp8 on Ampere (`07`); don't bother with HellaSwag/MMLU sub-150M (`06`); keep the 30M proxy as your daily driver and only spend big-model hours on changes already validated at proxy scale (`00`).
