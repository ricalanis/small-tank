# 08 — Open Questions, Gaps, Contradictions & The Next-Experiment Backlog

> **TL;DR (skeptic's read of docs 00–07 + RECOMMENDATION + BUILD-PLAN + CURRICULUM).** The research corpus is unusually coherent and well-cited, but it has **four hard contradictions that must be resolved before Stage 3** (the 30M architecture itself is specified two different ways; the success bar is stated at two different levels; the token budget disagrees; throughput estimates differ 2–4× between docs), a pile of **single-source / extrapolated numbers that have never touched the actual 3060 Ti**, and **one structural process gap**: almost everything is justified by *modeling*, not by a measurement on this card. Nothing here is fatal — but the project should treat docs 00–07 as *hypotheses to falsify*, not facts. This doc lists the gaps, resolves the contradictions where possible, and gives a **prioritized 12-experiment backlog** whose explicit purpose is to convert the corpus's many "estimated / extrapolated" claims into ground-truth numbers in the first two working sessions. The single highest-leverage action: **Experiment 0 (a 10-minute throughput + VRAM probe) retires more open questions than any other run.**

---

## 1. Gaps & risks found

Ordered by how much they can silently waste GPU-hours or mislead a learner.

### 1.1 Process gap: the corpus is modeled, not measured
Every throughput, VRAM, and time-to-train number across `00`, `02`, `04`, `07`, and `RECOMMENDATION` is **derived from spec sheets and scaled from other GPUs** (RTX 3090, RTX 5080), not measured on the 3060 Ti. Each doc *says* this honestly (good), but the **downstream docs then quote those estimates as if settled** (e.g. RECOMMENDATION §5 quotes "108–180K tok/s" as the planning basis for the 2–4 h train-time claim). Risk: if real throughput is 2× lower (plausible — see §2.1 contradiction), every wall-clock estimate, and therefore the "fits the ETH-Zurich minutes ethos" framing, shifts. **Mitigation: Experiment 0 must run before any planning number is trusted.**

### 1.2 Tokenizer ↔ model-size coupling is under-specified at the 5M proxy
Docs `03`/`RECOMMENDATION` nail the budget logic for 30M (vocab 8192 @ d=512 → 14%). But the **standard proxy is 5M params at d=256** (`00`, CURRICULUM). At d=256, an 8192-vocab *tied* embedding is 8192×256 = 2.1M params = **~42% of a 5M model** — far over the 15% ceiling the docs themselves set. No doc states which vocab the 5M proxy should use, yet the proxy is supposed to *predict* 30M behavior. If the proxy is embedding-dominated and the target is not, ablation transfer is compromised. **Gap: define a proxy vocab (likely 4096 or a char-level tokenizer) and verify embedding-fraction parity between proxy and target.** This is also a confound for the Week-5 vocab ablation in CURRICULUM.

### 1.3 No data-licensing / provenance note, and no "does TinyStories transfer" guard
The whole RECOMMENDATION rests on TinyStories being *the* visible win at 30M. But (a) the docs never note that a TinyStories-only model is **domain-locked** — it will be incoherent on anything outside the story domain, which the success bar (story prompts only) conveniently never tests; (b) there's no statement of how the 30M TinyStories result *transfers* to the 125M FineWeb-Edu "level 2," beyond "the pipeline is proven." The pipeline transfers; the *learned model* does not. A learner could over-update on "I trained a useful model" when what's really proven is "my infra works." Worth stating explicitly.

### 1.4 Muon is recommended-but-unvalidated at this scale, in three docs
`02`, `05`, `00`, and CURRICULUM Week 8 all flag the *same* open question — does Muon's ~2× advantage hold below 100M? — yet `05`'s key-findings still assert "Muon is the optimizer to use for all 2D weight matrices." That's a recommendation built on an explicitly-open question. At d=256 the 2D weight matrices are small and the card is memory-bandwidth-bound (`07`), where Muon's Newton-Schulz orthogonalization (compute overhead) may not pay off. **Risk: a learner adopts Muon as default and inherits config complexity for no gain.** Keep AdamW as default (RECOMMENDATION already does this — but `05` contradicts it; see §2.5).

### 1.5 8-bit AdamW quality at small tensor sizes is assumed benign but flagged uncertain
`07` itself notes the quality guarantee of bitsandbytes 8-bit AdamW *weakens* for very small models (more tensors fall under the 4096-element FP32 threshold). Yet `02`/`07`/BUILD-PLAN recommend it "from day one." At 5M–30M many weight tensors *are* small. **Gap: nobody has checked whether 8-bit AdamW changes the loss curve at 5M–30M on this project's models.** Cheap to test (Experiment 5).

### 1.6 `torch.compile` cost/benefit on *this* card is asserted, not measured
"20–40% throughput" with "60–90 s warmup" appears in `02`/`05`/`07`. On Ampere + PyTorch + a tiny model that is memory-bandwidth-bound, the realized gain could be much smaller (compile mostly helps compute-bound or fusion-heavy graphs). And the warmup cost directly attacks the "runs in minutes" ethos: for a 45-min proxy run, 90 s is fine; for a 5-min sweep it's 30% overhead. **No doc states the minimum run length where compile pays off on the 3060 Ti.** Measure it.

### 1.7 Python 3.13 ecosystem risk is acknowledged but not de-risked
The env is **Python 3.13**; docs recommend **3.11** for `flash-attn` / `bitsandbytes` / `datatrove` wheel availability. BUILD-PLAN says "try 3.13, fall back to 3.11." This is the **first thing that can block Stage 0** and it's left as a runtime surprise. The SDPA path avoids `flash-attn`, but `bitsandbytes` (needed for the 8-bit AdamW that the whole 125M VRAM plan depends on) and `datatrove` are the real risks. **Gap: verify these three wheels on 3.13 as a literal first action, before writing any model code.**

### 1.8 Evaluation harness assumes paid API + network, never stated as a dependency
`06`/BUILD-PLAN make the GPT-4-mini judge a *primary* metric and even part of the RECOMMENDATION success bar (grammar ≥ 8.0). That introduces an **external paid API + network dependency + non-determinism** into the supposedly-frictionless eval loop, and the RECOMMENDATION's `evaluate.py` "emits JSON for all four tiers" hides that tier 3 can't run offline or in CI. Also: the judge score's sensitivity to the prompt/rubric is an open question in `06` itself, yet a hard numeric bar (8.0 vs 7.0) is set on it. **Gap: the bar leans on the least reproducible metric; bits-per-byte (deterministic, offline) should be the primary gate, judge secondary.**

### 1.9 No baseline-comparison plan against the reference models
`06` names SmolLM2-135M (ceiling) and Cerebras-GPT-111M (floor), and TinyStories-1M/-33M exist on HF. But **no doc says "run your eval harness on these public checkpoints to calibrate it."** Without that, you don't know if a "PIQA 58%" from your harness means the same thing as the published 58%. **Gap: validate `evaluate.py` against at least one public checkpoint before trusting it on your own.**

### 1.10 WSD decay-branching is recommended but its mechanics are never specified
"Checkpoint at end of stable phase, branch multiple decay anneals, two models for 1.1× cost" appears in `02`/`RECOMMENDATION`/BUILD-PLAN/CURRICULUM — four times — but **no doc gives the decay length, the decay shape (linear/cosine/1-sqrt), or how much the branch actually differs.** It's an unimplemented superpower. Fine as a concept; needs one concrete recipe before Stage 3.

---

## 2. Contradictions to resolve

These are direct conflicts *between* docs. Each needs a single decision recorded in `log.md` before the relevant stage.

### 2.1 ❗ Throughput estimates differ 2–4× between docs 04 and 07
- `04` §3.2: "**8,000–12,000 tok/s for a 50–150M model**" (scaled from RTX 3090).
- `07` §throughput table: **125M = 26,000–43,000 tok/s**, 30M = 108–180K tok/s (scaled from spec / RTX 5080).
For 125M these ranges **don't even overlap** (12K vs 26K). RECOMMENDATION/BUILD-PLAN both quote the `07` numbers, so if `04` is closer to reality, every time estimate is ~2–3× optimistic. **Resolution: Experiment 0 measures the truth; update both docs to the measured number and delete the loser.** Until then, plan with the *pessimistic* `04` figure.

### 2.2 ❗ The 30M architecture is specified two incompatible ways
- `RECOMMENDATION` §3 and `configs/30m_tinystories.yaml`: **d=512, 8 layers, 8/2 GQA, d_ffn=1376, seq 1024.**
- `01-architecture-sota.md` and `BUILD-PLAN` Stage 3: **d=256, 18 layers, 8/2 GQA, d_ffn≈672.**
These are not minor — one is **wide-shallow (8 layers)**, the other **deep-narrow (18 layers)** — and `01`'s entire thesis (MobileLLM, depth>width for tiny models) argues *against* the RECOMMENDATION's own 8-layer choice. RECOMMENDATION even hand-waves this ("TinyStories doesn't need 30 layers") with no ablation. **Resolution: this is exactly what Experiment 2 (depth-vs-width at fixed ~30M) is for — run it and let data pick. Do not start the "main run" until it's settled.**

### 2.3 ❗ The success bar is stated at two different levels
- `RECOMMENDATION` §6: **val loss ≤ 1.5, judge grammar ≥ 8.0, ≥ 17/20 human-coherent, bpb ≤ 1.0**, clear ≥3/4.
- `06`/`BUILD-PLAN` Stage 4: **val loss < 2.5, PIQA > 58% OR ARC-easy > 35%, judge > 7.0, ≥ 10/20**, clear 3/4.
These differ by a lot (loss 1.5 vs 2.5; grammar 8.0 vs 7.0; 17/20 vs 10/20), *and* the metric sets differ (RECOMMENDATION drops PIQA/ARC for the 30M story model — correct per `06`'s "skip benchmarks at this scale" — but then `06`/BUILD-PLAN's bar *requires* PIQA). The RECOMMENDATION bar is domain-appropriate but **stricter and possibly arbitrary** (where does 1.5 / 8.0 / 17 come from? The 28M reference is "~1.3 loss" so ≤1.5 is plausible; 8.0 and 17/20 are unjustified). **Resolution: adopt RECOMMENDATION's *metric set* (loss/bpb/judge/human for the story model, no PIQA) but treat the *thresholds* as provisional until Experiment 1 + a calibration run on a public TinyStories checkpoint anchor what "good" actually is.**

### 2.4 Token budget for the 30M run disagrees
- `RECOMMENDATION` §4: **~1.5B tokens, ~3 epochs, ~50 tok/param** ("narrow domain saturates fast").
- `02`/`04`/`BUILD-PLAN` Stage 3: **100–500 tok/param** (so 3–15B tokens for 30M).
TinyStories only has ~475M unique tokens, so 100 tok/param = 3B = ~6 epochs and 500 tok/param = 15B = ~30 epochs — heavy repetition that may just memorize. RECOMMENDATION's 3-epoch / 50-tok-param call is more defensible *for this dataset*, but it contradicts the general "over-train to 100–500" rule the other docs state universally. **Resolution: the over-train rule is dataset-size-bounded; for a 475M-token corpus, run until val loss flattens (likely 2–4 epochs) rather than to a fixed tok/param. Make the doc say "tok/param targets assume a corpus larger than the budget."**

### 2.5 Default optimizer: AdamW vs Muon
- `RECOMMENDATION` §1 + `02`: **AdamW default; Muon as a deliberate later experiment.** ✅ (correct, given the open question)
- `05` key-findings: "**Muon is the optimizer to use for all 2D weight matrices**" — stated as settled.
**Resolution: AdamW is the default for runs 1–N; Muon is an Experiment (see backlog #8), not a default. Soften `05`'s claim to match `02`/RECOMMENDATION.**

### 2.6 Dataset size label: "~2 GB" vs "~7.6 GB"
Minor but sloppy: `04` TL;DR and §summary call TinyStories "~2 GB," while `04`'s own table sums the splits to "~7.6 GB total download" (the 1.92 GB figure is just the V1 train split). RECOMMENDATION quotes 7.6 GB. **Resolution: it's ~2 GB for the single V1 train split, ~7.6 GB if you pull all splits + V2 + Instruct. Say which you mean in the download step.** (Disk is not a constraint either way.)

### 2.7 Proxy size drift: "5M" vs "30M" as the daily driver
`00`/CURRICULUM standardize on a **5M / 50M-token** proxy (~5–8 min). `01`/`07`/BUILD-PLAN sometimes call the **30M** model itself "the fast-iteration vehicle" (sub-1-hr Chinchilla run). These aren't strictly contradictory (different loop tiers) but the docs use "proxy" loosely. **Resolution: name them — *micro-proxy* = 5M/50M-tok for HP sweeps; *target-proxy* = 30M for pre-commit fit checks. Keep ablations at 5M, validate the winner at 30M.**

---

## 3. Prioritized backlog — the next 8–12 experiments (ETH-Zurich style, fast)

Each is framed as: **hypothesis to test · what it retires · rough time on the 3060 Ti · the one number you write down.** Run roughly in this order; the first three are pure de-risking and should fill one working session.

> **Discipline reminder (`00`):** write your *prediction* before each run; change *one* variable; log to `research/log.md` + W&B with `tokens_seen` on the x-axis.

| # | Experiment | Hypothesis / question | Retires (open Qs / contradictions) | Time | The number to record |
|---|---|---|---|---|---|
| **0** | **Throughput + VRAM probe** (5M, 30M, 125M; bf16; B/S sweep; with & without SDPA) | Real tok/s and peak VRAM differ from spec-scaled estimates | §2.1 (throughput conflict), §1.1, `07`/`04`/`00` open Qs on real MFU | **~10–15 min** | tok/s & `max_memory_allocated()` for each size; realized MFU |
| **1** | **TinyStories coherence ladder** (1M/3M/10M/28M, ~50K steps) | Output becomes coherent < 10M params (paper claim) | `04` Exp1; anchors §2.3 success-bar thresholds; builds reading taste | **< 1 GPU-hr** | param count where human-read flips to "coherent"; val loss at 28M |
| **2** | **Depth vs width at fixed ~30M** (8L·d512 vs 18L·d256 vs 12L·d~360, equal params + tokens + seed) | Depth beats width for tiny models (MobileLLM); resolves which 30M config to ship | **§2.2 (the architecture contradiction)**, `01` Exp1, `01` open Q | ~1–2 hr (3 short runs) | Δ val loss & tok/s per config → pick the 30M spec |
| **3** | **Proxy vocab-fraction parity** (5M @ d256: vocab 4k vs 8k vs char-level; check embed %) | A 5M proxy with 8k vocab is embedding-dominated and won't transfer to 30M | §1.2, `03`/`04` vocab open Qs, validity of all 5M ablations | ~30–45 min | embed-param fraction; val loss vs vocab; chosen proxy tokenizer |
| **4** | **SDPA / FlashAttn-2 speedup** (naive attn vs `F.scaled_dot_product_attention(is_causal=True)`) | SDPA gives 20–30% throughput for one line on Ampere | `02`/`07` FA2-on-Ampere open Q; CURRICULUM Wk4 prediction | ~10 min | tok/s delta, peak-VRAM delta |
| **5** | **8-bit AdamW quality at small scale** (30M: fp32 AdamW vs bnb AdamW8bit, same seed) | 8-bit AdamW changes the loss curve at 5–30M (small-tensor effect) | §1.5, `07` open Q; gates the entire 125M VRAM plan | ~1 hr (2 runs) | final val-loss gap; peak-VRAM saving |
| **6** | **`torch.compile` break-even** (30M: eager vs compiled; measure warmup + steady-state tok/s) | compile pays off only past run length T on this card | §1.6, `00`/`07` open Q on compile in the "minutes" regime | ~20 min | warmup seconds; steady tok/s gain; break-even run length |
| **7** | **Over-train vs epochs on a 475M-token corpus** (30M at 1/3/6 epochs ≈ 16/50/100 tok/param) | Gains continue past Chinchilla but plateau (and risk memorization) on a fixed small corpus | **§2.4 (token-budget conflict)**, `02` Exp3, `04`/`06` open Qs | ~3–5 hr or 1 long run w/ checkpoints | val-loss & judge-grammar vs epochs; epoch where val flattens / train-val gap opens |
| **8** | **AdamW vs Muon at 5M & 25M** (Muon on 2D matrices, AdamW on embeddings/norms; equal tokens) | Muon's ~2× efficiency holds below 100M | **§2.5**, `00`/`02`/`05` Muon open Qs | ~1 hr | tokens-to-loss-X ratio Muon/AdamW; config overhead notes |
| **9** | **Eval-harness calibration on a public checkpoint** (run `evaluate.py` on TinyStories-33M and/or SmolLM2-135M) | My harness reproduces published numbers (±a few pts) | §1.9, §1.8; validates the bar before trusting it | ~20 min | my PIQA/bpb vs published; judge-score variance over 3 reruns |
| **10** | **WSD decay-branch recipe** (one stable checkpoint → 2 decays: linear vs cosine, to LR→0) | Branching gives "2 models for ~1.1× cost" with a real quality delta | §1.10, `02` WSD claim; CURRICULUM Wk7 | ~1.2× one run | cost ratio; val-loss delta between the two anneals |
| **11** | **Data-quality dominates architecture** (same 25M arch on raw-web vs FineWeb-Edu vs TinyStories+Cosmopedia) | Data curation moves quality more than architecture at tiny scale | `04`/`06` central claim; CURRICULUM Wk8 "most important" exp | ~2–3 hr | PIQA/bpb/judge delta from *data alone*, arch fixed |
| **12** | **30M → 125M scaling-law sanity line** (2M/5M/10M/25M at ~20 tok/param, log-log loss-vs-params) | A clean straight line ⇒ pipeline healthy & proxies trustworthy | `00` Exp; whether 5M proxy predicts 30M/125M; underpins all ablation transfer | ~1–2 hr | fitted slope; does 25M fall on the 2M/5M/10M extrapolation? |

**Suggested sessioning.** Session 1: #0, #4, #6 (all infra de-risking, < 1 hr total) + start #1. Session 2: #2 and #3 (settle the architecture + proxy validity — these unblock everything downstream). Session 3+: #5, #7, #8, #12. The "science" runs (#9, #10, #11) come once the pipeline and bar are trustworthy.

---

## 4. Claims that need empirical verification on the actual 3060 Ti

A consolidated checklist of every **estimated / extrapolated / "should"** claim in the corpus that the project quotes as if settled. Treat each as `unverified` in `log.md` until a run confirms it. (Star = highest-leverage / most-quoted.)

- ⭐ **Throughput.** 30M ≈ 108–180K tok/s, 125M ≈ 26–43K (`07`) **vs** 50–150M ≈ 8–12K (`04`). Spec-scaled from RTX 3090/5080, never measured. → Exp 0. *(They disagree 2–4×; see §2.1.)*
- ⭐ **Realized MFU.** "50% with compile + FA2" (`07`) is extrapolated from an RTX 5080. → Exp 0.
- ⭐ **Wall-clock train times.** "30M main run 2–4 h," "125M Chinchilla 10–16 h" (`RECOMMENDATION`/`07`) inherit the unverified throughput. → falls out of Exp 0 + 7.
- ⭐ **VRAM budget formulas.** "12 bytes/param static; 6 bytes/param with 8-bit Adam + grad-ckpt; 350M fits at 5.5–6.5 GB" (`07`/`02`). Allocator fragmentation (+5–15%, `07`'s own open Q) and SDPA workspace are not in the math. → Exp 0 (incl. 125M/350M fit check via `max_memory_allocated`, not nvidia-smi).
- **bf16 + SDPA = free FlashAttention-2 on GA104.** Asserted everywhere; the Ampere FA2 dispatch path *does* exist but the realized speedup is unmeasured. → Exp 4.
- **`torch.compile` 20–40% gain / 60–90 s warmup** on this card and these tiny graphs. → Exp 6.
- **8-bit AdamW: <0.5% perplexity impact**, even at 5–30M where many tensors are small. `07` flags this as weakening. → Exp 5.
- **Muon ~2× efficiency holds below 100M.** Open in `00`/`02`/`05`; `05` still recommends it as default. → Exp 8.
- **TinyStories coherence < 10M params** reproduces on *our* tokenizer/arch (paper used a different setup). → Exp 1.
- **Success-bar thresholds are achievable & meaningful.** "val loss ≤ 1.5 (ppl ≤ 4.5), bpb ≤ 1.0, judge ≥ 8.0, 17/20" — only the ~1.3-loss 28M reference anchors any of these; 8.0 and 17/20 are unsourced. → Exp 1 + 9.
- **Depth > width at 30M for this domain.** RECOMMENDATION's 8-layer choice contradicts `01`'s own MobileLLM-based claim. → Exp 2.
- **Custom 8k BPE beats GPT-2 tokenizer by 20–30% fertility** on TinyStories (`03`). Measurable in seconds, never measured. → cheap add-on to Exp 3.
- **WSD "two models for 1.1× cost"** with a real, useful quality difference between branches. → Exp 10.
- **Educational data worth 5–10× raw web tokens** at this scale (`04`). → Exp 11.
- **bitsandbytes + datatrove install cleanly on Python 3.13** (env reality). → verify in Stage 0, *before* any of the above.
- **GPT-judge score is stable enough to gate on** (±how many points across reruns/rubric changes?) — `06`'s own open Q, yet it's in the success bar. → Exp 9 (run the judge 3× on the same outputs).

---

### How this doc plugs back into the workflow
- Add an `unverified:` tag column to `research/log.md`; every claim in §4 starts unverified and is retired by the experiment that confirms/refutes it.
- When a contradiction in §2 is resolved by data, **edit the losing doc** (don't leave both numbers live) and stamp `last_verified` — the corpus is governed, not append-only.
- Re-run this critic pass after Stage 3: the biggest *new* risk after the first real runs will be silent confounds (LR not transferred under μP, data-loader repetition, eval leakage), none of which the current docs cover yet.
