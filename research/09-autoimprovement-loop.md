# 09 — The Auto-Improvement Loop (arXiv + Hugging Face + x.ai → experiments)

> **TL;DR (Objective 3).** This project doesn't end at one trained model — it ends with a **standing loop** that ingests the field's newest SOTA (from **arXiv**, **Hugging Face**, and **X / x.ai**), filters it down to what actually applies to a *sub-0.5B model on an 8 GB card*, turns each kept idea into a concrete experiment in the [`08`](./08-open-questions-next-experiments.md) backlog format, verifies it cheaply (proxy-first), and — only if it wins — **supersedes** the relevant research doc and updates the model. It is the small-tank-shaped port of claudemaxxing's `harness-scan` + `/self-improve` pattern. The **iron rule of verification** governs it: you author the contract (the metric/eval), the loop reads pass/fail — never re-derive a worker's work to check it. Fetch is deterministic; *judging applicability stays gated.*
>
> `created: 2026-06-13` · pillar tags reused from `00`–`07`: `architecture` `training` `data` `tokenizer` `eval` `hardware` `method`

---

## 1. Why a loop, not a one-shot

The model is the *artifact*; the loop is the *capability*. Objective 1 (AI research) and Objective 2 (AI engineering) make you able to **read a new paper and run the one ablation that falsifies it before lunch**. Objective 3 makes that habit **systematic and continuous**: every week, the frontier moves, and a disciplined intake turns that motion into experiments on *your* model instead of FOMO. The same governance that runs your memory vault (supersede on conflict, decay + re-verify, gate only belief-changing writes) runs this loop.

**The loop is also self-referential** (the payoff): once your model is "minimally useful," it can become a *worker in its own improvement* — e.g. a cheap first-pass classifier that scores arXiv abstracts for relevance (§6). That closes the auto-improvement circle: a model that helps decide how to improve itself.

---

## 2. The five stages (mirrors `harness-scan` → `/self-improve`)

```
 SCAN ──▶ TRIAGE ──▶ PROPOSE ──▶ VERIFY ──▶ INTEGRATE / SUPERSEDE
 (deterministic)  (LLM-gated)  (backlog)  (proxy-first)  (governed doc edit)
```

| Stage | What happens | Who does it | Output |
|---|---|---|---|
| **1. SCAN** | Pull last-N-day candidates from arXiv + HF + x.ai. **Deterministic** — no judgment. | `scripts/scan.py` + `xsearch.py` | `research/intake/YYYY-WW.jsonl` (raw candidates, pillar-tagged) |
| **2. TRIAGE** | Judge each candidate: *does this claim apply at sub-0.5B / 8 GB? what would test it?* | LLM (gated, cross-model critic for belief-changing keeps) | shortlist with applicability verdict |
| **3. PROPOSE** | Convert each kept item into an experiment spec in the `08` §3 format | you + LLM | new rows appended to the `08` backlog |
| **4. VERIFY** | Run the smallest experiment that could confirm/refute. Proxy first. | the training loop | a number in `research/log.md` + W&B |
| **5. INTEGRATE** | If it wins: **edit the losing doc in place**, update model/config, stamp `last_verified`. If it loses: log and drop. | you (governed) | superseded doc + journal entry |

**Cadence:** weekly SCAN+TRIAGE (cheap), experiments as backlog capacity allows. Don't let intake outrun verification — a 50-item shortlist you never test is noise.

---

## 3. The three sources & how to pull them

### 3.1 arXiv — the primary research feed (`architecture` `training` `data` `tokenizer` `eval`)
Use the public arXiv API (Atom; no key). Watch **cs.LG, cs.CL, cs.AI**. Filter by recency + small-model keywords, then TRIAGE for applicability.

```bash
# scripts/scan.py (sketch) — last 7 days, small-LM relevant
# arXiv API: http://export.arxiv.org/api/query
#   search_query=cat:cs.CL+AND+(abs:"small language model" OR abs:"efficient" OR
#                abs:"tokenizer" OR abs:"optimizer" OR abs:"distillation" OR abs:"data quality")
#   sortBy=submittedDate&sortOrder=descending&max_results=100
```
Keyword seeds (tune over time): `small language model`, `sub-billion`, `efficient pretraining`, `data curation / synthetic data`, `tokenizer / vocabulary`, `optimizer (Muon, Adam-mini, SOAP)`, `distillation`, `scaling laws`, `learning-rate schedule (WSD)`, `quantized training`, `single GPU`.
Also worth a standing watch: **Hugging Face Daily Papers** (`https://huggingface.co/papers`) — community-curated, already relevance-filtered.

### 3.2 Hugging Face — models, datasets, papers (`data` `architecture` `eval`)
Use the HF Hub API / `huggingface_hub`:
```python
from huggingface_hub import HfApi
api = HfApi()
# new/trending small models to learn architecture & recipes from:
api.list_models(sort="trending", limit=30, filter="text-generation")
# new datasets relevant to tiny-model training:
api.list_datasets(sort="trending", limit=30)
```
Watch specifically: SmolLM family updates, FineWeb/FineWeb-Edu refreshes, Cosmopedia, new tiny-model configs (Qwen, Gemma, Llama small), and any "we trained a <0.5B model" release — read its `config.json` + model card for the exact recipe.

### 3.3 X / x.ai — the practitioner frontier (`method` `training`)
The fast-iteration discourse (yacineMTB, Keller Jordan / modded-nanoGPT speedruns, PufferLib, karpathy) breaks on X before it's a paper. **`xsearch.py` is already in this repo** for exactly this — the x.ai/X pipe:
```bash
# recent practitioner signal on small-model training
./xsearch.py "small language model single GPU training new technique" --days 14 --sources x,web
./xsearch.py "nanogpt speedrun OR muon OR modded-nanogpt result" --days 30 --sources x
```
Treat X findings as **leads, not facts** — they point you at a repo/paper to verify, they don't end the verification.

---

## 4. TRIAGE — the applicability gate (the part that needs judgment)

For each candidate, answer four questions. Keep only items that survive all four:

1. **What is the concrete claim?** (one sentence, with the number)
2. **Does it plausibly apply at sub-0.5B / 8 GB?** Reject Hopper-only (FP8, FA3), 8×GPU-only, or >1B-only results — *unless* there's a portable idea inside. (This is where most frontier tricks die: `08` §1.6/§2.1 show how easily a multi-GPU result misleads a single-card project.)
3. **Which research doc does it touch?** (pillar tag → which of `00`–`07` it would supersede)
4. **What is the smallest experiment that would confirm/refute it on the micro-proxy?** If you can't name one, it's not actionable yet — park it.

**Governance (from the global memory rules):** only **belief-changing** keeps get a **cross-model critic** pass before they enter the backlog as "promising." Everything else is logged but not acted on. This prevents the backlog from filling with plausible-but-unfalsifiable noise.

---

## 5. PROPOSE → VERIFY → INTEGRATE

- **PROPOSE:** append the kept item to the [`08` §3 backlog](./08-open-questions-next-experiments.md#3-prioritized-backlog) using its exact format: *hypothesis · what it retires · rough time on the 3060 Ti · the one number to record.* Now it's a first-class experiment, not a bookmark.
- **VERIFY:** run it **proxy-first** (micro-proxy 5M, per [`DECISIONS.md`](./DECISIONS.md) D7). The **iron rule**: author the contract (the metric in `evaluate.py`, the fixed probe set) once; read pass/fail. Never re-train at scale just to "see" — that violates the rule and the ETH-Zurich ethos both.
- **INTEGRATE / SUPERSEDE:** if it wins, **edit the losing doc in place** (don't append a contradiction — that's exactly the mess the critic cleaned up in `08`), update the model/config, stamp `last_verified`, and journal the change. If it loses, one line in `log.md` and move on. A refuted idea is a *result*, not a waste.

**Decay & re-verify** (from the memory-governance rules): tag each integrated finding with a date; re-verify `method`/external claims after ~30 days and project-internal numbers after ~14 days, since the field and your own measurements both drift.

---

## 6. Automation & the self-referential payoff

- **Manual first, then automate.** Build `scripts/scan.py` incrementally: start by running the three queries by hand once a week and pasting results into `intake/`. Automate only the parts that are deterministic and boring (the fetch), never the triage judgment.
- **Scheduling:** once `scan.py` is solid, a weekly `/loop` or cron-style schedule can run SCAN+TRIAGE and drop a digest in `research/intake/`. Keep a human (you) in the INTEGRATE loop — superseding a doc is a belief change, not a cron job.
- **Digest format:** one markdown file per week — `## <title> · <pillar> · <source> · <link>` then *claim / applies? / experiment / verdict*. This *is* the running record of how the project tracked the frontier.
- **The model improves the loop (stretch):** when the 30M/125M model is useful, fine-tune or prompt it as a **first-pass relevance classifier** over arXiv abstracts (a tiny, cheap, local filter before the LLM triage). Now your model is a worker in its own improvement — Objective 3, fully closed.

---

## 7. 🎓 Learn-by-doing

1. **Week-1 intake by hand (30 min):** run the three §3 queries manually, collect ~15 candidates into `research/intake/2026-W24.jsonl`, and TRIAGE them with the four §4 questions. Notice how many "exciting" results die at question 2 (not applicable at your scale). That instinct *is* research taste.
2. **Build `scan.py` for one source (arXiv):** parse the Atom feed, filter by keyword + date, emit JSONL. ~50 lines. Verify it against what you found by hand.
3. **Run one full loop end-to-end:** take the single most applicable item from your first intake, PROPOSE it into `08`, VERIFY it on the micro-proxy, and INTEGRATE/supersede the doc it touches. One full turn of the crank teaches more than reading ten papers.
4. **Calibrate the gate:** after a month, review your `intake/` — how often did a TRIAGE "promising" survive VERIFY? If ~everything you kept failed, your gate is too loose; if you rejected something that later proved big, too tight. Tune the keyword seeds and the applicability bar.

---

### Sources & lineage
- claudemaxxing `bin/harness-scan` (external-research intake for `/self-improve`) and the orchestrator memory-governance rules — the pattern this doc ports.
- `xsearch.py` (this repo) — the x.ai/X ingestion pipe.
- arXiv API: http://export.arxiv.org/api/query · Hugging Face Daily Papers: https://huggingface.co/papers · `huggingface_hub` HfApi.
- Internal: ties into [`08`](./08-open-questions-next-experiments.md) (backlog format), [`DECISIONS.md`](./DECISIONS.md) (governance), [`00-eth-zurich-method.md`](./00-eth-zurich-method.md) (predict→run→journal).
