# 06 — Evaluating Tiny Models + Defining "Minimally Useful"

> **TL;DR** Evaluation for sub-150M models requires a layered strategy: start with training loss and perplexity as the primary training-time signal, use a targeted subset of the lm-evaluation-harness (PIQA, ARC-easy, WinoGrande) for standardized comparison, apply GPT-4-as-judge for qualitative generation quality in the TinyStories style, and anchor the entire project around a concrete "minimally useful" definition. This doc proposes that bar: **a 10M–50M parameter model trained on a single thematic corpus that can complete partial prompts coherently and score above chance on at least two standard benchmarks**, validated in under 2 minutes of evaluation on your RTX 3060 Ti.

---

## 1. Why Evaluation Is Hard at Tiny Scale

Most NLP benchmarks were designed for models with hundreds of millions to billions of parameters. At sub-150M scale — and especially sub-50M — several pathologies emerge:

- **Near-chance accuracy on reasoning tasks.** HellaSwag (4-way) has a random baseline of 25%. A 70M Pythia model scores only ~27%. That 2-point margin is not learning; it's noise. Tracking this number as your main signal is misleading.
- **Perplexity is domain-sensitive.** A model trained on Python code will have low perplexity on code and terrible perplexity on Wikipedia prose. Cross-corpus perplexity numbers are incomparable without knowing the evaluation set.
- **Benchmark contamination at tiny scale.** Models trained on large internet corpora (Pile, FineWeb) may have seen benchmark text. Domain-specific or synthetically trained models are less contaminated and often more informative to evaluate.
- **Qualitative failure modes dominate.** At 10M–50M parameters, the most meaningful signal is often human-readable: does the generated text make sense at all? Standard metrics fail to capture this.

The solution is a **tiered evaluation pipeline** that gives the right signal at the right stage of training.

---

## 2. Tier 1: Training-Time Signals (Every Run)

### 2.1 Training Loss and Validation Loss

The cross-entropy loss on your held-out validation set is the most fundamental metric. It tells you:

- Whether the model is learning at all
- Whether you are overfitting (train loss falling but val loss rising)
- Whether architecture or data changes improve generalization

**Target range (rough, for character/word-level text):**
- Random (untrained): `~ln(vocab_size)`, e.g. `ln(50257) ≈ 10.8` for GPT-2 BPE
- Trivial repetition baseline: ~4–5
- Decent small model (50M on a narrow domain): 2.5–3.5
- Good small model (150M, quality data): 2.2–2.8

Track both curves in W&B from step 1. Divergence between them is the earliest overfitting warning you have.

### 2.2 Perplexity

Perplexity is simply `exp(cross_entropy_loss)`. It answers: "how surprised is the model, on average, at each token?" Lower is better. For context:

| Model | Params | Dataset | Val Loss | Perplexity |
|---|---|---|---|---|
| Cerebras-GPT | 111M | The Pile | 2.566 | ~13.0 |
| GPT-2 | 124M | WebText | — | 29.4 (WikiText-2) |
| TinyStories model | 28M | TinyStories | ~1.3 | ~3.7 |
| Your target | 10–50M | Domain corpus | ~1.8–2.8 | ~6–16 |

Note: TinyStories perplexity is low because the vocabulary and task are extremely constrained. Don't compare across domains.

### 2.3 Bits Per Byte (BPB)

Bits per byte is the gold standard for tokenization-agnostic comparison. It is defined as:

```
BPB = (validation_loss_in_nats) × log2(e) / (bytes per token)
```

Or equivalently: `BPB = cross_entropy_loss / log(2) / avg_bytes_per_token`

For BPE tokenizers on English text, average token length is ~4 bytes, so BPB ≈ loss_nats / 4 × 1.443.

**Why BPB matters:** If you switch tokenizers (BPE → character → byte-level), perplexity numbers are incomparable but BPB is comparable. It also allows comparison with byte-level models (MambaByte, ByT5). For a 2026 project using modern BPE, BPB is a clean archival metric.

**Target:** BPB below 1.0 is generally considered "the model has learned something." Below 0.8 on well-curated domain text is respectable for a tiny model.

See: [OLMo evaluation with Paloma (bits-per-byte)](https://arxiv.org/html/2402.00838v4), [Understanding Evaluation Metrics for Language Models](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/).

### 2.4 Setting Up W&B (Minimal Code)

```python
import wandb

wandb.init(
    project="small-tank",
    name=f"run-{model_config['name']}",
    config={
        "n_params": count_params(model),
        "n_layers": model_config["n_layers"],
        "n_heads": model_config["n_heads"],
        "d_model": model_config["d_model"],
        "lr": lr,
        "batch_size": batch_size,
        "dataset": dataset_name,
        "context_len": context_len,
    }
)

# Inside your training loop:
wandb.log({
    "train/loss": loss.item(),
    "train/perplexity": math.exp(loss.item()),
    "val/loss": val_loss,
    "val/perplexity": math.exp(val_loss),
    "train/lr": scheduler.get_last_lr()[0],
    "train/step": global_step,
    "train/tokens_seen": global_step * batch_size * context_len,
})
```

Log by **tokens seen**, not just steps — this makes runs with different batch sizes comparable. W&B is free for individual use. See the [W&B 101 course](https://wandb.ai/site/courses/101/).

---

## 3. Tier 2: Standard Benchmarks with lm-evaluation-harness

### 3.1 Installation

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[hf]"
```

### 3.2 Which Benchmarks to Run (and Which to Skip) at Tiny Scale

| Benchmark | Type | Random Baseline | Useful at <150M? | Notes |
|---|---|---|---|---|
| **PIQA** | 2-way commonsense | 50% | Yes | Physical intuition; 70M models score ~59–62% |
| **ARC-easy** | 4-way science Q&A | 25% | Yes | Elementary level; 70M models score ~38–46% |
| **WinoGrande** | 2-way coreference | 50% | Marginal | 70M models ~49–51% (near chance) |
| **LAMBADA** | Open-ended word pred. | ~0% | No | Requires long-range context; tiny models score <20% and results are highly variable |
| **HellaSwag** | 4-way completion | 25% | No for <150M | 70M models score ~27% — at chance. Useful only for models >150M. |
| **ARC-challenge** | 4-way hard science | 25% | No | Even 410M models score ~22%. Below noise at tiny scale. |
| **MMLU** | 4-way knowledge | 25% | No | Requires factual memorization; meaningless below ~500M |
| **BoolQ** | Binary yes/no | 50% | Maybe | Inconsistent at tiny scale |
| **GSM8K** | Math reasoning | ~0% | No | Near 0% for all sub-500M models without RLHF/special training |

**Recommended tiny-model eval suite:**
```bash
lm_eval --model hf \
  --model_args pretrained=./my_model \
  --tasks piqa,arc_easy,winogrande \
  --device cuda:0 \
  --batch_size auto \
  --num_fewshot 0 \
  --output_path ./eval_results.json
```

VRAM requirement: trivial. A 150M model at FP32 uses ~600 MB VRAM for weights; inference needs ~1–2 GB total. Your 8 GB RTX 3060 Ti is more than sufficient.

### 3.3 Reference Scores: What Numbers Mean at This Scale

Use these as anchors to understand whether your model is competitive:

**Cerebras-GPT-111M** (trained Chinchilla-optimally on The Pile):

| Benchmark | Score |
|---|---|
| HellaSwag (0-shot) | 26.8% |
| PIQA (0-shot) | 59.4% |
| WinoGrande (0-shot) | 48.8% |
| ARC-easy (0-shot) | 38.0% |
| ARC-challenge (0-shot) | 16.6% |
| LAMBADA (0-shot) | 19.4% |

Source: [cerebras/Cerebras-GPT-111M on Hugging Face](https://huggingface.co/cerebras/Cerebras-GPT-111M)

**Pythia-70M** (EleutherAI, The Pile):

| Benchmark | Score |
|---|---|
| HellaSwag (10-shot) | 27.3% |
| ARC (25-shot avg) | 21.6% |
| WinoGrande (5-shot) | 51.5% |

Source: [EleutherAI/pythia-70m on Hugging Face](https://huggingface.co/EleutherAI/pythia-70m)

**SmolLM2-135M** (HuggingFace, 2T tokens on FineWeb-Edu + DCLM):

| Benchmark | Score |
|---|---|
| HellaSwag | 42.1% |
| ARC Average | 43.9% |
| PIQA | 68.4% |
| WinoGrande | 51.3% |
| MMLU (cloze) | 31.5% |
| GSM8K (5-shot) | 1.4% |

Source: [SmolLM2 paper (COLM 2025)](https://arxiv.org/pdf/2502.02737), [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

**Key insight:** SmolLM2-135M's gains over Cerebras-111M (HellaSwag 42% vs. 27%) come almost entirely from **data curation** — training on high-quality educational text rather than raw internet. This matters for your project: a 50M model on a carefully curated domain corpus can outperform a 150M model trained carelessly.

### 3.4 Interpreting Benchmark Results at Tiny Scale

A rule of thumb: **any result within 3–4 percentage points of random baseline is likely noise**, not learning signal. For 4-way tasks like HellaSwag (baseline 25%), meaningful learning means >30%, and you need >35% before you should trust the number. For 2-way tasks like PIQA (baseline 50%), meaningful learning means >58%, because variance is lower.

Do NOT average across benchmarks with different random baselines — this creates misleading aggregate scores.

---

## 4. Tier 3: GPT-4-as-Judge (Generation Quality)

### 4.1 The TinyStories Evaluation Paradigm

The [TinyStories paper (Eldan & Li, 2023)](https://arxiv.org/pdf/2305.07759) introduced a key insight: standard benchmarks fail to capture whether a model can generate **coherent, consistent text**. They replaced benchmarks with GPT-4 grading model outputs on:

- **Grammar:** Is the language grammatically correct?
- **Consistency:** Does the narrative stay coherent throughout?
- **Creativity:** Is there interesting structure, variety, or originality?

This is graded on a 1–10 scale per dimension, optionally with a rubric enforced in the GPT-4 system prompt. The result: a 1M-parameter model trained on TinyStories scores higher on these dimensions than a 125M-parameter GPT-2 trained on diverse text, because domain focus trumps scale for generation coherence.

### 4.2 Implementing GPT-4-as-Judge for Your Project

```python
from openai import OpenAI

JUDGE_PROMPT = """You are an expert evaluator of language model outputs.
Rate the following text on three dimensions, each scored 1-10:

GRAMMAR (1=many errors, 10=perfect grammar)
CONSISTENCY (1=incoherent/contradictory, 10=perfectly coherent throughout)
CREATIVITY (1=repetitive/boring, 10=interesting, varied, engaging)

Return only JSON: {"grammar": N, "consistency": N, "creativity": N, "rationale": "..."}

Text to evaluate:
{generated_text}
"""

def judge_generation(text: str, client: OpenAI) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use gpt-4o-mini for cost efficiency (~$0.001/eval)
        messages=[
            {"role": "user", "content": JUDGE_PROMPT.format(generated_text=text)}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

**Cost estimate:** At $0.15/1M input tokens, judging 100 generations of 200 tokens each costs ~$0.003. Extremely cheap. Use `gpt-4o-mini` for speed and cost; use `gpt-4o` for final evaluation gates.

### 4.3 Prompt Suite for Qualitative Probing

Run these fixed prompts at each checkpoint to track generation quality over training:

```python
PROBE_PROMPTS = [
    # Domain completion probes
    "Once upon a time, there was a little",
    "The robot looked at the screen and",
    "She opened the door and found",
    
    # Structural coherence probes
    "First, you need to",
    "The problem was that",
    "After many years,",
    
    # Knowledge probes (only relevant if training on factual data)
    "The capital of France is",
    "Water boils at",
]
```

Log the full generated text (not just scores) to W&B using `wandb.log({"samples": wandb.Table(data=...)})`. This creates a visual history of how generation quality evolves.

### 4.4 Domain-Specific Qualitative Tests

Define 5–10 "smoke tests" specific to your training domain. For a code model:
- Does it produce syntactically valid Python?
- Does it complete a function signature correctly?

For a story model:
- Does it introduce a conflict and resolve it?
- Are character names consistent across a paragraph?

These are binary (pass/fail) and cheap to evaluate manually.

---

## 5. Tier 4: Qualitative Human Probes

No evaluation pipeline replaces looking at model outputs yourself. At each major checkpoint, generate 20–30 samples and do a 5-minute reading pass. Things to check:

1. **Token repetition loops** — does the model get stuck repeating "the the the"? (Sign of low diversity or insufficient training)
2. **Completion plausibility** — does the text sound like it was written by a human, even a child?
3. **Semantic drift** — does the topic change mid-sentence without reason?
4. **Named entity consistency** — does a character's name stay stable within a generation?
5. **Simple factual anchoring** — does the model know basic facts if trained on factual data?

Build a habit: **every checkpoint run, read 10 random generations.** This is the ETH Zurich method applied to evaluation — fast, cheap, high-signal.

---

## 6. Experiment Tracking: The Full Dashboard

Recommended W&B dashboard layout for fast iteration:

### Training Overview Panel
- `train/loss` and `val/loss` — overlay, plotted against `train/tokens_seen`
- `train/perplexity` (log scale)
- `train/lr` (to verify scheduler)

### Benchmark Panel (run every N steps or at epoch end)
- `eval/piqa_acc`
- `eval/arc_easy_acc`
- `eval/winogrande_acc`

### Generation Quality Panel
- `gen/grammar_score` (GPT-judge avg over probe suite)
- `gen/consistency_score`
- `gen/creativity_score`
- `gen/samples` (W&B Table with full texts)

### System Panel
- `sys/gpu_util`
- `sys/gpu_mem_gb`
- `sys/tokens_per_second`

Enable `wandb.watch(model, log="all", log_freq=100)` to track gradient norms and weight histograms. Gradient norm explosion (>10) or collapse (<0.001) are early signs of instability.

---

## 7. Defining "Minimally Useful": The Core Question

This is the most important section of this document. Before training a single token, you need to define what success looks like. "Lower perplexity" is not a goal — it's a signal. The goal is capability.

### 7.1 The Problem with Common Definitions

- **"Passes HellaSwag"** — meaningless below 300M params (near chance)
- **"Better perplexity than GPT-2"** — incomparable across datasets
- **"Generates text"** — trivially achieved by a random bigram model
- **"Passes MMLU"** — impossible below ~500M without heavy data curation

### 7.2 The TinyStories Insight (Reframed)

The TinyStories paper showed that coherence is **domain-relative**: a 1M-param model trained on simple stories is MORE useful for story generation than a 125M-param model trained on messy internet text. This is the key insight for your project.

**Corollary:** "Minimally useful" should be defined relative to a specific domain and task, not absolute parameter count or benchmark percentile.

### 7.3 Proposed Definition: "Minimally Useful" for This Project

**Primary Target Capability:**

> A model is "minimally useful" if, given a partial prompt from its training domain (first sentence of a story, start of a Python function, beginning of a Q&A pair), it can complete the prompt in a way that a human would plausibly accept as a continuation — evaluated blind, with no knowledge of which completions are human vs. model-generated.

This is a **Turing-adjacent definition** scoped to a single domain. It does not require instruction following, factual accuracy, or general reasoning.

### 7.4 Measurable Success Bar (Your Scorecard)

Track ALL of these. The model is "minimally useful" when it satisfies criteria in at least 3 of the 4 groups:

**Group A — Training Signal (necessary but not sufficient)**
- [ ] Validation loss below 2.5 (perplexity < 12) on held-out domain text
- [ ] Validation loss consistently lower than training loss by end of run (no severe overfitting)

**Group B — Standard Benchmark (above-noise threshold)**
- [ ] PIQA zero-shot accuracy > 58% (vs. 50% random), OR
- [ ] ARC-easy zero-shot accuracy > 35% (vs. 25% random)
- At 10–50M parameters, clearing BOTH simultaneously is a stretch goal

**Group C — GPT-4 Judge (generation quality)**
- [ ] Mean grammar score ≥ 7.0 / 10 over 20 probe completions
- [ ] Mean consistency score ≥ 6.5 / 10 over 20 probe completions
- These thresholds are based on TinyStories paper methodology; ~5.0 is baseline for untrained models

**Group D — Human Qualitative Pass**
- [ ] 5-minute reading of 20 generations: fewer than 3 instances of token loops or severe semantic drift
- [ ] At least 10 of 20 completions are plausible continuations of the prompt

**Stretch Goal (for the 50M+ range):**
- [ ] HellaSwag > 32% (above chance + noise at 4-way task)
- [ ] ARC-easy > 45%
- [ ] Qualitative: a naive human reader, reading 10 completions, believes at least 6 were written by a person

### 7.5 What "Not Useful" Looks Like

Your model is NOT yet minimally useful if:
- It generates repetitive token loops (perplexity is low but outputs are degenerate)
- It scores below 55% on PIQA (near chance for a 2-way task)
- GPT-4 judge gives grammar < 5.0 (incoherent syntax)
- Human reading: >50% of completions are clearly machine-generated garbage

These are your "training failure" signals. When they appear, **stop the run, diagnose, adjust, restart** — the ETH Zurich method, applied to evaluation.

### 7.6 Why This Bar Is Right for Your Hardware

On an 8 GB RTX 3060 Ti:
- A 10M model can be fully trained (from scratch, 1B tokens) in ~2–4 hours
- A 50M model takes ~8–16 hours for 1B tokens
- A 150M model takes ~30–50 hours for 1B tokens (feasible for a weekend run)

With this hardware you can run 10–30 training experiments on 10–50M models before you'd complete a single 150M run. The ETH Zurich method says: **train many models, learn fast**. Therefore, define "minimally useful" at the 10–50M scale and iterate aggressively before scaling up.

A 50M model that achieves the scorecard above is **more valuable to your learning** than a 150M model trained once, imperfectly, with no iteration. The evaluation framework in this document is designed for models you can train in hours, not days.

---

## 8. BabyLM as External Reference

The [BabyLM Challenge (2024)](https://aclanthology.org/events/babylm-2024/) is the closest analog to your project: it challenges researchers to train competitive models on a strictly limited token budget (10M or 100M words), emphasizing data efficiency and architecture innovation over scale.

The BabyLM evaluation suite uses:
- **BLiMP** — Benchmark of Linguistic Minimal Pairs (syntactic acceptability)
- **(Super)GLUE** — Standard NLU benchmarks
- **MSGS** — Mixed Signals Generalization Set

At the 10M-word strict-small track, even state-of-the-art models struggle to break 70% on BLiMP. This is a useful sanity check: **data quantity matters enormously at tiny scale**. If you train on 100M tokens, expect capability equivalent to the lower range of BabyLM strict-small entries.

See: [BabyLM 2024 evaluation pipeline](https://github.com/babylm/evaluation-pipeline-2024), [Findings of BabyLM Challenge (2024)](https://arxiv.org/html/2504.08165v1).

---

## 9. The Evaluation Loop in Practice

Here is the complete evaluation cadence for the ETH Zurich fast-iteration method on your hardware:

```
Every training run (minutes to hours):
├── Plot train/val loss in W&B → is it learning? is it overfitting?
├── After 10k steps or end of run:
│   ├── Generate 10 probe completions → read them (2 min)
│   ├── Run lm_eval on piqa,arc_easy,winogrande (5–10 min on 3060 Ti)
│   └── If GPT API available: run GPT-4-mini judge on probe set (30 sec)
└── Log everything to W&B, add a note about what changed
```

Total evaluation cost per run: ~10–15 minutes of wall time. Total money cost (GPT API): ~$0.005 per evaluation. This is cheap enough to do on every run without friction.

**Automate it.** Write a `evaluate.py` script that takes a checkpoint path and runs all tiers, then outputs a structured JSON summary. Log the summary to W&B as a table. After 20 runs, you will have a learning curve across your entire experiment history — that is the compounding knowledge the ETH Zurich method produces.

---

## 10. Learn-by-Doing

### Experiment 1: Calibrate Your Loss Intuitions (15 min)

Load three models of increasing size from EleutherAI's Pythia suite and measure their validation perplexity on a fixed 5MB text file you care about (a book, a code corpus, whatever):

```bash
pip install transformers datasets lm_eval
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, math

models = ['EleutherAI/pythia-70m', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m']
text = open('my_corpus_sample.txt').read()[:50000]  # 50K chars

for m_name in models:
    tok = AutoTokenizer.from_pretrained(m_name)
    model = AutoModelForCausalLM.from_pretrained(m_name).cuda()
    ids = tok(text, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    print(f'{m_name}: loss={loss.item():.3f}, ppl={math.exp(loss.item()):.1f}')
"
```

**Learning goal:** Develop an intuition for what different perplexity values mean on your target domain. This is your calibration baseline for when you train your own models.

### Experiment 2: Run lm_eval on Pythia-70M (30 min)

Before you train anything, understand what the SOTA baseline at 70M parameters looks like on standard benchmarks:

```bash
lm_eval --model hf \
  --model_args pretrained=EleutherAI/pythia-70m \
  --tasks piqa,arc_easy,winogrande \
  --device cuda:0 \
  --batch_size auto:4 \
  --num_fewshot 0 \
  --output_path ./pythia_70m_eval.json
```

Compare to the numbers in Section 3.3. Then repeat for `pythia-160m`. Notice how PIQA moves from ~59% to ~63% — a meaningful jump. ARC-easy moves from ~38% to ~46%. HellaSwag moves from ~27% to ~30% — barely above noise. **Learning goal:** Internalize which benchmarks are meaningful at which scales.

### Experiment 3: GPT-4 Judge on Pre-Trained vs. Untrained Model (20 min)

Compare generation quality between a random-weight model and a pre-trained Pythia-70M using the judge framework in Section 4.2:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompts = ["Once upon a time", "The robot walked into the room and"]

for model_name in ["random", "EleutherAI/pythia-70m"]:
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    if model_name == "random":
        from transformers import GPTNeoXConfig
        model = AutoModelForCausalLM.from_config(GPTNeoXConfig()).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        out = model.generate(ids, max_new_tokens=100, do_sample=True, temperature=0.8)
        print(f"[{model_name}] {tok.decode(out[0], skip_special_tokens=True)}\n")
```

Then pass the outputs to the GPT-4-mini judge. **Learning goal:** Understand what the judge is measuring, and build intuition for what a "6/10 grammar" vs. "9/10 grammar" looks like. This calibrates your human reading instincts.

### Experiment 4: Build Your Evaluation Script (45 min)

Write `evaluate.py` that takes `--checkpoint_path`, runs all four evaluation tiers, and writes a structured JSON output. Wire it into your training loop to run automatically at the end of each experiment. Track the outputs across three training runs (varying only one hyperparameter, e.g., learning rate).

**Learning goal:** Evaluation should feel like flipping a switch, not a research exercise. Fast, automated, reproducible evaluation is the foundation of the ETH Zurich method. After this experiment, you will never evaluate a model manually again.

---

## 11. Key References

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/pdf/2305.07759) — Eldan & Li, 2023. Foundational paper for this project's methodology.
- [Cerebras-GPT: Open Compute-Optimal Language Models](https://arxiv.org/pdf/2304.03208) — Benchmark reference for 111M models.
- [Pythia: A Suite for Analyzing Large Language Models](https://arxiv.org/pdf/2304.01373) — EleutherAI's suite with 70M–12B models; excellent benchmark anchors.
- [SmolLM2: When Smol Goes Big](https://arxiv.org/pdf/2502.02737) — COLM 2025. State-of-the-art at 135M parameters; 42.1% HellaSwag from data curation alone.
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — The standard framework for standardized benchmark evaluation.
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/html/2402.00838v4) — Bits-per-byte methodology and Paloma evaluation framework.
- [BabyLM Challenge 2024](https://aclanthology.org/events/babylm-2024/) — The academic analog to this project; evaluation-pipeline-2024 for reference.
- [Pico: A Modular Framework for Hypothesis-Driven Small Language Model Research](https://arxiv.org/abs/2509.16413) — EMNLP 2025. A lightweight research framework worth studying.
- [LLM-as-a-Judge: From Generation to Judgment](https://arxiv.org/pdf/2411.16594) — Survey of LLM-judge methodology, limitations, and best practices.
- [Weights & Biases experiment tracking](https://wandb.ai/site/experiment-tracking/) — Essential tool for tracking fast-iteration experiments.
- [Lessons from the Trenches on Reproducible Evaluation](https://arxiv.org/pdf/2405.14782) — Practical guide to avoiding evaluation pitfalls.

---

## Appendix: Quick Reference — Benchmark Baselines at a Glance

| Model | Params | HellaSwag | ARC-easy | PIQA | WinoGrande |
|---|---|---|---|---|---|
| Random baseline | — | 25.0% | 25.0% | 50.0% | 50.0% |
| Cerebras-GPT | 111M | 26.8% | 38.0% | 59.4% | 48.8% |
| Pythia | 70M | 27.3% | ~36% | ~59% | 51.5% |
| Pythia | 160M | ~30% | ~46% | ~63% | ~52% |
| Pythia | 410M | ~40% | 51.2% | ~67% | ~55% |
| SmolLM2 | 135M | **42.1%** | **43.9%** | **68.4%** | 51.3% |
| GPT-2 | 124M | ~29.4% | — | — | — |
| **Your target** | **10–50M** | **>27% (stretch: 32%)** | **>35%** | **>58%** | **>51%** |

SmolLM2-135M dramatically outperforms Cerebras-111M at similar parameter counts, solely due to data quality. This is the most important number in this table.
