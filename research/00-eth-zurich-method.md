# The ETH Zurich Method & Fast-Iteration Research Philosophy

> **TL;DR** The "ETH Zurich method" is a single-GPU, massively-parallel training philosophy that turns a consumer workstation into a rapid-fire research lab. The core bet is simple: *the more models you train, the faster you learn*. Rather than waiting days for one big run to finish, you design experiments that complete in minutes, run dozens per day, and let accumulated empirical evidence build your intuition. This document explains where the method comes from, why it works, how the ML speedrunning culture extends it to language models, and how to apply it — including on your RTX 3060 Ti — to become a genuinely world-class researcher.

---

## 1. Origin: ETH Zurich + NVIDIA, 2021

The term "ETH Zurich method" comes from a landmark 2021 paper: **[Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)** (Rudin et al., ETH Zurich & NVIDIA). The paper's headline result was training a legged-robot locomotion policy to walk on flat terrain in **under 4 minutes**, and on rough terrain in **20 minutes** — speedups of multiple orders of magnitude over prior work. The same tasks had previously taken hours or days.

The mechanism was not algorithmic cleverness alone. It was **concentration of compute on a single workstation GPU via massive parallelism**: running thousands of virtual robots simultaneously in GPU-accelerated simulation (NVIDIA Isaac Gym), so a single GPU could generate millions of training steps per second. Instead of distributing across a cluster, the researchers packed everything onto one card and ran it very fast.

The three principles that made it work:

1. **Parallelism beats distribution.** A cluster introduces communication overhead, sync delays, and engineering complexity. Thousands of environments on one GPU share memory bandwidth without round-trips.
2. **Wall-clock time, not compute, is the research bottleneck.** Researchers don't wait for FLOPs — they wait for *answers*. Faster wall-clock means faster iteration.
3. **Fast iteration compounds.** If each experiment takes 4 minutes instead of 4 hours, you can run 60× more experiments in the same working day. That's not a linear gain in productivity — it changes what kinds of questions you can ask.

This paper was circulated widely on X (then Twitter) and became a touchstone in the RL and robotics communities. Researchers like **@yacineMTB** (Yacine Mahdid, robotics/RL engineer, ex-X/Stripe) explicitly cite "the ETH Zurich route" when recommending single-GPU, sub-minute training for anyone learning AI research. In a May 2026 post, he wrote: *"train models on a single GPU that finish in under a minute (Pufferlib is example) so you can run many experiments and learn faster"* ([X post, May 26 2026](https://x.com/yacineMTB/status/2059270455010169128)).

---

## 2. PufferLib: The Philosophy as a Library

[@jsuarez5341 (Joseph Suarez)](https://github.com/jsuarez5341), an MIT PhD (Phillip Isola's lab) and creator of Neural MMO, took the ETH Zurich insight and built it into **[PufferLib](https://github.com/pufferai/pufferlib)**: an open-source RL library designed around wall-clock speed as the primary metric.

PufferLib's stated numbers:
- **300k–1.2M steps/second** on a single GPU, depending on environment and policy
- "Train tiny, super-human models in **seconds**"
- 11 first-party environments (~20k lines of pure C) each exceeding **1M steps/second** on a single CPU core
- PufferLib 2.0 (2025, [RLC Best Paper](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_151.pdf)): 1M steps/second with broader environment compatibility

Suarez's research philosophy, articulated in his "Ultra Opinionated Guide to Reinforcement Learning" and project writeups:

> *"You don't need to be clever about your methodology when you can run 1000× the experiments."*

The implication: at sufficient iteration speed, brute-force empiricism becomes a valid research strategy. You can sweep hyperparameters exhaustively, test every ablation, and re-run old work with better algorithms — all in the time a cluster job used to take just to start.

His goal for 2025-2026: *"run hundreds of thousands of comprehensive experiments covering hyperparameter sensitivity, algorithmic choices, and architecture design"* — because the infrastructure now makes it possible.

---

## 3. The Speedrunning Culture: NanoGPT and Language Models

The ETH Zurich insight migrated from RL to language models through **Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)** and the **[modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)** by **Keller Jordan (@kellerjordan0)**.

### The Speedrun

The challenge: train a GPT-2-scale model (124M parameters) to a fixed validation loss (3.28 cross-entropy on FineWeb) in minimal wall-clock time on 8× NVIDIA H100 GPUs. The community tracks world records on a public leaderboard.

Timeline of records:
| Date | Time | Milestone |
|------|------|-----------|
| Early 2024 | ~45 min | Karpathy's llm.c baseline |
| Jan 2025 | ~3 min | Community modded-nanogpt |
| March 2026 | ~86 sec | Via partitioned hyperconnections + late-layer context vectors ([Larry Dial, @classiclarryd](https://x.com/classiclarryd/status/2030465730718908884)) |
| Current (Jun 2026) | ~80–90 sec | Ongoing; active leaderboard |

Over 83+ records contributed by the community, training time fell by **~34×** — from 45 minutes to under 90 seconds. Techniques discovered or validated through the speedrun: Muon optimizer, RoPE embeddings, QK-normalization, ReLU² activations, FlashAttention 3, value embeddings, multi-token prediction, FP8 training, document-aligned batching, dynamic attention window sizing.

### Why Speedrunning Produces Real Research

The LessWrong post [How the NanoGPT Speedrun WR dropped 20% in 3 months](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months) identified three lessons from participating:

1. **Feedback loops are everything.** Without peers in AI research, the speedrun's measurable timing metric became *the* feedback mechanism. Research environments without tight feedback loops massively underperform.
2. **Working backwards from data > reading papers.** Analyzing specific model behaviors (e.g., perfect predictions on rare sequences) revealed architectural insights impossible through abstract reasoning.
3. **Gradient descent deserves respect.** Giving models more context and choices consistently outperformed human-designed constraints. Trust the optimizer.

The speedrun also validated the research transfer hypothesis: optimizations discovered at 124M parameters generalized — scaled-up versions achieved GPT-2 Medium performance at **2.5× lower cost** than baselines.

Professor Ben Recht captured the deeper mechanism: *"inventing games and gloading to goad others to play"* — competitive benchmarks with explicit rules create incentive structures for collaborative innovation faster than any individual lab can sustain alone.

---

## 4. The Core Principle: Why Many Small Models Beat One Big Run

### The Information Rate Argument

Every training run answers a question. The question might be: "Does adding RoPE embeddings help my 50M-param model?" or "Is dropout hurting or helping at step 10k?" The *information content* of the answer is roughly the same whether the run takes 5 minutes or 5 hours. But a 5-hour run gives you 1 answer per day; a 5-minute run gives you 12 per hour. **Your learning rate as a researcher is bounded by your experimental throughput.**

More precisely: if you're doing science, you're running a sequence of hypothesis tests. Each test has an expected information gain `I`. Your research velocity is approximately:

```
research_velocity ≈ I × (experiments_per_day)
```

Maximizing `experiments_per_day` — by shrinking run time — is often more valuable than maximizing `I` per experiment (by running larger, "more definitive" tests).

### What Small Scale Reveals

A common misconception is that small-scale experiments don't tell you anything about large-scale behavior. The scaling laws literature (Kaplan et al. 2020, Hoffmann et al. 2022 "Chinchilla") shows the opposite: **loss curves follow predictable power laws across scales**. A technique that helps a 10M-parameter model almost always helps a 1B-parameter model. The effect size may differ, but the direction rarely reverses.

Practical implication for your project: if you run 50 experiments at 10M parameters and find the best architecture + training recipe, then train one 150M model with that recipe, you're likely to do better than someone who trained three 150M models blindly.

### The Compounding Intuition Argument

There's a second effect beyond information rate: **intuition compounding**. Watching 50 training curves builds pattern recognition that no paper can give you. You start seeing when a model is "healthy" vs. "stuck" at a glance — the shape of loss curves, when val loss diverges from train loss, when gradient norms spike. This intuition only comes from repetition. Fast iteration accelerates the development of *taste*.

---

## 5. Vivek (@itsreallyvivek): Research Taste as a Muscle

**@itsreallyvivek** (Vivek, research fellow at Anthropic/MATS program) has written and posted extensively on *how to be good at research* in ML/AI, with a focus on developing the qualitative judgment that separates great researchers from competent ones.

His key frames (from June 2026 [article post](https://x.com/itsreallyvivek/status/2065479789084033133) and [May 28 thread](https://x.com/itsreallyvivek/status/2059988004027117727) summarizing a conversation with Neel Nanda at DeepMind):

**"Taste is a muscle, not a gift."** Research taste — the ability to pick promising directions, notice when something is off, ask the right questions — is a trained skill that requires years of *low-stakes, messy projects*. You can't read your way to taste. You have to build things and fail at them, repeatedly, cheaply.

**Getting into frontier labs requires two things:**
1. A proven research track record — actual output that pushes boundaries (even small contributions count)
2. "Trench engineering" skills in large codebases

**Practical habits he recommends:**
- Read one research paper daily around a specific curiosity-driven problem, with structured reflection questions: *What assumptions does this make? What does this make possible? Where does it break? What ideas are transferable?*
- Use AI as a "brutal roaster" for your own ideas and writeups before sharing with humans
- Cold-email junior researchers (not just senior people) for mentorship
- Prioritize understanding architecture over outsourcing code

The connection to the ETH Zurich method: taste grows fastest when you're running experiments constantly, because each failure mode you observe teaches you something reading can't. The fast-iteration loop is the *taste accelerator*.

---

## 6. Karpathy's Recipe: The Canonical Framework

Before the speedrunning culture, **Andrej Karpathy** articulated the canonical framework for research-quality neural network training in his blog post [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (2019, still essential reading). The post predates the ETH Zurich paper but encodes the same philosophy:

**Six-stage recipe:**

1. **Inspect your data obsessively** before writing any model code. Spend hours in the dataloader.
2. **Build a tiny, trusted baseline** — a simple model that you understand completely. Verify loss at initialization (it should match theory: `-log(1/vocab_size)` for cross-entropy on random predictions).
3. **Overfit a small batch intentionally** before adding regularization. Proves the architecture can fit data at all.
4. **Add complexity one component at a time.** Each change is a hypothesis; test it in isolation.
5. **Run short proxy experiments** to validate ideas cheaply before committing compute.
6. **Monitor everything** — loss curves, gradient norms, learning rate schedules, activation statistics.

His key insight about iteration speed: *"Neural nets fail silently."* Bugs don't throw errors — they produce subtly wrong results that only appear as mysteriously bad loss curves. The only way to catch them is to run small, fast experiments where you can inspect everything. Long runs obscure bugs.

**Nicholas Carlini** (Anthropic) extends this further in [Rapid Iteration in Machine Learning Research](https://nicholas.carlini.com/writing/2022/rapid-iteration-machine-learning-research.html) (2022): eliminate *any* friction between idea and test. He built tools to snapshot Python state so experiments could restart from mid-run without re-loading data. His heuristic: *"any time it takes longer than ~a second for a script to start giving useful output, my ability to be productive drops off."*

---

## 7. Structuring a Tight Research Loop

Here is the practical experimental loop that emerges from combining the ETH Zurich method, Karpathy's recipe, and the speedrunning culture:

### The Loop

```
HYPOTHESIS → PROXY EXPERIMENT → VERDICT → LOG → NEXT HYPOTHESIS
```

Each step in detail:

**1. Write a falsifiable hypothesis before running anything.**
Bad: "let me try adding dropout and see what happens."
Good: "Adding 0.1 dropout after each attention layer will reduce val loss from 2.84 to below 2.80 within 5k steps, as seen in Karpathy's recipe and the GPT-2 paper."

A written hypothesis forces clarity and makes your log meaningful. You're not just recording results — you're recording *whether you were right*.

**2. Design a proxy experiment that runs in <10 minutes.**
On your RTX 3060 Ti (8GB VRAM):
- A 10M-parameter model trains ~33 minutes for a full Chinchilla-optimal run (200M tokens). That's too long for a proxy.
- A 5M-parameter model with 50M tokens trains in **~5–8 minutes** — ideal for hypothesis testing.
- A 2M-parameter model with 20M tokens trains in **~1–2 minutes** — for ultra-fast sanity checks.

Keep a "proxy config" in your codebase — a fixed small model + small dataset that you always run first. If the proxy doesn't validate the hypothesis, don't scale up.

**3. Make exactly one change per experiment.**
Confounding kills research. The speedrun community enforces this with Git: each record is a single PR with a single change. Your private experiments should do the same.

**4. Record results immediately.**
Your research journal entry should take 2 minutes to write. Template:

```
## Experiment: [date] [short name]
Hypothesis: [one sentence]
Change: [exactly what you changed]
Proxy result: val_loss X.XX at step N (was X.XX, Δ = +/-X.XX)
Full result: [if you ran it]
Verdict: CONFIRMED / REFUTED / INCONCLUSIVE
Intuition update: [what this changes about my mental model]
Next: [one follow-up experiment]
```

**5. Kill failures fast, scale successes carefully.**
If a proxy experiment doesn't show improvement, discard the idea immediately and move on. Don't rationalize with "maybe at larger scale it would help." If it shows improvement, *then* scale to your target model size.

### Tracking Tools

- **[Weights & Biases](https://wandb.ai)** (free tier): Log every run. The curve shape is as important as the final loss number.
- **Git branches**: One branch per experiment family. Merge only validated wins.
- **A flat research log** (e.g., `research/log.md`): Append only. One entry per meaningful experiment. This becomes your most valuable artifact over time — the institutional memory that no one else has.

---

## 8. Ablations: The Scientific Engine

An **ablation study** systematically removes or replaces one component to isolate its contribution. It is the core scientific method of deep learning research.

### How to Structure Ablations

Say you've built a 25M-parameter model that achieves val_loss 2.71 and you want to understand *why*. A proper ablation table looks like:

| Configuration | Val Loss | Δ vs baseline |
|--------------|----------|---------------|
| Full model (baseline) | 2.71 | — |
| — RoPE → learned pos embed | 2.84 | +0.13 |
| — ReLU² → GELU | 2.74 | +0.03 |
| — Muon → AdamW | 2.78 | +0.07 |
| — QK-norm | 2.77 | +0.06 |

This table tells you: RoPE and QK-norm are load-bearing; ReLU² barely matters for this model; Muon gives meaningful gains. You now know which components to carry forward and which to deprioritize.

**Fast ablation discipline on a 3060 Ti:**
- Run each row at 5M params first. The relative ordering almost always holds at 25M or 100M.
- 5 rows × 8 min = 40 minutes for a full ablation table. In one morning you can do 6 ablation tables. That's 30 components tested.

---

## 9. Reading vs. Building: The Right Balance

The fast-iteration philosophy does not mean ignoring papers. It means using papers differently.

**Bad pattern (common among beginners):** Read 10 papers, then try to implement the most sophisticated one from scratch. Spend 3 weeks debugging. Train one model. Learn almost nothing about *why* it works.

**Good pattern (ETH Zurich / speedrunning culture):**
1. Read papers to generate *hypotheses*, not to find algorithms to copy.
2. Build the simplest possible version of an idea. Test it in 10 minutes.
3. If it works, read the paper more carefully to understand *why*. If it doesn't, move on.
4. Let your experimental results guide *what to read next*.

@itsreallyvivek's [daily paper reflection practice](https://x.com/itsreallyvivek/status/2065376612422701443) frames reading as producing a research question, not a solution: *"What does this paper make possible? Where does it break?"* That question then drives an experiment.

The speedrun community exemplifies this: many of the 83+ world-record contributions drew from ideas in recent papers, but each one was validated empirically in the tight loop, not trusted on the basis of the paper's authority.

---

## 10. VRAM Math: Your Concrete Iteration Budget on the RTX 3060 Ti

The RTX 3060 Ti has **8 GB GDDR6 VRAM** and ~200 GB/s memory bandwidth. Here's the practical math for your machine:

### Memory Footprint of a Transformer

For a model with `P` parameters in FP32:
- **Parameters**: `P × 4` bytes
- **Gradients**: `P × 4` bytes
- **Adam optimizer state**: `P × 8` bytes (2 momentum terms)
- **Total (Adam, FP32)**: `P × 16` bytes

For P = 10M: 160 MB. Negligible.
For P = 100M: 1.6 GB. Leaves 6.4 GB for activations.
For P = 500M: 8 GB — already at your VRAM ceiling without activations.

**With bf16 training (mixed precision):**
- Parameters + activations in bf16 (2 bytes), optimizer state in fp32 (8 bytes)
- Effective per-parameter cost: ~10–12 bytes
- For 100M params: ~1.0–1.2 GB for model + optimizer, leaving ~6.8 GB for activations and batch

**Practical targets for your hardware:**

| Model Size | Params | VRAM (est.) | Tokens/sec (est.) | Time for 500M tokens |
|-----------|--------|-------------|-------------------|---------------------|
| Nano | 5M | ~0.5 GB | ~80k–120k | ~1.2–1.7 hrs |
| Small | 25M | ~1.5 GB | ~40k–60k | ~2.3–3.5 hrs |
| Medium | 100M | ~3–4 GB | ~15k–25k | ~6–9 hrs |
| Target ceiling | 150M | ~5–6 GB | ~10k–15k | ~9–14 hrs |

These numbers assume batch size tuned for throughput, sequence length 512, bf16, `torch.compile()`, and no gradient checkpointing. Actual numbers will vary — benchmark your own setup.

For **proxy experiments** (50M tokens, 5M params): ~5–7 minutes. Perfect for the fast-iteration loop.

**Throughput reference:** An RTX 3060 Ti has ~136 TFLOPS FP16 theoretical peak. Real utilization for small transformers is typically 40–60% due to memory bandwidth limits. Compute-limited scaling begins to matter above ~100M params.

---

## 11. The Researcher's Journal: Your Most Valuable Artifact

The single most differentiating habit among elite ML researchers is maintaining a detailed, honest research journal. Not a sanitized lab report — a running log of what you tried, why you tried it, what happened, and what you now think.

Why it matters:
- **You will forget.** In 3 months you won't remember why you made a design choice. The journal is the institutional memory.
- **Patterns emerge.** After 100 experiments, you start seeing which hypotheses are consistently right vs. consistently wrong. This is how you build taste.
- **It becomes a preprint.** The best ML papers are cleaned-up experiment journals. If you journal well, writing papers becomes editing, not starting from scratch.

### Minimal Journal Format

Keep `research/log.md` in your repo. One section per working session, one subsection per experiment. Use the template from section 7. After 2 weeks, review: what patterns do you see? What hypotheses have been consistently wrong? Update your priors.

---

## 12. Connecting to World-Class Research

The ETH Zurich method is not just a productivity hack — it's a theory of how expertise compounds:

**Speed → Volume → Pattern recognition → Taste → Better hypotheses → Better experiments → More discoveries**

The researchers who become world-class do not have more compute or more time. They have tighter loops and better calibrated intuitions. Keller Jordan's 83-record speedrun history is a public demonstration of this compounding — each record built intuition about the next opportunity, faster than any planning process could.

For you, with a 3060 Ti, 30 GB RAM, and the Python ecosystem available today, the gap between you and a well-funded lab is much smaller than it appears. The lab has more H100s. You have faster feedback on small-scale ideas, lower stakes for bold experiments, and — if you journal well — a compounding learning rate that a large team cannot match at the individual level.

The formula is: **short runs + honest logs + relentless hypothesis generation**.

---

## Learn-by-Doing

These four experiments are designed to be run in the first week. Each takes <30 minutes and teaches something irreplaceable.

### Experiment 1: Time Your Baseline (30 min)

**Goal:** Establish your personal throughput baseline and get comfortable with the measurement loop.

1. Install PyTorch nightly + bitsandbytes: `pip install torch --pre` (check [pytorch.org](https://pytorch.org) for current nightly install).
2. Clone nanoGPT: `git clone https://github.com/karpathy/nanoGPT && cd nanoGPT`.
3. Prepare the Shakespeare dataset: `python data/shakespeare_char/prepare.py`.
4. Edit `train.py`: set `n_layer=6, n_head=6, n_embd=384` (~10M params), `max_iters=1000`, `eval_interval=100`.
5. Run: `time python train.py`
6. Record: wall clock time, tokens/sec, final val_loss.
7. Run again with `torch.compile()` enabled. Record the speedup.

**What you learn:** Your real throughput number. The gap between theoretical FLOPS and actual tokens/sec. How much `torch.compile()` helps on your specific hardware. These numbers will calibrate every future estimate.

---

### Experiment 2: Your First Ablation Table (45 min)

**Goal:** Practice the hypothesis → single change → verdict loop across 4 runs.

Using your 10M-param Shakespeare config (5-minute runs), test these four hypotheses:

| Hypothesis | Change |
|-----------|--------|
| ReLU² > GELU at small scale | Replace activation in `new_gelu` |
| Rotary embeddings > learned | Add a minimal RoPE implementation |
| Larger LR helps at this size | Multiply `learning_rate` by 3× |
| Weight decay matters even at 10M | Set `weight_decay=0` vs default |

For each: write your prediction *before* running. Record actual result. Compare.

**What you learn:** How often your intuitions are right (hint: probably ~50% for novel combinations). The discipline of prediction before measurement. How much variance exists across 5-minute runs (important for knowing when differences are meaningful).

---

### Experiment 3: The Scaling Law Mini-Study (2–3 hours)

**Goal:** Empirically verify that scaling laws hold at your scale, so you can trust proxy experiments.

Train the same architecture at 4 sizes (2M, 5M, 10M, 25M params), holding tokens-per-parameter constant (Chinchilla: ~20 tokens/param). Record final val_loss for each. Plot log(params) vs log(val_loss).

If the relationship is approximately linear on the log-log plot, you have validated that small-scale experiments predict larger-scale behavior on your setup. Now you can trust your 5M-param proxy runs as predictors of your eventual 100M model.

**What you learn:** Whether your training code is healthy (unhealthy code produces non-monotonic scaling). Intuition about how much improvement comes "for free" from scale vs. from architecture. How to read a scaling curve.

---

### Experiment 4: Research Journal Sprint (ongoing)

**Goal:** Build the journaling habit before it feels burdensome.

For the next 7 days: after every experiment, immediately write a 5-line journal entry using the template from Section 7. On day 7, review all entries. Answer:
- Which hypotheses were correct?
- What pattern do you notice about your wrong predictions?
- What is one thing you now believe that you didn't believe on day 1?

**What you learn:** Your specific failure modes as a researcher (overconfidence in architectural changes? underconfidence in optimizer choices?). How fast your intuition improves with structured reflection. What a real research rhythm feels like vs. aimless tinkering.

---

## Key References

- [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978) — Rudin et al., ETH Zurich + NVIDIA (2021). The origin paper.
- [PufferLib 2.0: RL at 1M steps/s](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_151.pdf) — Suarez, 2025. The PufferLib paper.
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — Keller Jordan. The GPT-2 speedrun repo with full record history.
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) — Karpathy (2019). Still the canonical framework.
- [Rapid Iteration in Machine Learning Research](https://nicholas.carlini.com/writing/2022/rapid-iteration-machine-learning-research.html) — Carlini (2022). Eliminating friction.
- [How the NanoGPT Speedrun WR Dropped 20% in 3 Months](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months) — LessWrong (2025). Lessons from the speedrun community.
- [@yacineMTB on X](https://x.com/yacinemtb) — Fast-iteration RL and robotics; "ETH Zurich route" advocate.
- [@itsreallyvivek on X](https://x.com/itsreallyvivek) — Research taste and career development in ML.
- [@jsuarez5341 on X](https://x.com/jsuarez5341) — PufferLib, "train many experiments fast."
- [Super Tiny Language Models paper](https://arxiv.org/html/2405.14159v1) — Training 10M–100M models on consumer hardware.
- [Tyler Romero's NanoGPT Speedrun Worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/) — A practitioner's account of the fast-iteration speedrun loop.
