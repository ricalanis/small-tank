# Tokenization for Tiny Models

**TL;DR:** Tokenization is the interface between raw text and your model's embedding table — and for a tiny model (10M–150M params), your vocabulary size is not a free hyperparameter. It directly consumes a large slice of your parameter budget. A 30M-param model with a 50k-token vocab at 512 dims spends ~25M params on embeddings alone, leaving only 5M for actual intelligence. This document teaches you how tokenization works, how to pick and train the right tokenizer for your scale, and covers the frontier ideas (BLT, T-FREE) worth understanding even if you won't use them yet.

---

## 1. Why Tokenization Matters More at Tiny Scale

Every LLM converts a string of characters into a sequence of integer token IDs before anything else. The embedding layer (`nn.Embedding(vocab_size, d_model)`) turns those IDs into dense vectors. Its parameter count is:

```
embedding_params = vocab_size × d_model
```

For a 125M-param model (GPT-2 Medium territory), the math is manageable:
- GPT-2 Medium: 50,257 vocab × 1,024 dim = **51.5M params** in embeddings
- That's ~41% of total params, but 125M is large enough that ~74M go to actual transformer layers

For a **30M-param model**, the situation is critical:
- 50,257 vocab × 512 dim = **25.7M params** in embeddings → **86% of params!**
- 50,257 vocab × 256 dim = **12.9M params** → still **43% of params**
- **8,192 vocab × 256 dim = 2.1M params** → only **7% of params** ✓

The embedding table is also where **input and output projections live** in weight-tied models (standard practice). The output projection (sometimes called `lm_head`) maps from `d_model` back to `vocab_size` for next-token prediction. If you tie weights (reuse the embedding matrix transposed), you count it once. If not, double it. Most tiny models tie weights.

### Parameter Budget Table

| Vocab | d_model | Embedding params | % of 30M model | % of 125M model |
|-------|---------|-----------------|-----------------|-----------------|
| 50,257 | 768 | 38.6M | **>100%** (impossible) | 31% |
| 50,257 | 512 | 25.7M | **86%** | 21% |
| 32,000 | 512 | 16.4M | **55%** | 13% |
| 32,000 | 256 | 8.2M | **27%** | 7% |
| 8,192 | 512 | 4.2M | **14%** | 3% |
| 8,192 | 256 | 2.1M | **7%** | 2% |
| 4,096 | 256 | 1.1M | **3.5%** | 0.9% |

**Key insight:** A 30M-param model cannot responsibly use a 50k vocabulary unless you use a very small embedding dim (e.g., 256) *and* are prepared to sacrifice most of your parameter budget to tokenization rather than transformer depth/width. **For 30M, target vocab_size ≤ 16,000. For 125M, you can go up to 32,000.**

---

## 2. The Tokenization Landscape

### 2.1 Byte Pair Encoding (BPE)

**Origin:** Data compression algorithm adapted for NLP by [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909), popularized by GPT-2.

**How it works:**
1. Start with all individual characters (or bytes) as vocabulary
2. Count every adjacent pair of tokens in the corpus
3. Merge the most frequent pair into a new single token
4. Repeat until you hit `vocab_size`

The result: frequent words become single tokens ("the", "and"), uncommon words split into subwords ("un", "##happy"), rare strings fall back to individual characters or bytes.

**Variants:**
- **Character-level BPE** (classic): starts from Unicode code points. Can produce `<UNK>` tokens for unseen characters. Used in early models.
- **Byte-level BPE** (GPT-2, RoBERTa, GPT-NeoX, Falcon, Llama-3): starts from 256 raw byte values. *Guaranteed no `<UNK>` tokens ever.* Any text in any language is representable. Preferred for modern English-centric models.

**Key libraries:**
- [tiktoken](https://github.com/openai/tiktoken): OpenAI's Rust-based BPE library. Extremely fast (3–6× faster than alternatives). Pre-trained vocabs: `gpt2` (50,257 tokens), `cl100k_base` (100,277 tokens for GPT-4/text-embedding-ada-002), `o200k_base` (200k for GPT-4o). **Cannot train new vocabularies** — it only encodes against existing vocabs. Use it if reusing an OpenAI vocab.
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers): Rust-backed, trainable. Supports BPE, WordPiece, Unigram. **The recommended choice for training custom tokenizers from scratch.**

**GPT-2 tokenizer specifics:** vocab_size=50,257 (256 byte base + 50,000 merges + 1 special). The `cl100k_base` (GPT-4) doubled this to 100,277 for better multilingual coverage and longer effective context.

### 2.2 SentencePiece

[SentencePiece](https://github.com/google/sentencepiece) by Google is a *language-independent* tokenizer library supporting two algorithms:

- **SentencePiece BPE:** BPE on Unicode code points (not bytes). Falls back to UTF-8 byte encoding for rare characters. Used by Llama-2, Mistral, Gemma. Vocab typically 32,000.
- **SentencePiece Unigram:** Starts with a large candidate vocabulary, then prunes by fitting a unigram language model and removing tokens with lowest likelihood contribution. Tends to produce more linguistically natural segmentations and slightly lower fertility (fewer tokens per word). Used by T5, ALBERT, some multilingual models. Training is slower but often higher quality.

**SentencePiece vs HF BPE for tiny models:**
- SentencePiece requires pre-tokenization on spaces (or not, using the `add_dummy_prefix` flag). It treats the corpus as a raw byte stream by default.
- For English-only tiny models: both are fine. HF Tokenizers BPE is slightly simpler to script and saves in JSON format.
- For multilingual: SentencePiece Unigram generally outperforms BPE at low fertility (fewer tokens per sentence).

### 2.3 WordPiece

Used by BERT, DistilBERT. Similar to BPE but uses likelihood-based merge decisions instead of raw frequency. Not recommended for new projects — BPE and Unigram are better understood.

### 2.4 Character-level and Byte-level Tokenization

**Pure character-level:** Vocab = ~150–300 Unicode code points (or ~65k for full Unicode). Very long sequences (1 char = 1 token). Training is slow; context window consumed fast. Only viable for toy models or very constrained domains.

**Byte-level (256 base vocab, no merges):** Used by ByT5. Full coverage, zero unknown tokens, but sequences are ~3–4× longer than BPE. Transformers pay O(n²) in attention, so 4× longer sequences = 16× more attention compute at equal depth. Not practical for standard transformers at scale.

**Byte-level BPE** (best of both worlds): Start from 256 bytes, run BPE merges. Results in reasonable sequence lengths (~1.2–1.5× GPT-2 for typical English text) with zero `<UNK>`. This is what GPT-2, GPT-NeoX, and Falcon use. Recommended for new English-centric small models.

---

## 3. Training Your Own Tokenizer

When to train your own: always, unless you're intentionally reusing a pretrained model's tokenizer for compatibility.

A custom tokenizer trained on your target domain will:
1. Use its merges on *domain-relevant* subwords (code, scientific terms, etc.)
2. Achieve better compression (lower "fertility" = fewer tokens per character)
3. Let you choose vocab_size precisely to match your parameter budget

### 3.1 Code Recipe (HuggingFace Tokenizers)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Byte-level BPE (GPT-2 style) — recommended for tiny models
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]

trainer = BpeTrainer(
    vocab_size=8192,          # adjust to your param budget
    special_tokens=special_tokens,
    min_frequency=2,           # ignore pairs seen < 2 times
    show_progress=True,
)

# files: list of .txt paths (your training corpus)
files = ["data/train.txt"]
tokenizer.train(files, trainer)
tokenizer.save("tokenizer-8k.json")

# Wrap for HuggingFace compatibility
from tokenizers.processors import ByteLevel as ByteLevelProcessor
tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

# Load and verify
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer-8k.json")
print(tok.get_vocab_size())   # should print 8192
enc = tok.encode("Hello, world!")
print(enc.tokens)
```

**Training speed:** On a 500MB text corpus, HF Tokenizers trains a 8k-vocab BPE in ~10–30 seconds on a single CPU core. Even 32k vocab on 10GB trains in minutes — this is never the bottleneck.

### 3.2 Vocab Size Decision Matrix

For the ETH Zurich "train many small models fast" approach, use this rule of thumb:

| Model target | Param budget | Max embedding params | Recommended vocab | d_model |
|---|---|---|---|---|
| Toy (10M) | 10M | 1.5M | **4,096** | 256 or 384 |
| Small (30M) | 30M | 5M | **8,192** | 512 |
| Medium (125M) | 125M | 20M | **16,384–32,000** | 512–768 |

The formula to check yourself:
```
vocab_size × d_model ≤ (target_params × 0.15)   # embedding ≤ 15% of params
```

For a 30M model with d_model=512: `vocab_size ≤ (30M × 0.15) / 512 = 8,789`. Round to **8,192** (power of 2 preferred for GPU efficiency).

### 3.3 NeurIPS 2024: Scaling Laws with Vocabulary

A [NeurIPS 2024 paper](https://arxiv.org/abs/2407.13623) (Tao et al., Singapore-MIT) derived compute-optimal vocabulary scaling laws by training 33M–3B parameter models on up to 500B characters. Key findings:

- Optimal vocab size scales with compute budget following a **power law** — bigger models need bigger vocabularies
- Most models historically used vocabularies too small for their compute (Llama-2 at 32k was ~7× too small for a 70B model optimal)
- For **small models under 1B on limited compute**, 16k–32k is typically near-optimal for English
- Inference speed also matters: larger vocabs mean shorter sequences, which reduces KV-cache and attention cost

**Practical implication for 30M models:** Since our compute budget is small (single GPU, short runs), the optimal vocabulary is smaller than what larger models use. 8,192 is defensible, 16,384 is acceptable if d_model is ≥ 512.

---

## 4. Special Tokens and Chat Templates

### 4.1 Essential Special Tokens

Every tokenizer needs at minimum:

| Token | Purpose | Common symbol |
|-------|---------|---------------|
| BOS | Beginning of sequence | `<s>`, `<|bos|>`, `<|startoftext|>` |
| EOS | End of sequence | `</s>`, `<|eos|>`, `<|endoftext|>` |
| PAD | Batch padding (set `attention_mask=0`) | `<|pad|>` |
| UNK | Unknown (byte-level BPE doesn't need this) | `<unk>` |

For instruction/chat-tuned models, add:
- `<|im_start|>`, `<|im_end|>` (ChatML format)
- `<|user|>`, `<|assistant|>`, `<|system|>` (Phi-2 style)

### 4.2 Chat Templates

A chat template is a Jinja2 string stored in `tokenizer_config.json` that formats multi-turn conversations into a flat token sequence. The de facto standard for small open models is [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

HuggingFace `tokenizer_config.json` stores this as:
```json
{
  "chat_template": "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
}
```

**For pretraining a tiny base model:** You don't need a chat template. Just EOS between documents during pretraining. Add chat template only during supervised fine-tuning (SFT).

**Key rule:** The tokenizer used during training must be identical to the one used at inference. This includes special tokens, normalization, and pre-tokenization rules. Save `tokenizer.json` and `tokenizer_config.json` alongside model weights.

---

## 5. Frontier Ideas: Tokenizer-Free Approaches

### 5.1 Byte Latent Transformer (BLT)

Meta's [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) (December 2024) eliminates fixed tokenization entirely. Instead:
- Raw bytes are grouped into **variable-length patches** using entropy of the next byte as a signal
- Simple, small "local" encoder/decoder transformers convert between byte sequences and patch representations
- A large "global" transformer operates on patches — this is where most compute goes
- Patches are **longer when content is predictable** (saves compute), **shorter when content is complex/unpredictable** (spends compute wisely)

BLT scaled to 8B parameters and matched LLaMA-3 (a strong BPE baseline) on standard benchmarks while showing better robustness to character-level perturbations and orthographic tasks.

**For tiny models:** BLT adds architectural complexity (two extra local transformers) and is not yet well-studied at <500M scale. The overhead of local transformers may consume too much of a 30M param budget. **Not recommended for your first model — revisit once you have a working BPE baseline.**

### 5.2 T-FREE

[T-FREE](https://aclanthology.org/2024.emnlp-main.1217/) (Aleph Alpha + TU Darmstadt, EMNLP 2024) takes a different approach:
- Words are represented as **sparse superpositions of character trigram embeddings**
- "hello" → hash("hel") + hash("ell") + hash("llo") → sparse activation over a fixed-size embedding matrix
- No tokenizer training required. No vocabulary to choose.
- Achieves **>85% parameter reduction** in the embedding layer
- Naturally handles morphological similarity (similar words share trigrams)
- Strong cross-lingual transfer

**Practical use:** T-FREE's embedding matrix size is fixed (typically 65,536 slots for trigram hashes), with sparse activation. For small models, this could be advantageous — the embedding layer never grows with vocabulary. However, the research is new (2024) and there's no widely-used open implementation yet. Watch this space.

### 5.3 Other Directions

- **MegaByte** (2023, Meta): hierarchical byte-level model; less efficient than BLT
- **Mamba/SSM models on bytes**: Linear-time sequence models reduce the cost of long byte sequences; [Mamba on bytes](https://arxiv.org/abs/2401.13660) is feasible but still experimental
- **Cross-tokenizer distillation** (2025): [Recent work](https://arxiv.org/abs/2604.07466) enables distilling knowledge across models with different tokenizers via a shared byte-level interface

**Verdict for this project:** Use byte-level BPE now. Monitor BLT and T-FREE for your next iteration.

---

## 6. Reusing vs. Training a Tokenizer

### When to reuse an existing tokenizer:
- You want to use pretrained weights or distill from a large model (tokenizer must match)
- Your domain is standard English text — GPT-NeoX (50,277 tokens) is excellent quality
- You want to skip this step and focus on architecture experiments

### Good reuse candidates:

| Tokenizer | Vocab size | Algorithm | Notes |
|-----------|-----------|-----------|-------|
| GPT-2 | 50,257 | byte-level BPE | Classic, widely understood |
| GPT-NeoX-20B | 50,277 | byte-level BPE | Trained on The Pile; better for diverse text |
| Llama-2 | 32,000 | SentencePiece BPE | Good balance; used in TinyLlama |
| Llama-3 | 128,256 | byte-level BPE | Too large for tiny models |
| Phi-2/Phi-3 | 32,064 | byte-level BPE | Optimized for small models |

**For reuse at tiny scale:** The GPT-NeoX or Phi-2 tokenizer (32k–50k) will burn ~25% of params at d_model=512. This is acceptable if you choose a reused tokenizer deliberately and scale d_model accordingly, or if you truncate the vocabulary (keep only the top 8k–16k most frequent tokens, remap IDs). Vocab truncation is a valid technique: [Efficient Vocabulary Reduction for Small Language Models (COLING 2025)](https://aclanthology.org/2025.coling-industry.64.pdf).

### When to train your own:
- Domain-specific data (code, scientific, multilingual) — custom tokenizer achieves better compression
- You want full control of vocab_size for parameter budgeting
- You're building from scratch for learning (recommended in this project)

---

## 7. Concrete Recommendations for This Project

### For the 30M-param model:
- **Algorithm:** Byte-level BPE (HuggingFace Tokenizers library)
- **Vocab size:** **8,192** — leaves only 4.2M params in embeddings (14% of 30M), giving 25.8M params for transformer layers
- **d_model:** 512 (8,192 × 512 = 4.2M — acceptable)
- **Special tokens:** `<|endoftext|>` (BOS/EOS shared), `<|pad|>`
- **Training data for tokenizer:** Same corpus as model pretraining (or a representative 500MB–1GB sample)

### For the 125M-param model:
- **Algorithm:** Byte-level BPE
- **Vocab size:** **16,384** — 16,384 × 768 = 12.6M params (10% of 125M) — excellent
- Or push to **32,000** if d_model=512: 32,000 × 512 = 16.4M (13% of 125M) — still fine
- **Special tokens:** same as above, add chat tokens during SFT

### Embedding parameter budget check:
```python
# Run this before finalizing your architecture
def embedding_param_pct(vocab_size, d_model, total_params):
    emb_params = vocab_size * d_model
    pct = emb_params / total_params * 100
    print(f"Embedding: {emb_params/1e6:.1f}M / {total_params/1e6:.0f}M total = {pct:.1f}%")
    print(f"Remaining for transformer: {(total_params-emb_params)/1e6:.1f}M params")

embedding_param_pct(8192, 512, 30_000_000)
# Embedding: 4.2M / 30M total = 14.0%
# Remaining for transformer: 25.8M params

embedding_param_pct(16384, 768, 125_000_000)
# Embedding: 12.6M / 125M total = 10.1%
# Remaining for transformer: 112.4M params
```

### VRAM cost of the embedding table:
- 8,192 × 512 × 2 bytes (bfloat16) = **8.4 MB** — negligible
- 32,000 × 512 × 2 bytes = **32.8 MB** — still fine
- 100,277 × 1024 × 2 bytes = **205 MB** — starts to matter on 8GB VRAM

For an 8GB RTX 3060 Ti, embedding tables up to ~200MB are fine during training. The bottleneck is activations and optimizer states, not embeddings.

---

## 8. Learn-by-Doing

### Experiment 1: Vocabulary Parameter Budget Calculator (5 min)

Run this script to internalize the parameter budget implications before you code anything:

```python
# vocab_budget.py
import math

configs = [
    ("30M toy",  30_000_000,  256, [1024, 2048, 4096, 8192, 16384, 32000, 50257]),
    ("30M main", 30_000_000,  512, [1024, 2048, 4096, 8192, 16384, 32000, 50257]),
    ("125M",    125_000_000,  768, [8192, 16384, 32000, 50257, 100277]),
]

for label, total, d_model, vocabs in configs:
    print(f"\n=== {label} model, d_model={d_model} ===")
    print(f"{'Vocab':>8} | {'Emb (M)':>8} | {'% of total':>10} | {'Remaining (M)':>13}")
    print("-" * 50)
    for v in vocabs:
        emb = v * d_model
        pct = emb / total * 100
        rem = (total - emb) / 1e6
        marker = " *** OVER BUDGET" if emb > total * 0.3 else (" <- good" if emb < total * 0.15 else "")
        print(f"{v:>8,} | {emb/1e6:>8.1f} | {pct:>10.1f}% | {rem:>13.1f}M{marker}")
```

**Goal:** Viscerally understand that vocab size is a first-class architectural decision, not an afterthought.

### Experiment 2: Train Three Tokenizers, Measure Fertility (20 min)

Train BPE tokenizers at three vocab sizes on the same corpus, then compare "fertility" (avg tokens per word):

```python
# train_and_compare_tokenizers.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as BLD
import statistics

# Download a small corpus (TinyStories or WikiText-2)
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/shakespeare.txt

corpus_file = "data/shakespeare.txt"
test_text = open(corpus_file).read()[:10000]  # first 10k chars for comparison

results = {}
for vocab_size in [1024, 4096, 8192, 16384, 32000]:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = BLD()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        min_frequency=2,
    )
    tokenizer.train([corpus_file], trainer)
    
    # Measure fertility (tokens per 1000 chars)
    n_tokens = len(tokenizer.encode(test_text).ids)
    n_chars = len(test_text)
    fertility = n_tokens / (n_chars / 4)  # ~4 chars/word in English
    
    results[vocab_size] = {
        "tokens_per_1k_chars": n_tokens / n_chars * 1000,
        "fertility": fertility,
        "train_time_note": "measure with time.time()"
    }
    print(f"vocab={vocab_size:6d}: {n_tokens/n_chars*1000:.1f} tokens/1000 chars")

# Key insight: larger vocab = fewer tokens per text = faster training (shorter sequences)
# But larger embedding table = more params wasted on tokenization
```

**What you'll observe:** vocab_size=1024 might produce ~700 tokens/1000 chars; 8192 drops to ~250; 32000 to ~180. Every halving of tokens/char roughly halves training time per epoch (at the cost of more embedding params).

### Experiment 3: Compare Reused vs. Custom Tokenizer on Your Domain (30 min)

```python
# compare_tokenizers.py
import tiktoken

# Load GPT-4 tokenizer (cl100k_base)
gpt4_tok = tiktoken.get_encoding("cl100k_base")  # 100,277 vocab

# Your custom 8k tokenizer (from Experiment 2)
from tokenizers import Tokenizer
custom_tok = Tokenizer.from_file("tokenizer-8k.json")

test_sentences = [
    "The transformer architecture uses self-attention mechanisms.",
    "def forward(self, x): return self.layer(x)",
    "Backpropagation computes gradients using the chain rule.",
    "The loss decreased from 2.4 to 1.8 after 1000 steps.",
]

for sent in test_sentences:
    gpt4_n = len(gpt4_tok.encode(sent))
    custom_n = len(custom_tok.encode(sent).ids)
    print(f"GPT4={gpt4_n:3d} | Custom={custom_n:3d} | '{sent[:50]}'")
```

**What to look for:** In your domain (if it's code or technical English), a custom tokenizer will often use fewer tokens. But if your corpus is too small (<50MB), a custom 8k tokenizer may actually be _worse_ than a pretrained 50k tokenizer trained on billions of tokens.

### Experiment 4: Verify Chat Template Round-Trip (15 min)

Before spending hours training, verify your tokenizer correctly encodes and decodes chat turns:

```python
# verify_chat_template.py
import json

# Build a minimal tokenizer_config.json with ChatML template
chat_template = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Add special chat tokens to your tokenizer
from tokenizers import Tokenizer, AddedToken
tok = Tokenizer.from_file("tokenizer-8k.json")
tok.add_special_tokens([
    AddedToken("<|im_start|>", special=True),
    AddedToken("<|im_end|>", special=True),
])

# Simulate a chat turn
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]

# Manual template rendering (for verification)
formatted = ""
for msg in messages:
    formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
formatted += "<|im_start|>assistant\n"

encoded = tok.encode(formatted)
decoded = tok.decode(encoded.ids)
assert formatted == decoded, "Round-trip failed!"
print("Chat template round-trip: OK")
print(f"Token count: {len(encoded.ids)}")
print("Token IDs of special tokens:", {
    "<|im_start|>": tok.token_to_id("<|im_start|>"),
    "<|im_end|>": tok.token_to_id("<|im_end|>"),
})
```

**Why this matters:** Many people spend days debugging why their SFT model generates garbage — it's almost always because the tokenizer during SFT didn't match pretraining, or the chat template wasn't applied correctly during both training and inference.

---

## 9. Reference Summary

| Question | Answer for this project |
|----------|------------------------|
| Algorithm | Byte-level BPE |
| Library | HuggingFace Tokenizers (`pip install tokenizers`) |
| Vocab for 30M model | **8,192** |
| Vocab for 125M model | **16,384** (d_model=768) or **32,000** (d_model=512) |
| Embedding % budget | Keep under 15% of total params |
| Special tokens minimum | `<\|endoftext\|>` (BOS+EOS), `<\|pad\|>` |
| Tiktoken for training? | No — use for inference only if reusing OAI vocab |
| BLT / T-FREE now? | No — interesting frontier, not production-ready at tiny scale |
| Weight-tied lm_head? | Yes — halves effective embedding cost |
| Chat template format | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) — add only at SFT stage |

---

## References

- [Byte Pair Encoding for Neural Machine Translation (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)
- [Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies (NeurIPS 2024)](https://arxiv.org/abs/2407.13623)
- [Byte Latent Transformer: Patches Scale Better Than Tokens (Meta AI, Dec 2024)](https://arxiv.org/abs/2412.09871)
- [T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1217/)
- [Efficient Vocabulary Reduction for Small Language Models (COLING 2025)](https://aclanthology.org/2025.coling-industry.64.pdf)
- [BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training (2024)](https://arxiv.org/abs/2409.04599)
- [HuggingFace Tokenizers Documentation](https://huggingface.co/docs/tokenizers/en/quicktour)
- [tiktoken (OpenAI)](https://github.com/openai/tiktoken)
- [SentencePiece (Google)](https://github.com/google/sentencepiece)
- [Implementing A BPE Tokenizer From Scratch (Raschka, 2025)](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)
- [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://s10251.pcdn.co/pdf/2022-black-gptneox-20b.pdf)
- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface (2025)](https://arxiv.org/abs/2604.07466)
