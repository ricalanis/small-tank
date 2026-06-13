"""
Data pipeline for TinyStories (research/04-datasets.md).

  python -m src.data prepare --train-stories 300000 --val-stories 3000 --vocab 4096

Steps (idempotent — skips work already done):
  1. stream roneneldan/TinyStories from the HF hub, write data/train.txt + data/val.txt
  2. train a byte-level BPE tokenizer (default vocab 4096, per DECISIONS.md D3 for the
     5M micro-proxy) on the train text; save to data/tokenizer.json
  3. tokenize both splits to uint16 binary shards data/train.bin + data/val.bin

The .bin files are flat uint16 token streams (vocab < 65536), memmapped by the trainer.
Stories are joined by the <|endoftext|> token so the model learns document boundaries.
"""
import argparse
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EOT = "<|endoftext|>"


def _paths():
    return {k: os.path.join(DATA, v) for k, v in dict(
        train_txt="train.txt", val_txt="val.txt", tok="tokenizer.json",
        train_bin="train.bin", val_bin="val.bin").items()}


def stream_to_text(n_train, n_val):
    p = _paths()
    if os.path.exists(p["train_txt"]) and os.path.exists(p["val_txt"]):
        print(f"[data] text splits exist, skipping download")
        return
    from datasets import load_dataset
    os.makedirs(DATA, exist_ok=True)
    print(f"[data] streaming TinyStories: {n_train} train + {n_val} val stories")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    it = iter(ds)
    with open(p["val_txt"], "w") as f:
        for _ in range(n_val):
            f.write(next(it)["text"].strip().replace("\n", " ") + "\n")
    with open(p["train_txt"], "w") as f:
        for i in range(n_train):
            f.write(next(it)["text"].strip().replace("\n", " ") + "\n")
            if (i + 1) % 50000 == 0:
                print(f"[data]   wrote {i+1} train stories")
    print("[data] text splits written")


def train_tokenizer(vocab):
    p = _paths()
    if os.path.exists(p["tok"]):
        print("[data] tokenizer exists, skipping")
        return
    from tokenizers import ByteLevelBPETokenizer
    print(f"[data] training byte-level BPE tokenizer, vocab={vocab}")
    tok = ByteLevelBPETokenizer()
    tok.train([p["train_txt"]], vocab_size=vocab, min_frequency=2, special_tokens=[EOT])
    tok.save(p["tok"])
    # mandatory round-trip check before spending GPU hours (research/03)
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(p["tok"])
    s = "Once upon a time, a small fox found a red ball."
    assert t.decode(t.encode(s).ids) == s, "tokenizer round-trip FAILED"
    print(f"[data] tokenizer ok (round-trip verified), real vocab={t.get_vocab_size()}")


def tokenize_split(txt_path, bin_path, tok):
    if os.path.exists(bin_path):
        print(f"[data] {os.path.basename(bin_path)} exists, skipping")
        return
    eot_id = tok.token_to_id(EOT)
    ids = []
    with open(txt_path) as f:
        for line in f:
            ids.extend(tok.encode(line.strip()).ids)
            ids.append(eot_id)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"[data] wrote {os.path.basename(bin_path)}: {len(arr):,} tokens")


def prepare(n_train, n_val, vocab):
    p = _paths()
    stream_to_text(n_train, n_val)
    train_tokenizer(vocab)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(p["tok"])
    tokenize_split(p["train_txt"], p["train_bin"], tok)
    tokenize_split(p["val_txt"], p["val_bin"], tok)
    print("[data] done.")


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(_paths()["tok"])


def get_batch(split, batch_size, seq_len, device):
    """Sample a random contiguous batch from the memmapped .bin shard."""
    path = _paths()["train_bin" if split == "train" else "val_bin"]
    data = np.memmap(path, dtype=np.uint16, mode="r")
    import torch
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--train-stories", type=int, default=300000)
    pp.add_argument("--val-stories", type=int, default=3000)
    pp.add_argument("--vocab", type=int, default=4096)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.train_stories, a.val_stories, a.vocab)
