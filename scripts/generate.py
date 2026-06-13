"""
Sample from a trained checkpoint (research/06-evaluation.md: the human reading pass).

  python scripts/generate.py --ckpt research/checkpoints/5m.pt --prompt "Once upon a time"
  python scripts/generate.py --ckpt research/checkpoints/5m.pt --n 5 --tokens 200
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_tokenizer
from src.model import ModelConfig, TinyLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n", type=int, default=3, help="number of samples")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = TinyLM(ModelConfig.from_dict(cfg)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = load_tokenizer()

    print(f"# ckpt {a.ckpt} | step {ck.get('step')} | val_loss {ck.get('val_loss'):.3f}")
    print(f"# prompt: {a.prompt!r} | temp {a.temperature} | top_k {a.top_k}\n")
    ids = tok.encode(a.prompt).ids
    for i in range(a.n):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, a.tokens, a.temperature, a.top_k)[0].tolist()
        print(f"--- sample {i+1} ---")
        print(tok.decode(out).strip(), "\n")


if __name__ == "__main__":
    main()
