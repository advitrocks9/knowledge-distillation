"""
Build a Python training corpus for distillation.

Pulls N samples from codeparrot-clean-valid, drops very-short ones,
truncates to a fixed sequence length, and packs into a tensor on disk.
Teacher and student share the Qwen2.5 tokenizer, so tokenisation
happens once here and the .pt files get reused across every
distillation run.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def build(
    n_train: int,
    n_val: int,
    seq_len: int,
    tokenizer_path: str,
    out_dir: Path,
    seed: int = 0,
) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # codeparrot-clean-valid is a 61k-sample held-out slice from the
    # CodeParrot Python pretraining set. Public, no auth needed, and not
    # in any of the obvious benchmark training mixes.
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train")
    ds = ds.shuffle(seed=seed)

    train_ids: list[torch.Tensor] = []
    val_ids: list[torch.Tensor] = []
    target_total = n_train + n_val
    n_seen = 0
    for ex in ds:
        text = ex["content"]
        # drop very short files; we want enough context to actually distill
        if len(text) < 200:
            continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"][0]
        if ids.size(0) < seq_len // 2:
            continue
        if ids.size(0) < seq_len:
            pad = torch.full(
                (seq_len - ids.size(0),), tok.pad_token_id, dtype=ids.dtype
            )
            ids = torch.cat([ids, pad])
        n_seen += 1
        if n_seen <= n_train:
            train_ids.append(ids)
        else:
            val_ids.append(ids)
        if n_seen >= target_total:
            break

    train = torch.stack(train_ids)
    val = torch.stack(val_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train, out_dir / "train.pt")
    torch.save(val, out_dir / "val.pt")
    print(f"train {tuple(train.shape)} -> {out_dir / 'train.pt'}")
    print(f"val   {tuple(val.shape)} -> {out_dir / 'val.pt'}")
    # sanity print
    print("\nsample 0 (first 200 chars):")
    print(tok.decode(train[0][:80]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--tokenizer", default="/home/prannayk/models/qwen-coder-0.5b")
    ap.add_argument("--out", type=Path, default=Path("data/cache"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.n_train, args.n_val, args.seq_len, args.tokenizer, args.out, args.seed)
