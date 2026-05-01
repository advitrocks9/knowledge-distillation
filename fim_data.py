"""
Build FIM (fill-in-the-middle) training and eval examples from a Python
corpus. For each source file, pick a span to be the "middle" the model has
to fill in. Save (prefix, middle, suffix) text triples.

Three masking strategies, mirroring the HumanEval Infilling axes:

  single_line  -- mask one whole line, prefix and suffix on either side
  multi_line   -- mask 2-5 contiguous lines
  random_span  -- mask a random character span of 50-300 chars

The data this produces is used both for SFT-on-gold-middles and for
Mellum-generates-the-middle seq-KD. The same triples are also used for
held-out eval against ground-truth middles.

Why we need our own FIM corpus on top of HumanEval Infilling: HumanEval
Infilling is the eval set, but for *training* I want a separate corpus
that's more diverse than 164 hand-written tasks. CodeParrot Python files
give that.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class FIMExample:
    kind: str             # "single_line" | "multi_line" | "random_span"
    file_idx: int
    prefix: str
    middle: str
    suffix: str
    middle_len_chars: int
    middle_len_lines: int


def split_single_line(text: str, rng: random.Random) -> tuple[str, str, str] | None:
    lines = text.split("\n")
    if len(lines) < 4:
        return None
    # avoid trivial first / last lines
    candidates = [
        i for i, ln in enumerate(lines[1:-1], start=1)
        if 5 <= len(ln.strip()) <= 120 and not ln.strip().startswith("#")
    ]
    if not candidates:
        return None
    i = rng.choice(candidates)
    prefix = "\n".join(lines[:i]) + "\n"
    middle = lines[i] + "\n"
    suffix = "\n".join(lines[i + 1 :])
    return prefix, middle, suffix


def split_multi_line(text: str, rng: random.Random) -> tuple[str, str, str] | None:
    lines = text.split("\n")
    if len(lines) < 8:
        return None
    span_len = rng.randint(2, 5)
    max_start = len(lines) - span_len - 1
    if max_start <= 1:
        return None
    i = rng.randint(1, max_start)
    j = i + span_len
    prefix = "\n".join(lines[:i]) + "\n"
    middle = "\n".join(lines[i:j]) + "\n"
    suffix = "\n".join(lines[j:])
    if not (10 <= len(middle) <= 600):
        return None
    return prefix, middle, suffix


def split_random_span(text: str, rng: random.Random) -> tuple[str, str, str] | None:
    if len(text) < 200:
        return None
    span_len = rng.randint(50, 300)
    max_start = len(text) - span_len - 50
    if max_start <= 50:
        return None
    start = rng.randint(50, max_start)
    end = start + span_len
    return text[:start], text[start:end], text[end:]


def build(
    n_per_kind: int,
    out_path: Path,
    seed: int = 0,
    skip: int = 0,
) -> None:
    rng = random.Random(seed)
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train")
    ds = ds.shuffle(seed=seed + 1)

    # use a different slice from the distillation training corpus so eval
    # isn't on training files. distill.py uses the first 2048 + 128 with
    # seed 0 + 999 for val. fim_data uses files starting at offset `skip`
    # under a different shuffle seed.
    ds = ds.select(range(skip, min(skip + 8000, len(ds))))

    examples: list[FIMExample] = []
    counters = {"single_line": 0, "multi_line": 0, "random_span": 0}
    targets = {"single_line": n_per_kind, "multi_line": n_per_kind, "random_span": n_per_kind}
    splitters = {
        "single_line": split_single_line,
        "multi_line": split_multi_line,
        "random_span": split_random_span,
    }
    kinds_left = list(splitters.keys())

    for idx, ex in enumerate(ds):
        text: str = ex["content"]
        if len(text) < 300 or len(text) > 8000:
            continue
        # pick the kind with smallest progress
        kinds_left = [k for k in counters if counters[k] < targets[k]]
        if not kinds_left:
            break
        kind = min(kinds_left, key=lambda k: counters[k])
        out = splitters[kind](text, rng)
        if out is None:
            continue
        prefix, middle, suffix = out
        examples.append(FIMExample(
            kind=kind,
            file_idx=idx,
            prefix=prefix,
            middle=middle,
            suffix=suffix,
            middle_len_chars=len(middle),
            middle_len_lines=middle.count("\n"),
        ))
        counters[kind] += 1
        if all(counters[k] >= targets[k] for k in counters):
            break

    print("counters:", counters)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    print(f"wrote {len(examples)} examples -> {out_path}")
    # sanity print
    print("\nfirst single_line example:")
    sl = next(e for e in examples if e.kind == "single_line")
    print(f"prefix (last 60 chars): ...{sl.prefix[-60:]!r}")
    print(f"middle: {sl.middle!r}")
    print(f"suffix (first 60 chars): {sl.suffix[:60]!r}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-kind", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("data/fim/examples.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=10000,
                    help="skip the first N codeparrot files so we don't overlap with distill train+val")
    args = ap.parse_args()
    build(args.n_per_kind, args.out, args.seed, args.skip)


if __name__ == "__main__":
    main()
