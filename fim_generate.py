"""
Run Mellum-4b-sft-python on the FIM examples and save its generated middle
as the seq-KD target.

Mellum's FIM template is suffix-prefix-middle (SPM order); see the model
card's example. The token IDs are looked up by string and used directly.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_examples(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for ln in f:
            out.append(json.loads(ln))
    return out


def build_mellum_fim_prompt(prefix: str, suffix: str, tok) -> str:
    # SPM order. Mellum's special tokens are <fim_prefix>, <fim_suffix>,
    # <fim_middle>. Concatenated as raw strings; the tokenizer will emit
    # the special-token ids when it sees those exact substrings.
    return f"<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>"


@torch.no_grad()
def generate_one(
    teacher: torch.nn.Module,
    tok,
    prefix: str,
    suffix: str,
    max_new: int,
    device: torch.device,
    greedy: bool,
) -> str:
    prompt = build_mellum_fim_prompt(prefix, suffix, tok)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
    eos_id = tok.eos_token_id
    out = teacher.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=not greedy,
        temperature=0.0 if greedy else 0.7,
        pad_token_id=eos_id,
        eos_token_id=eos_id,
    )
    new_ids = out[0, ids.size(1) :]
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="/home/prannayk/models/mellum-sft-python")
    ap.add_argument("--in-jsonl", type=Path, default=Path("data/fim/examples.jsonl"))
    ap.add_argument("--out-jsonl", type=Path, default=Path("data/fim/mellum_completions.jsonl"))
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--greedy", action="store_true",
                    help="greedy decoding (codex prefers this for seq-KD targets at first)")
    ap.add_argument("--limit", type=int, default=0,
                    help="limit number of examples to generate (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device)
    teacher.eval()
    print(f"loaded mellum, params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"free MiB: {torch.cuda.mem_get_info()[0] // 1024**2}")

    examples = load_examples(args.in_jsonl)
    if args.limit:
        examples = examples[: args.limit]
    print(f"generating for {len(examples)} examples (greedy={args.greedy}, max_new={args.max_new})")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with args.out_jsonl.open("w") as out_f:
        for i, ex in enumerate(examples):
            try:
                gen = generate_one(
                    teacher, tok,
                    prefix=ex["prefix"],
                    suffix=ex["suffix"],
                    max_new=args.max_new,
                    device=device,
                    greedy=args.greedy,
                )
            except Exception as e:
                print(f"  ex {i} failed: {e!r}; skipping")
                continue
            row = dict(ex)
            row["mellum_middle"] = gen
            row["mellum_middle_len_chars"] = len(gen)
            out_f.write(json.dumps(row) + "\n")
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(examples) - i - 1) / max(rate, 1e-6)
                print(f"  {i+1}/{len(examples)}  rate={rate:.1f}/s  eta={eta:.0f}s")

    print(f"wrote {args.out_jsonl}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
