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


@torch.no_grad()
def generate_batch(
    teacher: torch.nn.Module,
    tok,
    examples: list[dict],
    max_new: int,
    device: torch.device,
    greedy: bool,
) -> list[str]:
    """Batched generation for speed. Pads to max input length in the batch."""
    prompts = [build_mellum_fim_prompt(ex["prefix"], ex["suffix"], tok) for ex in examples]
    enc = tok(prompts, return_tensors="pt", truncation=True, max_length=1536, padding=True)
    ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    eos_id = tok.eos_token_id
    out = teacher.generate(
        ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        do_sample=not greedy,
        temperature=0.0 if greedy else 0.7,
        pad_token_id=eos_id,
        eos_token_id=eos_id,
    )
    new_ids = out[:, ids.size(1):]
    texts = tok.batch_decode(new_ids, skip_special_tokens=True)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="/home/prannayk/models/mellum-sft-python")
    ap.add_argument("--in-jsonl", type=Path, default=Path("data/fim/examples.jsonl"))
    ap.add_argument("--out-jsonl", type=Path, default=Path("data/fim/mellum_completions.jsonl"))
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--greedy", action="store_true",
                    help="greedy decoding for seq-KD targets (the conservative choice; sampling is the alternative if greedy targets turn out to be too narrow)")
    ap.add_argument("--limit", type=int, default=0,
                    help="limit number of examples to generate (0 = all)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="batched generation; pads to longest in batch")
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
    bs = args.batch_size
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.padding_side != "left":
        tok.padding_side = "left"
    with args.out_jsonl.open("w") as out_f:
        for start in range(0, len(examples), bs):
            chunk = examples[start : start + bs]
            try:
                gens = generate_batch(
                    teacher, tok, chunk,
                    max_new=args.max_new,
                    device=device,
                    greedy=args.greedy,
                )
            except Exception as e:
                print(f"  batch {start} failed: {e!r}; falling back to per-example")
                gens = []
                for ex in chunk:
                    try:
                        gens.append(generate_one(teacher, tok, ex["prefix"], ex["suffix"],
                                                  args.max_new, device, args.greedy))
                    except Exception as e2:
                        print(f"    inner fail: {e2!r}")
                        gens.append("")
            for ex, gen in zip(chunk, gens):
                row = dict(ex)
                row["mellum_middle"] = gen
                row["mellum_middle_len_chars"] = len(gen)
                out_f.write(json.dumps(row) + "\n")
            done = start + len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            eta = (len(examples) - done) / max(rate, 1e-6)
            print(f"  {done}/{len(examples)}  rate={rate:.2f}/s  eta={eta:.0f}s")

    print(f"wrote {args.out_jsonl}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
