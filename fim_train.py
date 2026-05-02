"""
Train Qwen2.5-Coder-0.5B on FIM examples. Three sources of middle text:

  --middle gold            -- ground truth middle from the corpus
  --middle mellum          -- Mellum-generated middle (seq-KD target)
  --middle mix             -- 50/50 mix of gold and mellum on the same examples

Loss is masked to the middle tokens only -- the prefix and suffix are
context, not learning targets.

The Qwen FIM format is PSM order:
    <|fim_prefix|> + prefix + <|fim_suffix|> + suffix + <|fim_middle|> + middle

Qwen2.5-Coder is FIM-pretrained so it has these tokens natively.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import math
import random
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer


def cosine_warmup(opt, warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(opt, lr_lambda=fn)


@dataclass
class FIMTokens:
    prefix: int
    suffix: int
    middle: int
    pad: int


def get_qwen_fim_tokens(tok) -> FIMTokens:
    p = tok.convert_tokens_to_ids("<|fim_prefix|>")
    s = tok.convert_tokens_to_ids("<|fim_suffix|>")
    m = tok.convert_tokens_to_ids("<|fim_middle|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if p is None or s is None or m is None:
        raise SystemExit("qwen tokenizer is missing FIM tokens; expected <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>")
    return FIMTokens(prefix=p, suffix=s, middle=m, pad=pad)


def load_examples(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for ln in f:
            out.append(json.loads(ln))
    return out


def build_one(
    ex: dict,
    middle_text: str,
    tok,
    fim: FIMTokens,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build (input_ids, target_mask). Mask is 1 over middle tokens only."""
    prefix_ids = tok(ex["prefix"], add_special_tokens=False).input_ids
    suffix_ids = tok(ex["suffix"], add_special_tokens=False).input_ids
    middle_ids = tok(middle_text, add_special_tokens=False).input_ids
    eos = tok.eos_token_id

    # truncate prefix/suffix if combined too long
    overhead = 4  # 3 fim tokens + 1 eos
    budget = seq_len - len(middle_ids) - overhead
    if budget < 16 or len(middle_ids) < 1:
        return None
    # split budget between prefix and suffix; biased a bit toward prefix for L2R
    pre_budget = budget // 2
    suf_budget = budget - pre_budget
    prefix_ids = prefix_ids[-pre_budget:]
    suffix_ids = suffix_ids[:suf_budget]

    ids = (
        [fim.prefix] + prefix_ids
        + [fim.suffix] + suffix_ids
        + [fim.middle] + middle_ids
        + [eos]
    )
    middle_start = 1 + len(prefix_ids) + 1 + len(suffix_ids) + 1
    middle_end = middle_start + len(middle_ids) + 1  # include eos in loss
    mask = [0] * len(ids)
    for i in range(middle_start, middle_end):
        mask[i] = 1

    # pad to seq_len
    if len(ids) < seq_len:
        pad_len = seq_len - len(ids)
        ids = ids + [fim.pad] * pad_len
        mask = mask + [0] * pad_len
    elif len(ids) > seq_len:
        ids = ids[:seq_len]
        mask = mask[:seq_len]

    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def make_batches(
    examples: list[dict],
    middle_source: str,
    tok,
    fim: FIMTokens,
    seq_len: int,
    rng: random.Random,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for ex in examples:
        if middle_source == "gold":
            middle = ex["middle"]
        elif middle_source == "mellum":
            if "mellum_middle" not in ex:
                continue
            middle = ex["mellum_middle"]
            if len(middle.strip()) < 3:
                continue
        elif middle_source == "mix":
            # alternate per example based on rng
            if rng.random() < 0.5:
                middle = ex["middle"]
            else:
                if "mellum_middle" not in ex:
                    middle = ex["middle"]
                else:
                    middle = ex["mellum_middle"]
                    if len(middle.strip()) < 3:
                        middle = ex["middle"]
        else:
            raise ValueError(middle_source)
        built = build_one(ex, middle, tok, fim, seq_len)
        if built is None:
            continue
        out.append(built)
    return out


def fim_loss(s_logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # next-token prediction with shift, loss only on positions where mask=1
    sl = s_logits[:, :-1, :].contiguous()
    tg = ids[:, 1:].contiguous()
    ms = mask[:, 1:].float()
    flat = sl.view(-1, sl.size(-1))
    losses = F.cross_entropy(flat, tg.view(-1), reduction="none").view_as(tg)
    return (losses * ms).sum() / ms.sum().clamp_min(1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--middle", required=True, choices=["gold", "mellum", "mix"])
    ap.add_argument("--student", default="/home/prannayk/models/qwen-coder-0.5b")
    ap.add_argument("--data", type=Path, default=Path("data/fim/mellum_completions.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.student)
    fim = get_qwen_fim_tokens(tok)
    print(f"qwen FIM tokens: prefix={fim.prefix} suffix={fim.suffix} middle={fim.middle} pad={fim.pad}")

    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16).to(device)
    student.gradient_checkpointing_enable()
    student.train()
    print(f"student params: {sum(p.numel() for p in student.parameters()):,}")

    examples = load_examples(args.data)
    rng.shuffle(examples)
    train_ex = examples[: max(1, int(0.92 * len(examples)))]
    val_ex   = examples[max(1, int(0.92 * len(examples))) :]
    train_batches = make_batches(train_ex, args.middle, tok, fim, args.seq_len, rng)
    # for val we always use gold middles (it's the held-out signal)
    val_batches = make_batches(val_ex, "gold", tok, fim, args.seq_len, rng)
    print(f"train examples: {len(train_batches)}  val: {len(val_batches)}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = cosine_warmup(opt, args.warmup, args.steps)

    log = {"step": [], "train_loss": [], "val_middle_nll": [], "wallclock_s": []}
    t0 = time.time()
    accum = 0
    accum_loss = 0.0
    opt.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train_batches), (args.batch_size,)).tolist()
        batch_ids = torch.stack([train_batches[i][0] for i in idx]).to(device)
        batch_mask = torch.stack([train_batches[i][1] for i in idx]).to(device)
        out = student(batch_ids)
        loss = fim_loss(out.logits, batch_ids, batch_mask) / args.grad_accum
        loss.backward()
        accum_loss += loss.item()
        accum += 1
        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            shown = accum_loss
            accum_loss = 0.0
        else:
            shown = None
        sched.step()

        if step % args.log_every == 0:
            wc = time.time() - t0
            print(f"step {step:5d}/{args.steps}  loss={(shown if shown else loss.item()*args.grad_accum):.4f}  lr={sched.get_last_lr()[0]:.2e}  ({wc:.0f}s)")

        if step % args.eval_every == 0 or step == args.steps:
            student.eval()
            with torch.no_grad():
                total = 0.0
                tokens = 0
                for vi in range(0, len(val_batches), args.batch_size):
                    chunk = val_batches[vi : vi + args.batch_size]
                    if not chunk:
                        break
                    bi = torch.stack([c[0] for c in chunk]).to(device)
                    bm = torch.stack([c[1] for c in chunk]).to(device)
                    o = student(bi)
                    sl = o.logits[:, :-1, :]
                    tg = bi[:, 1:]
                    ms = bm[:, 1:].float()
                    losses = F.cross_entropy(sl.reshape(-1, sl.size(-1)), tg.reshape(-1), reduction="none").view_as(tg)
                    total += (losses * ms).sum().item()
                    tokens += ms.sum().item()
                val_nll = total / max(tokens, 1)
            student.train()
            wc = time.time() - t0
            print(f"step {step:5d}  val_middle_nll={val_nll:.4f}  ({wc:.0f}s)")
            log["step"].append(step)
            log["train_loss"].append(shown)
            log["val_middle_nll"].append(val_nll)
            log["wallclock_s"].append(wc)

    out_path = args.out_dir / f"student_fim_{args.middle}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_path)
    log_path = Path("results") / f"train_fim_{args.middle}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"args": vars(args), "log": log}, indent=2, default=str))
    print(f"saved {out_path}")
    print(f"saved log {log_path}")


if __name__ == "__main__":
    main()
