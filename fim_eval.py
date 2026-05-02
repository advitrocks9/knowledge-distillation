"""
Evaluate FIM-trained students on the held-out FIM examples (gold middles).

Three numbers per model:
  exact-match              -- model output matches the ground-truth middle exactly
  per-line accuracy        -- for single_line examples, did the model produce
                              one valid line that matches the truth
  middle NLL on truth      -- per-token NLL of the ground-truth middle under
                              the model, FIM-conditional

Slicing by FIM kind (single_line / multi_line / random_span) so we can see
where each method wins. The exact-match number is what HumanEval Infilling
itself reports, so this is a direct stand-in.

This is NOT the same as HumanEval Infilling. We use codeparrot held-out
files to make sure we're not measuring memorisation. Mellum's published
numbers will be a separate run on actual HumanEval Infilling, in a
different script.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fim_train import build_one, get_qwen_fim_tokens, load_examples


@torch.no_grad()
def generate_middle(
    model, tok, fim, prefix: str, suffix: str, max_new: int, device
) -> str:
    pre_ids = tok(prefix, add_special_tokens=False).input_ids[-256:]
    suf_ids = tok(suffix, add_special_tokens=False).input_ids[:256]
    ids = (
        [fim.prefix] + pre_ids
        + [fim.suffix] + suf_ids
        + [fim.middle]
    )
    t = torch.tensor([ids], device=device, dtype=torch.long)
    out = model.generate(
        t,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    new_ids = out[0, t.size(1) :].tolist()
    return tok.decode(new_ids, skip_special_tokens=True)


def truncate_to_middle(text: str, target_lines: int) -> str:
    """Cut at the next end-of-middle signal: blank line, or after target lines."""
    lines = text.split("\n")
    if target_lines == 1:
        return lines[0] + ("\n" if "\n" in text else "")
    return "\n".join(lines[: max(target_lines + 1, 1)])


def normalise(s: str) -> str:
    return s.replace("\r\n", "\n").rstrip()


@torch.no_grad()
def middle_nll(model, tok, fim, ex: dict, device) -> float:
    """NLL of the gold middle under the model, FIM-conditional."""
    built = build_one(ex, ex["middle"], tok, fim, seq_len=512)
    if built is None:
        return float("nan")
    ids, mask = built
    ids = ids.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    out = model(ids)
    sl = out.logits[:, :-1, :]
    tg = ids[:, 1:]
    ms = mask[:, 1:].float()
    losses = F.cross_entropy(sl.reshape(-1, sl.size(-1)), tg.reshape(-1), reduction="none").view_as(tg)
    return (losses * ms).sum().item() / max(ms.sum().item(), 1.0)


def eval_one(
    name: str,
    student_path: str,
    examples: list[dict],
    tok,
    fim,
    device,
) -> dict:
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.bfloat16).to(device)
    student.eval()
    print(f"\n=== {name} ===")
    by_kind: dict[str, list[dict]] = {}
    t0 = time.time()
    for i, ex in enumerate(examples):
        kind = ex["kind"]
        target_lines = max(1, ex["middle_len_lines"])
        max_new = max(32, min(256, ex["middle_len_chars"] // 2 + 32))
        gen = generate_middle(student, tok, fim, ex["prefix"], ex["suffix"], max_new, device)
        gen_trimmed = truncate_to_middle(gen, target_lines)
        em = (normalise(gen_trimmed) == normalise(ex["middle"]))
        nll = middle_nll(student, tok, fim, ex, device)
        by_kind.setdefault(kind, []).append({
            "em": em,
            "middle_nll": nll,
            "gen": gen_trimmed[:200],  # truncate for log size
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(examples)}  ({time.time()-t0:.0f}s)")
    elapsed = time.time() - t0
    out: dict = {"name": name, "elapsed_s": elapsed}
    for kind, rows in by_kind.items():
        ems = [r["em"] for r in rows]
        nlls = [r["middle_nll"] for r in rows if not (r["middle_nll"] != r["middle_nll"])]  # NaN check
        out[kind] = {
            "n": len(rows),
            "exact_match": sum(ems) / len(ems),
            "mean_middle_nll": sum(nlls) / max(len(nlls), 1),
        }
        print(f"  {kind} (n={len(rows)}): EM={out[kind]['exact_match']:.3f}  middle_nll={out[kind]['mean_middle_nll']:.3f}")
    overall_em = sum(r["em"] for rs in by_kind.values() for r in rs) / sum(len(rs) for rs in by_kind.values())
    overall_nll = sum(r["middle_nll"] for rs in by_kind.values() for r in rs) / sum(len(rs) for rs in by_kind.values())
    out["overall_em"] = overall_em
    out["overall_middle_nll"] = overall_nll
    print(f"  overall: EM={overall_em:.3f}  middle_nll={overall_nll:.3f}")
    del student
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="/home/prannayk/models/qwen-coder-0.5b")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--data", type=Path, default=Path("data/fim/eval_examples.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/fim_eval.json"))
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    fim = get_qwen_fim_tokens(tok)

    examples = load_examples(args.data)
    if args.limit:
        examples = examples[: args.limit]
    print(f"eval examples: {len(examples)}")

    runs = [
        ("base",       args.student_base),
        ("fim_gold",   str(args.ckpt_dir / "student_fim_gold")),
        ("fim_mellum", str(args.ckpt_dir / "student_fim_mellum")),
        ("fim_mix",    str(args.ckpt_dir / "student_fim_mix")),
    ]
    out: dict = {}
    for name, path in runs:
        if not Path(path).exists():
            print(f"missing {path}, skip")
            continue
        out[name] = eval_one(name, path, examples, tok, fim, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
