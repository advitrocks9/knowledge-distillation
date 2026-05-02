"""
HumanEval Infilling eval -- the metric Mellum's own model card reports.

Mellum-4b-base reports:
  Single-Line:  66.21%
  Multi-Line:   38.52%
  Random Span:  29.70%

(JetBrains/Mellum-4b-base model card, accessed 2026-04-30.)

Mellum-4b-sft-python reports the RepoBench numbers
(Avg ≤ 8k = 0.299, 8k = 0.298) per the same source.

The HumanEval Infilling dataset (Bavarian et al., 2022) carves three subsets
out of the standard HumanEval problems by masking different kinds of spans
in the canonical solution. The infilling setup gives the model a prefix and
a suffix and asks for the span between them. Pass@1 is computed by running
the original test against the reconstructed solution.

This script puts all four students -- base, fim_gold, fim_mellum, fim_mix --
on the same eval, plus Mellum itself as the upper bound.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import signal
import subprocess
import sys
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from fim_train import get_qwen_fim_tokens


SUBSETS = ["single-line", "multi-line", "random-span"]


def build_qwen_fim(prefix: str, suffix: str, tok, fim) -> torch.Tensor:
    pre_ids = tok(prefix, add_special_tokens=False).input_ids
    suf_ids = tok(suffix, add_special_tokens=False).input_ids
    ids = (
        [fim.prefix] + pre_ids
        + [fim.suffix] + suf_ids
        + [fim.middle]
    )
    return torch.tensor([ids], dtype=torch.long)


def build_mellum_fim(prefix: str, suffix: str, tok) -> torch.Tensor:
    # Mellum's SPM order: <fim_suffix> suffix <fim_prefix> prefix <fim_middle>
    prompt = f"<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>"
    return tok(prompt, return_tensors="pt").input_ids


def truncate_completion(text: str) -> str:
    """Cut at the first sign the model has finished the infill: a blank line
    followed by something at column 0, or another fim/eos token."""
    for stop in ["<|endoftext|>", "<|fim_pad|>", "<fim_pad>"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text


def run_test(prefix: str, completion: str, suffix: str, test: str, entry: str) -> bool:
    program = prefix + completion + suffix + "\n" + test + f"\ncheck({entry})\n"
    try:
        out = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=10,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False
    return out.returncode == 0


@torch.no_grad()
def eval_subset(
    name: str,
    model,
    tok,
    fim,
    is_mellum: bool,
    subset: str,
    n_problems: int,
    max_new: int,
    device,
) -> dict:
    ds = load_dataset("loubnabnl/humaneval_infilling", subset.replace("-", "_") + "_infilling", split="test")
    if n_problems and n_problems < len(ds):
        ds = ds.select(range(n_problems))
    n_pass = 0
    rows: list[dict] = []
    t0 = time.time()
    for ex in ds:
        prefix = ex["prompt"]
        suffix = ex["suffix"]
        if is_mellum:
            ids = build_mellum_fim(prefix, suffix, tok).to(device)
        else:
            ids = build_qwen_fim(prefix, suffix, tok, fim).to(device)
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0, ids.size(1) :], skip_special_tokens=True)
        gen = truncate_completion(gen)
        passed = run_test(prefix, gen, suffix, ex["test"], ex["entry_point"])
        n_pass += int(passed)
        rows.append({"task_id": ex["task_id"], "passed": passed, "completion": gen[:200]})
    elapsed = time.time() - t0
    return {
        "subset": subset,
        "n": len(rows),
        "pass@1": n_pass / max(len(rows), 1),
        "elapsed_s": elapsed,
        "per_problem": rows,
    }


def evaluate(
    name: str, path: str, n_problems: int, max_new: int,
    base_tok_path: str, device,
) -> dict:
    is_mellum = "mellum" in name.lower() and "fim_mellum" not in name.lower()
    if is_mellum:
        tok = AutoTokenizer.from_pretrained(path)
        fim = None
    else:
        tok = AutoTokenizer.from_pretrained(base_tok_path)
        fim = get_qwen_fim_tokens(tok)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"\n=== {name} ({path}) ===")
    out = {"name": name, "ckpt": path, "subsets": {}}
    for subset in SUBSETS:
        r = eval_subset(name, model, tok, fim, is_mellum, subset, n_problems, max_new, device)
        out["subsets"][subset] = r
        print(f"  {subset}: pass@1 = {r['pass@1']:.3f}  (n={r['n']}, {r['elapsed_s']:.0f}s)")
    overall = sum(out["subsets"][s]["pass@1"] for s in SUBSETS) / 3
    out["mean_pass@1"] = overall
    print(f"  mean pass@1: {overall:.3f}")
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="/home/prannayk/models/qwen-coder-0.5b")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--mellum", default="/home/prannayk/models/mellum-sft-python")
    ap.add_argument("--out", type=Path, default=Path("results/humaneval_infilling.json"))
    ap.add_argument("--n-problems", type=int, default=164)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda")
    runs = [
        ("base",        args.student_base),
        ("fim_gold",    str(args.ckpt_dir / "student_fim_gold")),
        ("fim_mellum",  str(args.ckpt_dir / "student_fim_mellum")),
        ("fim_mix",     str(args.ckpt_dir / "student_fim_mix")),
        ("mellum_4b",   args.mellum),
    ]
    out: dict = {}
    for name, path in runs:
        if not Path(path).exists():
            print(f"missing {path}, skip")
            continue
        out[name] = evaluate(name, path, args.n_problems, args.max_new, args.student_base, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
