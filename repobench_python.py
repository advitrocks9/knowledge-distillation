"""
RepoBench-Python next-line prediction eval.

Mellum-4b-sft-python's model card reports `RepoBench Avg ≤8k = 0.299`
on this benchmark. The metric is edit-similarity (difflib ratio) between
the model's first generated line and the canonical next line, averaged
across the three RepoBench subsets (in_file, cross_file_first,
cross_file_random) at file lengths ≤ 8k tokens.

I run all five models from the FIM follow-up on this third eval column,
because (a) it's the published Mellum number and (b) it's pure
left-to-right next-line prediction, not FIM. If FIM-only fine-tuning
made the students forget LM ability, this is the column where it
shows up.

Sub-sample: 60 problems per subset (180 total) at ≤ 8k file length,
fixed seed shared across models.
"""

from __future__ import annotations
from difflib import SequenceMatcher
from pathlib import Path
import argparse
import json
import random
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SUBSETS = ["cross_file_first", "cross_file_random", "in_file"]


def first_line(text: str) -> str:
    if not text:
        return ""
    for sep in ("\n", "\r"):
        i = text.find(sep)
        if i != -1:
            return text[:i]
    return text


def edit_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def build_prompt(ex: dict) -> str:
    """Format the same way the RepoBench paper concatenates cross-file context:
    snippet blocks first (each labelled by source path), then the in-file
    imports, then the cropped code that ends right before next_line."""
    parts: list[str] = []
    for snip in (ex.get("context") or []):
        path = snip.get("path", "")
        body = snip.get("snippet", "")
        if not body:
            continue
        parts.append(f"# {path}\n{body}")
    if parts:
        cross = "\n\n".join(parts) + "\n\n"
    else:
        cross = ""
    file_hdr = f"# {ex.get('file_path', '')}\n"
    imp = ex.get("import_statement") or ""
    code = ex.get("cropped_code") or ""
    return cross + file_hdr + imp + ("\n" if imp and not imp.endswith("\n") else "") + code


@torch.no_grad()
def eval_one_model(
    name: str,
    path: str,
    tok_path: str,
    subsets_data: dict[str, list[dict]],
    max_new: int,
    device,
    is_mellum: bool = False,
) -> dict:
    print(f"\n=== {name} ({path}) ===")
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
    model.eval()
    out: dict = {"name": name, "ckpt": path, "subsets": {}}
    for sub in SUBSETS:
        rows = subsets_data[sub]
        es_scores = []
        em_scores = []
        per_problem = []
        t0 = time.time()
        for ex in rows:
            prompt = build_prompt(ex)
            if is_mellum:
                # mellum-sft-python is FIM-only; raw L2R prompts make it hit
                # EOS immediately. Wrap as FIM with an empty suffix so it
                # behaves like a code-completion request, not a "predict next
                # token outside any FIM scaffold" request.
                prompt = f"<fim_suffix><fim_prefix>{prompt}<fim_middle>"
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=8192).input_ids.to(device)
            gen = model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
            new_ids = gen[0, ids.size(1):]
            text = tok.decode(new_ids, skip_special_tokens=True)
            pred_line = first_line(text).rstrip()
            gold_line = first_line(ex["next_line"]).rstrip()
            es = edit_similarity(pred_line, gold_line)
            em = int(pred_line == gold_line)
            es_scores.append(es)
            em_scores.append(em)
            per_problem.append({"pred": pred_line[:200], "gold": gold_line[:200], "es": es, "em": em})
        elapsed = time.time() - t0
        mean_es = sum(es_scores) / max(len(es_scores), 1)
        mean_em = sum(em_scores) / max(len(em_scores), 1)
        out["subsets"][sub] = {
            "n": len(es_scores),
            "edit_sim": mean_es,
            "exact_match": mean_em,
            "elapsed_s": elapsed,
            "per_problem": per_problem,
        }
        print(f"  {sub:18}  ES={mean_es:.3f}  EM={mean_em:.3f}  (n={len(es_scores)}, {elapsed:.0f}s)")
    avg_es = sum(out["subsets"][s]["edit_sim"] for s in SUBSETS) / len(SUBSETS)
    avg_em = sum(out["subsets"][s]["exact_match"] for s in SUBSETS) / len(SUBSETS)
    out["avg_edit_sim"] = avg_es
    out["avg_exact_match"] = avg_em
    print(f"  AVG: ES={avg_es:.3f}  EM={avg_em:.3f}")
    del model
    torch.cuda.empty_cache()
    return out


def load_subsets(n_per_subset: int, levels: tuple[str, ...], seed: int) -> dict[str, list[dict]]:
    """Load the three RepoBench subsets (splits in the default config),
    filter rows to the listed difficulty levels (≤8k means 2k/4k/8k),
    subsample n_per_subset with a fixed seed shared across model runs."""
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for sub in SUBSETS:
        ds = load_dataset("tianyang/repobench_python_v1.1", split=sub)
        rows = [dict(r) for r in ds if r.get("level") in levels]
        rng.shuffle(rows)
        rows = rows[:n_per_subset]
        out[sub] = rows
        print(f"loaded {sub}: {len(rows)} examples (levels={levels})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="/models/qwen-coder-0.5b")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--mellum", default="/models/mellum-sft-python")
    ap.add_argument("--out", type=Path, default=Path("results/repobench_python.json"))
    ap.add_argument("--n-per-subset", type=int, default=60)
    ap.add_argument("--levels", default="2k,4k,8k",
                    help="comma-separated difficulty levels to keep (mellum card reports avg over <=8k)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="comma-separated subset of run names to evaluate (default: all)")
    args = ap.parse_args()

    device = torch.device("cuda")
    levels = tuple(s.strip() for s in args.levels.split(",") if s.strip())
    subsets_data = load_subsets(args.n_per_subset, levels, args.seed)

    runs = [
        ("base",       args.student_base),
        ("fim_gold",   str(args.ckpt_dir / "student_fim_gold")),
        ("fim_mellum", str(args.ckpt_dir / "student_fim_mellum")),
        ("fim_mix",    str(args.ckpt_dir / "student_fim_mix")),
        ("mellum_4b",  args.mellum),
    ]
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if args.out.exists() and only:
        # merge into existing results so re-running a subset doesn't drop the rest
        out = json.loads(args.out.read_text())
        out["args"] = {"n_per_subset": args.n_per_subset, "levels": args.levels,
                       "max_new": args.max_new, "seed": args.seed}
    else:
        out = {
            "args": {"n_per_subset": args.n_per_subset, "levels": args.levels,
                     "max_new": args.max_new, "seed": args.seed},
            "subset_sizes": {s: len(subsets_data[s]) for s in SUBSETS},
        }
    for name, path in runs:
        if only and name not in only:
            continue
        if not Path(path).exists():
            print(f"missing {path}, skip")
            continue
        is_mellum = "mellum" in name.lower() and "fim_mellum" not in name.lower()
        tok_path = args.mellum if is_mellum else args.student_base
        out[name] = eval_one_model(name, path, tok_path, subsets_data, args.max_new, device, is_mellum=is_mellum)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
