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
    cfc = ex.get("cross_file_context") or ""
    code = ex.get("code") or ex.get("file_prefix") or ""
    if cfc:
        return cfc + "\n" + code
    return code


@torch.no_grad()
def eval_one_model(
    name: str,
    path: str,
    tok_path: str,
    subsets_data: dict[str, list[dict]],
    max_new: int,
    device,
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


def load_subsets(n_per_subset: int, max_file_tokens: int, seed: int) -> dict[str, list[dict]]:
    """Load the three RepoBench subsets, filter to ≤ max_file_tokens, subsample."""
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for sub in SUBSETS:
        # tianyang/repobench_python_v1.1 ships configs as `<subset>_2k`, `<subset>_4k`, `<subset>_8k`, etc.
        # ≤8k means we pull 2k + 4k + 8k buckets and pool them.
        buckets = ["2k", "4k", "8k"]
        rows: list[dict] = []
        for b in buckets:
            try:
                ds = load_dataset("tianyang/repobench_python_v1.1", f"{sub}_{b}", split="train", trust_remote_code=True)
            except Exception as e:
                print(f"  fallback for {sub}_{b}: {e!r}")
                try:
                    ds = load_dataset("tianyang/repobench_python_v1.1", f"{sub}_{b}", split="test", trust_remote_code=True)
                except Exception as e2:
                    print(f"  also failed: {e2!r}")
                    continue
            rows.extend(ds.to_list() if hasattr(ds, "to_list") else [dict(r) for r in ds])
        if not rows:
            print(f"  WARN: no rows for {sub}")
            out[sub] = []
            continue
        rng.shuffle(rows)
        rows = rows[:n_per_subset]
        out[sub] = rows
        print(f"loaded {sub}: {len(rows)} examples")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-base", default="/models/qwen-coder-0.5b")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--mellum", default="/models/mellum-sft-python")
    ap.add_argument("--out", type=Path, default=Path("results/repobench_python.json"))
    ap.add_argument("--n-per-subset", type=int, default=60)
    ap.add_argument("--max-file-tokens", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    subsets_data = load_subsets(args.n_per_subset, args.max_file_tokens, args.seed)

    runs = [
        ("base",       args.student_base),
        ("fim_gold",   str(args.ckpt_dir / "student_fim_gold")),
        ("fim_mellum", str(args.ckpt_dir / "student_fim_mellum")),
        ("fim_mix",    str(args.ckpt_dir / "student_fim_mix")),
        ("mellum_4b",  args.mellum),
    ]
    out: dict = {
        "args": {"n_per_subset": args.n_per_subset, "max_file_tokens": args.max_file_tokens,
                 "max_new": args.max_new, "seed": args.seed},
        "subset_sizes": {s: len(subsets_data[s]) for s in SUBSETS},
    }
    for name, path in runs:
        if not Path(path).exists():
            print(f"missing {path}, skip")
            continue
        tok_path = args.mellum if "mellum" in name.lower() and "fim_mellum" not in name.lower() else args.student_base
        out[name] = eval_one_model(name, path, tok_path, subsets_data, args.max_new, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
