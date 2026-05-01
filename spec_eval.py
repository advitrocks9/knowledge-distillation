"""
Hardened speculative-decoding draft acceptance eval.

The original spec-decode in eval.py used 32 HumanEval prompts and reported
a single mean. Codex called this out: 32 prompts isn't enough to make a
ranking statement, the prompt mix matters, draft cycles within a prompt
aren't independent, and a sign flip between K=2 and K=4 was probably noise.

This script does the eval the way it should have been done in the first place.

Decisions, with the reasoning written down so I can defend each one:

  Prefix corpus: all 164 HumanEval prompts. Function-completion shape, the
  closest thing to IDE-relevant code completion. Yes, HumanEval is in
  pretraining mixes -- so are the alternatives. I don't mix with random
  codeparrot truncations because random mid-statement truncation inflates
  next-token sharpness artefactually, and mixing two corpora without a
  weighting rationale lets the variant ranking flip from mix changes.

  Drafts: sampled at T=1.0 (the rule from Leviathan et al. 2023). Greedy
  drafts test mode agreement only and flatter the numbers; deployment uses
  sampled drafts and that's what should be reported. T is locked across
  variants -- if it's a free knob, it manufactures gaps.

  K: 4 only. K=8 is only worth running if K=4 is precise enough to
  discriminate trend from variance, and at this prompt count it isn't.

  Bootstrap: by prompt, 1000 resamples, percentile CI. Cycles within a
  prompt aren't independent so cycle-level bootstrap would understate
  uncertainty. To not hide cycle-level variance, the report also surfaces
  the within-prompt CV.

  Eval seed: fixed and shared across variants. Different seeds across
  runs would let evaluation noise show up as ranking signal.

  Per-position figure: only the first drafted block per prompt. Later
  cycles are conditioned on accepted student prefixes and bias the
  position-1 number; first-block-only avoids that. Caveat: first blocks
  are disproportionately near prompt boundaries, so the curve overstates
  later-position acceptability for the actual deployed regime. Reported
  per-position N alongside the mean.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def spec_decode_one_prompt(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    prompt_ids: torch.Tensor,
    tok,
    K: int,
    max_drafts: int,
    rng_seed: int,
    device: torch.device,
) -> dict:
    """
    For one prompt, run up to max_drafts draft cycles. Return per-cycle
    accepted run lengths, plus the per-position acceptance probabilities
    *for the first block only* (to avoid contamination from previously
    accepted student prefixes).
    """
    torch.manual_seed(rng_seed)
    cur = prompt_ids.clone()
    cycles: list[float] = []
    first_block_pos_probs: list[float] | None = None

    for cycle_idx in range(max_drafts):
        draft = student.generate(
            cur,
            max_new_tokens=K,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id,
        )
        new_tokens = draft.size(1) - cur.size(1)
        if new_tokens == 0:
            break

        # one-shot scoring of the whole draft: cheaper than re-running per
        # token because both models can amortise across the K positions
        s_logits = student(draft).logits
        t_logits = teacher(draft).logits
        p_s = F.softmax(s_logits, dim=-1)
        p_t = F.softmax(t_logits, dim=-1)

        start = cur.size(1) - 1
        ratios = []
        for offset in range(new_tokens):
            pos = start + offset
            tok_id = draft[0, pos + 1].item()
            ps = p_s[0, pos, tok_id].item()
            pt = p_t[0, pos, tok_id].item()
            ratios.append(min(1.0, pt / ps) if ps > 1e-9 else 0.0)

        # E[accepted run length] = sum_i prod_{j<=i} a_j
        cum = 1.0
        run = 0.0
        for r in ratios:
            cum *= r
            run += cum
        cycles.append(run)

        if first_block_pos_probs is None:
            first_block_pos_probs = ratios + [None] * (K - len(ratios))

        # advance by an integer accepted prefix length: standard spec-decode
        # advances by the actually accepted prefix; for the proxy I round
        # the expected run length, since we don't roll the Bernoulli
        advance = max(1, min(int(round(run)), new_tokens))
        cur = draft[:, : cur.size(1) + advance]
        if cur.size(1) >= 512:
            break

    return {
        "cycles": cycles,
        "first_block_pos_probs": first_block_pos_probs,
    }


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap on the mean. Returns (mean, lo, hi)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(arr.mean()), lo, hi


@torch.no_grad()
def evaluate_one(
    name: str,
    student_path: str,
    teacher: torch.nn.Module,
    tok,
    prompts: list[str],
    K: int,
    max_drafts: int,
    device: torch.device,
    eval_seed: int,
) -> dict:
    student = load_model(student_path, device)

    per_prompt_mean_run: list[float] = []
    per_prompt_cv: list[float] = []
    per_position_probs: list[list[float | None]] = []
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=384).input_ids.to(device)
        # per-prompt seed so that across variants the same prompt sees the
        # same RNG -- evaluation noise can't masquerade as training signal
        rng_seed = eval_seed * 1_000_003 + i
        out = spec_decode_one_prompt(
            student, teacher, ids, tok, K=K, max_drafts=max_drafts,
            rng_seed=rng_seed, device=device,
        )
        if not out["cycles"]:
            continue
        cycles = out["cycles"]
        per_prompt_mean_run.append(float(np.mean(cycles)))
        if len(cycles) > 1:
            cv = float(np.std(cycles) / max(np.mean(cycles), 1e-9))
            per_prompt_cv.append(cv)
        if out["first_block_pos_probs"] is not None:
            per_position_probs.append(out["first_block_pos_probs"])

    elapsed = time.time() - t0
    mean, lo, hi = bootstrap_ci(per_prompt_mean_run)

    # per-position aggregates -- skip None (positions where generation stopped early)
    pos_aggregates = []
    for k in range(K):
        vals = [row[k] for row in per_position_probs if row[k] is not None]
        if not vals:
            pos_aggregates.append({"k": k + 1, "n": 0, "mean": None})
            continue
        m, lo_p, hi_p = bootstrap_ci(vals)
        pos_aggregates.append({"k": k + 1, "n": len(vals), "mean": m, "ci_lo": lo_p, "ci_hi": hi_p})

    del student
    torch.cuda.empty_cache()

    return {
        "name": name,
        "n_prompts_with_cycles": len(per_prompt_mean_run),
        "K": K,
        "max_drafts": max_drafts,
        "mean_accepted_run_length": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "within_prompt_cv_mean": float(np.mean(per_prompt_cv)) if per_prompt_cv else None,
        "per_position": pos_aggregates,
        "elapsed_s": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="/home/prannayk/models/qwen-coder-1.5b")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--student-base", default="/home/prannayk/models/qwen-coder-0.5b")
    ap.add_argument("--out", type=Path, default=Path("results/spec_eval.json"))
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max-drafts", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.student_base)

    # all 164 HumanEval prompts; this is the one corpus -- chosen because
    # it has a principled truncation point (the function signature + docstring)
    # and represents the IDE function-completion shape
    he = load_dataset("openai_humaneval", split="test")
    prompts = [ex["prompt"] for ex in he]

    teacher = load_model(args.teacher, device)

    runs = [
        ("teacher_self", args.teacher),
        ("student_base", args.student_base),
        ("student_ce", str(args.ckpt_dir / "student_ce")),
        ("student_fkl", str(args.ckpt_dir / "student_fkl")),
        ("student_rkl", str(args.ckpt_dir / "student_rkl")),
        ("student_gkd", str(args.ckpt_dir / "student_gkd")),
    ]
    out: dict = {"prompts": "humaneval-164", "K": args.K, "max_drafts": args.max_drafts}

    for name, path in runs:
        if not Path(path).exists():
            print(f"missing {path}, skip")
            continue
        print(f"\n=== {name} ===")
        out[name] = evaluate_one(
            name, path, teacher, tok, prompts,
            K=args.K, max_drafts=args.max_drafts,
            device=device, eval_seed=args.eval_seed,
        )
        r = out[name]
        print(f"  mean run: {r['mean_accepted_run_length']:.3f} / {args.K}")
        print(f"  95% CI:  [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]")
        if r['within_prompt_cv_mean'] is not None:
            print(f"  within-prompt CV: {r['within_prompt_cv_mean']:.3f}")
        print(f"  per-position mean acceptance:")
        for p in r['per_position']:
            if p['mean'] is None:
                continue
            print(f"    pos {p['k']} (n={p['n']}): {p['mean']:.3f}  [{p['ci_lo']:.3f}, {p['ci_hi']:.3f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
