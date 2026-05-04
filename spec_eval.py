"""
Hardened spec-decode draft-acceptance eval.

The first version (in eval.py) used 32 HumanEval prompts, no error
bars, asymmetric max_drafts at K=2 vs K=4, and gave me a K=2/K=4
sign-flip I attached a story to. Code-review pointed out that 32
prompts can't bracket a K=4 mean, draft cycles within a prompt aren't
independent, and most of the gap I was reading was noise. This script
fixes all of that.

What it does: all 164 HumanEval prompts (function-completion shape,
the closest thing to IDE relevant), drafts sampled at T=1.0
(Leviathan rule, locked across variants), K=4 with up to 8 cycles per
prompt, by-prompt bootstrap CIs at 95%, within-prompt cycle CV
reported alongside the mean (a stable mean with high cycle CV is uneven
user-perceived latency), shared eval seed across variants. Per-position
acceptance reported only on the first drafted block per prompt to keep
position-1 from being biased by previously-accepted student prefixes.
First-block positions are disproportionately near prompt boundaries,
so the per-position curve overstates later-position acceptance for the
actual deployed regime; per-position N is reported alongside.
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

        # Per-block analytic expected accept-run E[L] = sum_i prod_{j<=i} a_j,
        # where a_j = min(1, p_T/p_S). This is the closed-form expectation
        # of the Leviathan rejection chain with student-temp-1 drafts; with
        # 164 prompts and ~8 cycles each, the sample mean of an actual
        # rejection roll converges to this same number, with slightly wider
        # bootstrap CIs. Reporting the expectation directly removes one
        # source of eval noise so the variant ranking surfaces.
        cum = 1.0
        run = 0.0
        for r in ratios:
            cum *= r
            run += cum
        cycles.append(run)

        if first_block_pos_probs is None:
            first_block_pos_probs = ratios + [None] * (K - len(ratios))

        # Advance step is a proxy: real spec-decode advances by the actually
        # accepted prefix length per cycle (a sample from the rejection
        # chain); I advance by the integer-rounded expectation so cycles
        # progress at the same average rate without rolling a Bernoulli.
        # First-cycle stats (the per_position numbers) are unaffected by
        # this choice; later-cycle prefix positions shift slightly.
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
