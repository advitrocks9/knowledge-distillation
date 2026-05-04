"""
Path-A spec-decode eval: actual Leviathan rejection sampling.

`spec_eval.py` reports the analytic per-block expected accept-run length
E[L] = sum_i prod_{j<=i} a_j (closed form of the rejection chain with
student-temp-1 drafts). That estimator is correct in expectation but not
bit-exact Leviathan: real spec-decode rolls a Bernoulli per drafted
position and stops at the first rejection.

This script does the bit-exact version. Same prompts, same K, same
max_drafts, same per-prompt seeds, but the inner loop draws u ~ U[0,1]
and accepts iff u <= p_T/p_S. The reported `mean_accepted_run_length`
is then the sample mean over per-prompt sample means of the rejection
chain itself, not the analytic expectation.

Output goes to results/spec_eval_rejection.json so it sits next to
spec_eval.json for a side-by-side comparison.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
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
def spec_decode_rejection(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    prompt_ids: torch.Tensor,
    tok,
    K: int,
    max_drafts: int,
    rng_seed: int,
    device: torch.device,
) -> dict:
    """One prompt, up to max_drafts cycles, real rejection sampling.

    Per cycle: draft K student tokens at T=1.0, score under both, draw
    u_i ~ U[0,1] for i in [0, K). Accept the prefix up to the first i
    where u_i > p_T/p_S; advance by that many tokens.
    """
    torch.manual_seed(rng_seed)
    rng = np.random.default_rng(rng_seed)
    cur = prompt_ids.clone()
    cycles_run: list[int] = []           # actual sampled accepted run lengths
    cycles_analytic: list[float] = []    # analytic expectation (for comparison)
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

        s_logits = student(draft).logits
        t_logits = teacher(draft).logits
        p_s = F.softmax(s_logits, dim=-1)
        p_t = F.softmax(t_logits, dim=-1)

        start = cur.size(1) - 1
        ratios: list[float] = []
        for offset in range(new_tokens):
            pos = start + offset
            tok_id = draft[0, pos + 1].item()
            ps = p_s[0, pos, tok_id].item()
            pt = p_t[0, pos, tok_id].item()
            ratios.append(min(1.0, pt / ps) if ps > 1e-9 else 0.0)

        # Real rejection sampling: walk the chain, stop at first rejection.
        u = rng.random(size=len(ratios))
        run = 0
        for r, ui in zip(ratios, u):
            if ui <= r:
                run += 1
            else:
                break
        cycles_run.append(run)

        # Analytic expectation kept for the comparison plot.
        cum = 1.0
        analytic = 0.0
        for r in ratios:
            cum *= r
            analytic += cum
        cycles_analytic.append(analytic)

        if first_block_pos_probs is None:
            first_block_pos_probs = ratios + [None] * (K - len(ratios))

        # Advance by the actually-accepted prefix length (Leviathan).
        advance = max(1, run)
        cur = draft[:, : cur.size(1) + advance]
        if cur.size(1) >= 512:
            break

    return {
        "cycles_run": cycles_run,
        "cycles_analytic": cycles_analytic,
        "first_block_pos_probs": first_block_pos_probs,
    }


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
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
    per_prompt_mean_analytic: list[float] = []
    per_prompt_cv: list[float] = []
    per_position_probs: list[list[float | None]] = []
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=384).input_ids.to(device)
        out = spec_decode_rejection(
            student, teacher, ids, tok, K=K, max_drafts=max_drafts,
            rng_seed=eval_seed * 1000 + i, device=device,
        )
        runs = out["cycles_run"]
        analytics = out["cycles_analytic"]
        if runs:
            per_prompt_mean_run.append(float(np.mean(runs)))
            per_prompt_mean_analytic.append(float(np.mean(analytics)))
            if len(runs) > 1:
                m = float(np.mean(runs))
                if m > 0:
                    per_prompt_cv.append(float(np.std(runs, ddof=1) / m))
        if out["first_block_pos_probs"] is not None:
            per_position_probs.append(out["first_block_pos_probs"])
    elapsed = time.time() - t0
    mean, lo, hi = bootstrap_ci(per_prompt_mean_run)
    mean_a, lo_a, hi_a = bootstrap_ci(per_prompt_mean_analytic)

    pos_aggregates = []
    for k in range(K):
        vals = [p[k] for p in per_position_probs if p[k] is not None]
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
        "mean_analytic": mean_a,
        "analytic_ci95_lo": lo_a,
        "analytic_ci95_hi": hi_a,
        "within_prompt_cv_mean": float(np.mean(per_prompt_cv)) if per_prompt_cv else None,
        "per_position": pos_aggregates,
        "elapsed_s": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--student-base", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--out", type=Path, default=Path("results/spec_eval_rejection.json"))
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max-drafts", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.student_base)

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
    out: dict = {"prompts": "humaneval-164", "K": args.K, "max_drafts": args.max_drafts, "method": "rejection-sampling"}

    for name, path in runs:
        if name != "teacher_self" and not Path(path).exists() and "/" in path and not path.startswith("Qwen/"):
            print(f"missing {path}, skip")
            continue
        print(f"\n=== {name} ===", flush=True)
        out[name] = evaluate_one(
            name, path, teacher, tok, prompts,
            K=args.K, max_drafts=args.max_drafts,
            device=device, eval_seed=args.eval_seed,
        )
        r = out[name]
        print(f"  sampled mean run: {r['mean_accepted_run_length']:.3f} / {args.K}  CI [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]")
        print(f"  analytic mean:    {r['mean_analytic']:.3f}            CI [{r['analytic_ci95_lo']:.3f}, {r['analytic_ci95_hi']:.3f}]")
        if r['within_prompt_cv_mean'] is not None:
            print(f"  within-prompt CV (sampled): {r['within_prompt_cv_mean']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
