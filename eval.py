"""
First-pass eval. Each student is scored on:
  - HumanEval pass@1
  - held-out NLL on the val corpus
  - speculative-decoding draft acceptance length (student drafts, teacher
    verifies). Per Leviathan et al. (2023, arXiv:2211.17192), the
    probability of accepting a drafted token x is min(1, p_T(x)/p_S(x)),
    and expected accepted run length is the wall-clock-relevant quantity
    for inference-cost-bound code models like Mellum.

The hardened spec-decode eval (164 prompts, bootstrap CIs, per-position
acceptance) lives in spec_eval.py; this script's spec-decode column is
kept for reference because it's what the report's first-pass section
walks through.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from distill import shift_logits_targets


# -------- HumanEval --------

STOP_SEQUENCES = ["\nclass", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```"]


def truncate_at_stops(text: str) -> str:
    """The model often keeps generating after the requested function. Cut at
    the first top-level structural break that signals the next thing in a
    file. The list above is what HumanEval-style harnesses standardise on."""
    earliest = len(text)
    for s in STOP_SEQUENCES:
        idx = text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


def run_one_humaneval(prompt: str, completion: str, test: str, entry: str) -> bool:
    """Execute prompt + completion + test in a subprocess with a hard timeout.

    Generated code is untrusted, so the subprocess runs with python -I
    (isolated mode: no PYTHON* env, no user site-packages, no implicit cwd
    on sys.path) inside a fresh tempdir, with a fixed PYTHONHASHSEED so
    HumanEval's check() is deterministic. This isn't a real sandbox -- the
    test program can still touch the filesystem and the network -- but it
    keeps stray files out of the repo and stops the program from reaching
    sibling modules in this directory."""
    program = prompt + completion + "\n" + test + f"\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as td:
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        try:
            out = subprocess.run(
                [sys.executable, "-I", "-c", program],
                capture_output=True,
                timeout=8,
                text=True,
                cwd=td,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False
    return out.returncode == 0


@torch.no_grad()
def humaneval(
    model: torch.nn.Module,
    tok,
    n_problems: int,
    max_new: int,
    temperature: float,
    device: torch.device,
) -> dict:
    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(n_problems, len(ds))))
    n_pass = 0
    per_problem: list[dict] = []
    t0 = time.time()
    for ex in ds:
        prompt = ex["prompt"]
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-6),
            top_k=0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
        gen = truncate_at_stops(gen)
        passed = run_one_humaneval(prompt, gen, ex["test"], ex["entry_point"])
        n_pass += int(passed)
        per_problem.append(
            {"task_id": ex["task_id"], "passed": passed, "completion": gen}
        )
    return {
        "n": len(per_problem),
        "pass@1": n_pass / len(per_problem),
        "wallclock_s": time.time() - t0,
        "per_problem": per_problem,
    }


# -------- Held-out NLL --------

@torch.no_grad()
def held_out_nll(model: torch.nn.Module, val: torch.Tensor, pad_id: int, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    for i in range(0, val.size(0), 4):
        b = val[i : i + 4].to(device)
        out = model(b)
        sl, tg, mask = shift_logits_targets(out.logits, b, pad_id)
        flat = sl.view(-1, sl.size(-1))
        tgt = tg.view(-1)
        losses = F.cross_entropy(flat, tgt, reduction="none").view_as(tg)
        total_loss += (losses * mask).sum().item()
        total_tokens += mask.sum().item()
    return total_loss / max(total_tokens, 1.0)


# -------- Speculative-decoding draft acceptance --------

@torch.no_grad()
def spec_decode(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    prompts: list[str],
    tok,
    draft_len: int,
    max_drafts: int,
    temperature: float,
    device: torch.device,
) -> dict:
    student.eval()
    teacher.eval()
    n_seqs = len(prompts)
    accept_lens: list[float] = []
    accept_probs: list[float] = []

    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
        runs_for_seq: list[float] = []
        cur = ids
        for _ in range(max_drafts):
            # student drafts draft_len tokens autoregressively
            draft = student.generate(
                cur,
                max_new_tokens=draft_len,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
            # we score the drafted tokens at positions [cur.len-1 .. cur.len + draft_len - 2]
            # (next-token predictions are made from positions one before the target)
            s_logits = student(draft).logits
            t_logits = teacher(draft).logits
            # softmax at temperature 1.0 for the spec-decode rule (paper rule)
            p_s = F.softmax(s_logits / max(temperature, 1e-6), dim=-1)
            p_t = F.softmax(t_logits / max(temperature, 1e-6), dim=-1)

            # if generate stopped early (EOS), only score actually-generated tokens
            actual_drafted = draft.size(1) - cur.size(1)
            start = cur.size(1) - 1
            tgt_positions = list(range(start, start + actual_drafted))
            if not tgt_positions:
                break
            ratios = []
            for pos in tgt_positions:
                tok_id = draft[0, pos + 1].item()
                ps = p_s[0, pos, tok_id].item()
                pt = p_t[0, pos, tok_id].item()
                if ps < 1e-9:
                    ratio = 0.0
                else:
                    ratio = min(1.0, pt / ps)
                ratios.append(ratio)
            # expected accepted run length = sum_{i} prod_{j<=i} a_j
            cum = 1.0
            run = 0.0
            for r in ratios:
                cum *= r
                run += cum
            runs_for_seq.append(run)
            accept_probs.extend(ratios)
            # advance: keep all tokens up to first rejection (in expectation, run length)
            advance = max(1, int(round(run)))
            cur = draft[:, : cur.size(1) + min(advance, draft_len)]
            if cur.size(1) >= 256:
                break
        accept_lens.extend(runs_for_seq)

    return {
        "draft_len": draft_len,
        "n_seqs": n_seqs,
        "n_drafts": len(accept_lens),
        "mean_accepted_run_length": sum(accept_lens) / max(len(accept_lens), 1),
        "mean_per_token_accept_prob": sum(accept_probs) / max(len(accept_probs), 1),
        "max_possible_run": draft_len,
    }


# -------- Driver --------

def evaluate_one(
    name: str,
    student_path: Path,
    teacher_path: Path,
    val: torch.Tensor,
    tok,
    device: torch.device,
    args,
) -> dict:
    print(f"\n=== {name} ===")
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.bfloat16).to(device)
    student.eval()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll = held_out_nll(student, val, pad_id, device)
    print(f"  held-out NLL: {nll:.4f}")

    he = humaneval(
        student, tok,
        n_problems=args.he_problems,
        max_new=args.he_max_new,
        temperature=args.he_temperature,
        device=device,
    )
    print(f"  HumanEval pass@1 (n={he['n']}, T={args.he_temperature}): {he['pass@1']:.3f}  ({he['wallclock_s']:.0f}s)")

    # spec-decode needs teacher; load it once per call. Could be optimised
    # to share across runs but four reloads on a 4090 is ~10s total.
    teacher = AutoModelForCausalLM.from_pretrained(teacher_path, dtype=torch.bfloat16).to(device)
    teacher.eval()
    prompts_he = [ex["prompt"] for ex in load_dataset("openai_humaneval", split="test").select(range(min(32, args.he_problems)))]
    # Symmetric max_drafts across K so the per-prompt sample size is the same.
    # The original first-pass eval used max_drafts=4 for K=2 and 2 for K=4,
    # which gave the K=2 column twice as many cycles per prompt and was part
    # of why the K=2 / K=4 signs disagreed. The hardened eval in spec_eval.py
    # is what the report leans on; this script kept for historical compare.
    sd2 = spec_decode(student, teacher, prompts_he, tok, draft_len=2, max_drafts=4, temperature=1.0, device=device)
    sd4 = spec_decode(student, teacher, prompts_he, tok, draft_len=4, max_drafts=4, temperature=1.0, device=device)
    print(f"  spec-decode K=2 mean run: {sd2['mean_accepted_run_length']:.3f} / 2")
    print(f"  spec-decode K=4 mean run: {sd4['mean_accepted_run_length']:.3f} / 4")

    del student, teacher
    torch.cuda.empty_cache()
    return {
        "name": name,
        "ckpt": str(student_path),
        "held_out_nll": nll,
        "humaneval": {k: v for k, v in he.items() if k != "per_problem"},
        "humaneval_completions": he["per_problem"],
        "spec_decode_K2": sd2,
        "spec_decode_K4": sd4,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--student-base", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--val-pt", type=Path, default=Path("data/cache/val.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/eval.json"))
    ap.add_argument("--he-problems", type=int, default=164)
    ap.add_argument("--he-max-new", type=int, default=256)
    ap.add_argument("--he-temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    val = torch.load(args.val_pt)

    runs = [
        ("teacher", args.teacher),
        ("student_base", args.student_base),
        ("student_ce", args.ckpt_dir / "student_ce"),
        ("student_fkl", args.ckpt_dir / "student_fkl"),
        ("student_rkl", args.ckpt_dir / "student_rkl"),
        ("student_gkd", args.ckpt_dir / "student_gkd"),
    ]

    out: dict = {}
    for name, path in runs:
        if not Path(path).exists():
            print(f"missing {path}, skip")
            continue
        # for the teacher itself we still want held-out NLL + HumanEval, but
        # spec-decode comparing teacher to itself would always be 1.0.
        out[name] = evaluate_one(name, Path(path), Path(args.teacher), val, tok, device, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
