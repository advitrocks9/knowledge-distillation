"""Run hardened spec-decode eval over the seed checkpoints (CE + RKL)
to confirm the rkl > ce gap survives seed variance."""

from __future__ import annotations
from pathlib import Path
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from spec_eval import evaluate_one, load_model


def main() -> None:
    teacher_path = "Qwen/Qwen2.5-Coder-1.5B"
    student_base = "Qwen/Qwen2.5-Coder-0.5B"
    ckpt_dir = Path("checkpoints")
    out_path = Path("results/spec_eval_seeds.json")

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(student_base)
    he = load_dataset("openai_humaneval", split="test")
    prompts = [ex["prompt"] for ex in he]

    teacher = load_model(teacher_path, device)

    runs = [
        ("ce_seed0",  ckpt_dir / "student_ce"),
        ("ce_seed1",  ckpt_dir / "student_ce_seed1"),
        ("ce_seed2",  ckpt_dir / "student_ce_seed2"),
        ("rkl_seed0", ckpt_dir / "student_rkl"),
        ("rkl_seed1", ckpt_dir / "student_rkl_seed1"),
        ("rkl_seed2", ckpt_dir / "student_rkl_seed2"),
    ]
    out: dict = {}
    for name, path in runs:
        if not path.exists():
            print(f"missing {path}, skip")
            continue
        print(f"\n=== {name} ===")
        out[name] = evaluate_one(
            name, str(path), teacher, tok, prompts,
            K=4, max_drafts=8, device=device, eval_seed=42,
        )
        r = out[name]
        print(f"  mean run: {r['mean_accepted_run_length']:.3f}  CI [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
