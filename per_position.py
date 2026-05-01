"""
Per-position analysis on the val corpus.

For each non-pad position in the val tensor, compute the teacher's
Shannon entropy. Bucket positions by entropy. Inside each bucket, for
each student variant, measure:

  agreement   = fraction of positions where argmax(p_S) == argmax(p_T)
  top-1 mass  = mean p_S(argmax(p_T)) -- how much mass the student puts
                on the teacher's preferred token
  total var   = 0.5 * sum |p_S - p_T| -- L1 distance to the teacher
                distribution

The hypothesis the report has been hand-waving at: forward-KL students
match the teacher better at high-entropy positions (where the teacher
spreads mass) and reverse-KL students match the teacher better at
low-entropy positions (where the teacher commits to one token). This
script either confirms or kills that.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def per_position(
    student_path: str,
    teacher: torch.nn.Module,
    val_ids: torch.Tensor,
    pad_id: int,
    device: torch.device,
    batch: int = 4,
) -> dict:
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.bfloat16).to(device)
    student.eval()

    H_buckets: list[torch.Tensor] = []
    agreement_buckets: list[torch.Tensor] = []
    top1_mass_buckets: list[torch.Tensor] = []
    tv_buckets: list[torch.Tensor] = []

    for i in range(0, val_ids.size(0), batch):
        b = val_ids[i : i + batch].to(device)
        sl = student(b).logits[:, :-1, :]
        tl = teacher(b).logits[:, :-1, :]
        # mask: positions where the *next* token isn't pad
        tg = b[:, 1:]
        mask = (tg != pad_id)

        p_t = F.softmax(tl.float(), dim=-1)
        log_t = F.log_softmax(tl.float(), dim=-1)
        H = -(p_t * log_t).sum(dim=-1)  # teacher entropy per position

        p_s = F.softmax(sl.float(), dim=-1)
        argmax_t = p_t.argmax(dim=-1)
        argmax_s = p_s.argmax(dim=-1)
        agreement = (argmax_s == argmax_t).float()
        top1_mass = p_s.gather(-1, argmax_t.unsqueeze(-1)).squeeze(-1)
        tv = 0.5 * (p_s - p_t).abs().sum(dim=-1)

        m = mask
        H_buckets.append(H[m])
        agreement_buckets.append(agreement[m])
        top1_mass_buckets.append(top1_mass[m])
        tv_buckets.append(tv[m])

    H_all = torch.cat(H_buckets)
    agr_all = torch.cat(agreement_buckets)
    top_all = torch.cat(top1_mass_buckets)
    tv_all = torch.cat(tv_buckets)

    # bucket boundaries chosen so each bucket has comparable population on
    # this corpus. teacher entropy is bimodal (low for deterministic syntax
    # positions, high for semantic positions), so quartiles split nicely.
    qs = torch.tensor([0.25, 0.5, 0.75])
    edges = torch.quantile(H_all.cpu(), qs).tolist()
    edges = [0.0] + edges + [float(H_all.max().item()) + 1e-6]
    bucket_names = ["q1_lo_entropy", "q2", "q3", "q4_hi_entropy"]

    out: dict = {"n_positions": int(H_all.numel()), "bucket_edges": edges, "buckets": {}}
    for i, name in enumerate(bucket_names):
        lo, hi = edges[i], edges[i + 1]
        m = (H_all >= lo) & (H_all < hi)
        if m.sum() == 0:
            continue
        out["buckets"][name] = {
            "n": int(m.sum().item()),
            "H_range": [float(lo), float(hi)],
            "H_mean": float(H_all[m].mean().item()),
            "agreement": float(agr_all[m].mean().item()),
            "top1_mass": float(top_all[m].mean().item()),
            "tv_to_teacher": float(tv_all[m].mean().item()),
        }
    out["overall"] = {
        "agreement": float(agr_all.mean().item()),
        "top1_mass": float(top_all.mean().item()),
        "tv_to_teacher": float(tv_all.mean().item()),
    }
    del student
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="/home/prannayk/models/qwen-coder-1.5b")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--student-base", default="/home/prannayk/models/qwen-coder-0.5b")
    ap.add_argument("--val-pt", type=Path, default=Path("data/cache/val.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/per_position.json"))
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.student_base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    val_ids = torch.load(args.val_pt)

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    runs = [
        ("base",  args.student_base),
        ("ce",    str(args.ckpt_dir / "student_ce")),
        ("fkl",   str(args.ckpt_dir / "student_fkl")),
        ("rkl",   str(args.ckpt_dir / "student_rkl")),
        ("gkd",   str(args.ckpt_dir / "student_gkd")),
    ]
    out: dict = {}
    for name, path in runs:
        if not Path(path).exists():
            print(f"missing {path}, skip")
            continue
        print(f"\n=== {name} ===")
        r = per_position(path, teacher, val_ids, pad_id, device)
        out[name] = r
        print(f"  n_positions: {r['n_positions']}")
        print(f"  overall: agreement={r['overall']['agreement']:.3f}  top1_mass={r['overall']['top1_mass']:.3f}  tv={r['overall']['tv_to_teacher']:.3f}")
        for bname, b in r["buckets"].items():
            print(f"  {bname} (H in [{b['H_range'][0]:.2f},{b['H_range'][1]:.2f}), n={b['n']}, H_mean={b['H_mean']:.2f}):")
            print(f"    agreement={b['agreement']:.3f}  top1_mass={b['top1_mass']:.3f}  tv={b['tv_to_teacher']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
