"""
Predict spec-decode K=4 from per-position acceptance using the proper
survival-weighted formula (the equal-weight average in the report is an
approximation; this is what the math actually says).

E[accept_run / K] = (1/K) * sum_{i=1..K} prod_{j=1..i} β_j

where β_j is the per-position acceptance probability at draft position j.
β_j is exactly `1 - TV(p_S(. | prefix_j), p_T(. | prefix_j))` per
Leviathan et al. (2023) Corollary 3.6.

I read β_j from `results/spec_eval.json` (per-position acceptance from
the first drafted block, the cleanest place to read it from).
"""

from __future__ import annotations
import json
from pathlib import Path


def survival_weighted(betas: list[float]) -> float:
    """E[accepted run / K]."""
    K = len(betas)
    cum = 1.0
    run = 0.0
    for b in betas:
        cum *= b
        run += cum
    return run / K


def main() -> None:
    spec = json.loads(Path("results/spec_eval.json").read_text())

    print(f"{'method':<14}  {'β1':>6} {'β2':>6} {'β3':>6} {'β4':>6}  {'pred E[run/4]':>14}  {'observed':>9}")
    print("-" * 70)
    for name in ["teacher_self", "student_base", "student_ce",
                 "student_fkl", "student_rkl", "student_gkd"]:
        if name not in spec:
            continue
        per_pos = spec[name]["per_position"]
        betas = []
        for p in per_pos:
            if p["mean"] is None:
                betas.append(0.0)
            else:
                betas.append(p["mean"])
        if len(betas) != 4:
            print(f"{name}: unexpected number of positions: {len(betas)}")
            continue
        pred = survival_weighted(betas)
        obs = spec[name]["mean_accepted_run_length"] / 4.0
        bs = "  ".join(f"{b:.3f}" for b in betas).split("  ")
        print(f"{name:<14}  {betas[0]:>6.3f} {betas[1]:>6.3f} {betas[2]:>6.3f} {betas[3]:>6.3f}  {pred:>14.3f}  {obs:>9.3f}")

    print()
    print("note: prediction uses ONLY first-block per-position acceptance.")
    print("observed run/K is over multiple draft cycles (max_drafts=8 with")
    print("advance heuristic), so prediction systematically overestimates")
    print("because later cycles are conditioned on accepted student prefixes")
    print("which have already drifted slightly from the teacher.")


if __name__ == "__main__":
    main()
