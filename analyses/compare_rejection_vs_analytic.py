"""
Side-by-side: bit-exact rejection sampling vs the analytic per-block expectation.

`spec_eval.py` reports E[L] = sum_i prod_{j<=i} a_j (closed-form per-block
expected accept-run length under the rejection chain). `spec_eval_rejection.py`
runs the same chain and rolls a Bernoulli per draft position. The two are the
same quantity in expectation, the second is sample-noisier.

If the analytic shortcut is faithful, the two columns should agree to within
the rejection sampling's own bootstrap CI. If they don't, the analytic
estimator was wrong and the headline ranking might shift.
"""
from __future__ import annotations
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ANALYTIC = ROOT / "results" / "spec_eval.json"
SAMPLED = ROOT / "results" / "spec_eval_rejection.json"


def main() -> None:
    a = json.loads(ANALYTIC.read_text())
    s = json.loads(SAMPLED.read_text())

    rows = ["teacher_self", "student_base", "student_ce", "student_fkl", "student_rkl", "student_gkd"]
    print(f"{'model':<18} {'analytic':>9}  {'sampled':>9}  {'sampled CI':>20}  {'delta':>7}")
    for r in rows:
        if r not in a or r not in s:
            continue
        am = a[r]["mean_accepted_run_length"]
        sm = s[r]["mean_accepted_run_length"]
        slo, shi = s[r]["ci95_lo"], s[r]["ci95_hi"]
        delta = sm - am
        ci = f"[{slo:.3f}, {shi:.3f}]"
        flag = ""
        if abs(delta) > (shi - slo) / 2:
            flag = "  *out-of-CI"
        print(f"{r:<18} {am:>9.3f}  {sm:>9.3f}  {ci:>20}  {delta:>+7.3f}{flag}")


if __name__ == "__main__":
    main()
