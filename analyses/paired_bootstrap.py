"""
Paired bootstrap on per-prompt deltas (rkl-ce, gkd-ce, fkl-ce).

The README/report headline used to claim "CIs don't overlap therefore
significant at 95%". That's not a paired test. Per-prompt RNG is shared
across variants in spec_eval.py:168-170 (`rng_seed = eval_seed *
1_000_003 + i`), so the data does support a real paired test.

Two paths:

(a) If `results/spec_eval.json` has per-prompt arrays
(`per_prompt_mean_run`), do the actual paired bootstrap: for each of
n_boot draws, sample 164 prompt indices with replacement, take
delta_b[i] = a[i] - b[i] on that resample, store its mean, percentile
the resulting distribution.

(b) If the per-prompt arrays aren't persisted (current spec_eval.py
only stores aggregates), fall back to the upper bound
var(delta) <= var(a) + var(b). The marginal half-widths recover sigma
via the normal-approx (half-width ~= 1.96 * sigma / sqrt(n)), and the
paired CI's worst-case half-width is sqrt(sigma_a^2 + sigma_b^2) *
1.96 / sqrt(n). This is *conservative* -- the real paired CI is
narrower because per-prompt deltas are positively correlated by the
shared eval seed.

Run:
    python analyses/paired_bootstrap.py
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


N_BOOT = 10000
CI = 0.95
N_PROMPTS = 164  # all 164 humaneval prompts, hardcoded in spec_eval.py


def have_per_prompt(d: dict, name: str) -> bool:
    r = d.get(name, {})
    return isinstance(r.get("per_prompt_mean_run"), list) and len(r["per_prompt_mean_run"]) > 0


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = N_BOOT) -> dict:
    """Paired percentile bootstrap on mean(a - b)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    assert a_arr.shape == b_arr.shape, "paired arrays must have same length"
    n = a_arr.size
    rng = np.random.default_rng(0)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = (a_arr[idx] - b_arr[idx]).mean()
    point = float((a_arr - b_arr).mean())
    lo = float(np.quantile(deltas, (1 - CI) / 2))
    hi = float(np.quantile(deltas, 1 - (1 - CI) / 2))
    return {
        "method": "paired_bootstrap",
        "n_prompts": n,
        "delta_mean": point,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "n_boot": n_boot,
    }


def upper_bound_paired_ci(
    a_mean: float, a_lo: float, a_hi: float,
    b_mean: float, b_lo: float, b_hi: float,
    n: int = N_PROMPTS,
) -> dict:
    """
    Conservative paired-CI from marginals only.

    Per-prompt sigma recovered from CI half-width assuming normal-approx:
        half_width = 1.96 * sigma / sqrt(n)  =>  sigma = half_width * sqrt(n) / 1.96

    Worst-case paired-delta sigma when the two arms are independent:
        sigma_delta <= sqrt(sigma_a^2 + sigma_b^2)

    The actual paired CI is narrower (per-prompt RNG is shared, so deltas
    are positively correlated), but this gives an honest upper bound when
    per-prompt arrays aren't persisted.
    """
    z = 1.959963984540054  # 95% normal quantile
    half_a = (a_hi - a_lo) / 2.0
    half_b = (b_hi - b_lo) / 2.0
    sigma_a = half_a * math.sqrt(n) / z
    sigma_b = half_b * math.sqrt(n) / z
    sigma_delta = math.sqrt(sigma_a * sigma_a + sigma_b * sigma_b)
    half_delta = z * sigma_delta / math.sqrt(n)
    delta_mean = a_mean - b_mean
    return {
        "method": "marginal_upper_bound",
        "n_prompts": n,
        "delta_mean": delta_mean,
        "ci95_lo": delta_mean - half_delta,
        "ci95_hi": delta_mean + half_delta,
        "sigma_a": sigma_a,
        "sigma_b": sigma_b,
        "sigma_delta_upper": sigma_delta,
    }


def main() -> None:
    spec = json.loads(Path("results/spec_eval.json").read_text())

    pairs = [
        ("student_rkl", "student_ce"),
        ("student_gkd", "student_ce"),
        ("student_fkl", "student_ce"),
    ]

    have_arrays = all(have_per_prompt(spec, p[0]) and have_per_prompt(spec, p[1]) for p in pairs)

    print(f"Paired bootstrap on per-prompt spec-decode deltas, K=4, n={N_PROMPTS}.")
    print(f"Per-prompt arrays in JSON: {have_arrays}")
    print()

    rows = []
    for a_name, b_name in pairs:
        if have_arrays:
            r = paired_bootstrap(
                spec[a_name]["per_prompt_mean_run"],
                spec[b_name]["per_prompt_mean_run"],
            )
        else:
            r = upper_bound_paired_ci(
                spec[a_name]["mean_accepted_run_length"],
                spec[a_name]["ci95_lo"],
                spec[a_name]["ci95_hi"],
                spec[b_name]["mean_accepted_run_length"],
                spec[b_name]["ci95_lo"],
                spec[b_name]["ci95_hi"],
            )
        r["pair"] = f"{a_name} - {b_name}"
        rows.append(r)

    label = "paired bootstrap" if have_arrays else "marginal-CI upper bound"
    print(f"{'pair':<32}  {'delta':>7}  {'95% CI':>20}  test")
    print("-" * 80)
    for r in rows:
        sig = "lo > 0" if r["ci95_lo"] > 0 else "crosses 0"
        ci_str = f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]"
        print(f"{r['pair']:<32}  {r['delta_mean']:>+7.3f}  {ci_str:>20}  {sig}")
    print()
    print(f"Method: {label}.")
    if not have_arrays:
        print("Upper-bound CI assumes independent arms; the real paired CI is")
        print("narrower because per-prompt RNG is shared across variants.")

    out = {
        "method": "paired_bootstrap" if have_arrays else "marginal_upper_bound",
        "n_prompts": N_PROMPTS,
        "n_boot": N_BOOT if have_arrays else None,
        "pairs": rows,
    }
    Path("results/paired_bootstrap.json").write_text(json.dumps(out, indent=2))
    print()
    print("wrote results/paired_bootstrap.json")


if __name__ == "__main__":
    main()
