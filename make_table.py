"""Pretty-print results/eval.json as a markdown table.
Run after eval.py. The table goes into the report by hand."""

from __future__ import annotations
import json
from pathlib import Path


def main() -> None:
    p = Path("results/eval.json")
    if not p.exists():
        raise SystemExit("run eval.py first")
    data = json.loads(p.read_text())
    rows = []
    for name, r in data.items():
        rows.append({
            "name": name,
            "nll": r.get("held_out_nll"),
            "he": r.get("humaneval", {}).get("pass@1"),
            "sd2": r.get("spec_decode_K2", {}).get("mean_accepted_run_length"),
            "sd4": r.get("spec_decode_K4", {}).get("mean_accepted_run_length"),
        })

    print(f"{'run':<22} {'NLL':>7} {'HE@1':>8} {'spec K=2':>10} {'spec K=4':>10}")
    print("-" * 60)
    for r in rows:
        nll = f"{r['nll']:.4f}" if r["nll"] is not None else "-"
        he = f"{r['he']:.3f}" if r["he"] is not None else "-"
        sd2 = f"{r['sd2']:.3f}" if r["sd2"] is not None else "-"
        sd4 = f"{r['sd4']:.3f}" if r["sd4"] is not None else "-"
        print(f"{r['name']:<22} {nll:>7} {he:>8} {sd2:>10} {sd4:>10}")

    print("\nmarkdown:\n")
    print(f"| run | held-out NLL | HumanEval pass@1 | spec-decode K=2 | spec-decode K=4 |")
    print(f"|---|---|---|---|---|")
    for r in rows:
        nll = f"{r['nll']:.4f}" if r["nll"] is not None else "-"
        he = f"{r['he']:.3f}" if r["he"] is not None else "-"
        sd2 = f"{r['sd2']:.3f} / 2" if r["sd2"] is not None else "-"
        sd4 = f"{r['sd4']:.3f} / 4" if r["sd4"] is not None else "-"
        print(f"| {r['name']} | {nll} | {he} | {sd2} | {sd4} |")


if __name__ == "__main__":
    main()
