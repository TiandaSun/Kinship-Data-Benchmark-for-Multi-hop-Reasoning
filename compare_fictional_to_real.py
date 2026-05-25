#!/usr/bin/env python3
"""
Compare Real vs Fictional Cat.4 Accuracy
========================================
Reads per-system Cat.4 accuracies from the original protocol-matrix
results (results_<model>_zero_shot_direct/) and from the fictional-
control results (results_fictional_<model>_zero_shot_direct/) and
emits a side-by-side table.

If accuracies are similar across the two conditions, the failure on
Cat.4 is best characterised as in-context rule-application difficulty,
not an absence of cultural knowledge -- supporting the paper's framing
in the Discussion.
"""

import argparse
import json
from pathlib import Path

# Only the five non-Western systems were relabelled.
COMPARABLE = ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]


def load_cat4_accuracy(results_dir: Path) -> dict:
    """Return {system: cat4_em%} for each per-system results file."""
    out = {}
    for system in COMPARABLE:
        path = results_dir / f"{system}_results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        by_cat = data.get("summary", {}).get("by_category") or data.get("by_category") or {}
        cat4 = (by_cat.get("cat_4") or by_cat.get("4")
                or by_cat.get(4) or by_cat.get("Category 4") or {})
        if isinstance(cat4, dict) and cat4.get("total"):
            # llm_tester_v6 stores exact_match as a fraction in [0,1].
            em_frac = cat4.get("exact_match", cat4.get("exact", 0))
            if em_frac > 1:  # legacy: stored as count
                em_frac = em_frac / cat4["total"]
            em = 100.0 * em_frac
            out[system] = {"em": round(em, 2),
                           "exact_match": round(em_frac, 4),
                           "total": cat4["total"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir-template", required=True,
                    help="e.g. ./results_{model}_zero_shot_direct")
    ap.add_argument("--fictional-dir-template", required=True,
                    help="e.g. ./results_fictional_{model}_zero_shot_direct")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    for model in args.models:
        real_dir = Path(args.real_dir_template.format(model=model))
        fic_dir = Path(args.fictional_dir_template.format(model=model))
        real = load_cat4_accuracy(real_dir)
        fic = load_cat4_accuracy(fic_dir)
        for system in COMPARABLE:
            if system in real and system in fic:
                rows.append({
                    "model": model,
                    "system": system,
                    "real_em": real[system]["em"],
                    "fictional_em": fic[system]["em"],
                    "delta": round(fic[system]["em"] - real[system]["em"], 2),
                    "n_real": real[system]["total"],
                    "n_fictional": fic[system]["total"],
                    "real_em_frac": real[system]["exact_match"],
                    "fictional_em_frac": fic[system]["exact_match"],
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} comparison rows to {args.output}\n")

    # Pretty-print
    if rows:
        print(f"{'model':<18} {'system':<10} {'real EM':>9} {'fic EM':>9} {'Δ':>7}")
        print("-" * 60)
        for r in rows:
            print(f"{r['model']:<18} {r['system']:<10} "
                  f"{r['real_em']:>8.1f}% {r['fictional_em']:>8.1f}% "
                  f"{r['delta']:>+7.1f}")


if __name__ == "__main__":
    main()
