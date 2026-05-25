#!/usr/bin/env python3
"""
Summarize Cat.4 by hop depth, using the augmented v6.2 dataset
==============================================================
Reads per-system results from results_4hop_<model>_zero_shot_direct/
and produces a Cat.4-only hop-scaling table for the five non-Western
systems (which now have 2/3/4-hop Cat.4 questions thanks to the
v6.2 augmentation).  Output is a JSON summary plus a LaTeX table
suitable for inclusion in the paper.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

NON_WESTERN = ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]
WESTERN = ["eskimo", "sudanese"]


def load_per_question_predictions(results_dir: Path):
    """Yield (system, n_hops, exact_bool) from saved per-question records."""
    for path in sorted(results_dir.glob("*_results.json")):
        if "combined" in path.stem:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for q in data.get("questions", []):
            if q.get("category") != 4:
                continue
            sys_ = q.get("kinship_system")
            hop = q.get("n_hops")
            ok = bool(q.get("exact_match", q.get("exact", False)))
            yield (sys_, hop, ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir-template", required=True,
                    help="e.g. ./results_4hop_{model}_zero_shot_direct")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--output-tex", type=Path, required=True)
    args = ap.parse_args()

    # (model, system, hop) -> {correct, total}
    counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for model in args.models:
        rdir = Path(args.results_dir_template.format(model=model))
        if not rdir.exists():
            print(f"WARN: missing {rdir}")
            continue
        for sys_, hop, ok in load_per_question_predictions(rdir):
            key = (model, sys_, hop)
            counts[key]["total"] += 1
            if ok:
                counts[key]["correct"] += 1

    # Aggregate per-system per-hop across models (mean of model EMs)
    rows = []
    for system in WESTERN + NON_WESTERN:
        for hop in [2, 3, 4]:
            ems = []
            ns = []
            for model in args.models:
                s = counts.get((model, system, hop))
                if s and s["total"] > 0:
                    ems.append(100 * s["correct"] / s["total"])
                    ns.append(s["total"])
            if not ems:
                continue
            mean_em = sum(ems) / len(ems)
            std_em = (sum((e - mean_em) ** 2 for e in ems) / len(ems)) ** 0.5 if len(ems) > 1 else 0.0
            rows.append({
                "system": system,
                "hop": hop,
                "n_per_model": ns[0] if ns else 0,
                "mean_em": round(mean_em, 2),
                "std_em": round(std_em, 2),
                "per_model_em": [round(e, 2) for e in ems],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} rows to {args.output}")

    # LaTeX table: per-system Cat.4 EM at 2/3/4 hops (Other-5 only)
    by_sys_hop = {(r["system"], r["hop"]): r for r in rows}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cat.4 (cultural override) accuracy by reasoning depth on the v6.2 dataset (with 4-hop overrides; EM \\%, mean $\\pm$ cross-model std across 3 open-source models). The 4-hop column is filled by the augmentation in this paper; in the original v6.1 dataset the 4-hop Cat.4 cell was empty for all five non-Western systems.}",
        "\\label{tab:cat4_4hop}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{System} & \\textbf{2-hop} & \\textbf{3-hop} & \\textbf{4-hop} \\\\",
        "\\midrule",
    ]
    for system in NON_WESTERN:
        cells = []
        for hop in [2, 3, 4]:
            r = by_sys_hop.get((system, hop))
            if r is None:
                cells.append("--")
            else:
                cells.append(f"{r['mean_em']:.1f} $\\pm$ {r['std_em']:.1f}")
        lines.append(f"{system.capitalize()} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote LaTeX table to {args.output_tex}")

    # Also print a quick console table
    print()
    print(f"{'System':<10} {'2-hop':>14} {'3-hop':>14} {'4-hop':>14}")
    print("-" * 56)
    for system in WESTERN + NON_WESTERN:
        cells = []
        for hop in [2, 3, 4]:
            r = by_sys_hop.get((system, hop))
            if r is None:
                cells.append(f"{'--':>14}")
            else:
                cells.append(f"{r['mean_em']:>5.1f} ± {r['std_em']:>4.1f} ")
        print(f"{system:<10} {cells[0]} {cells[1]} {cells[2]}")


if __name__ == "__main__":
    main()
