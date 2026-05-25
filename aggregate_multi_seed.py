#!/usr/bin/env python3
"""
Aggregate Multi-Seed Cat.4 Results
==================================
Takes the per-model JSON files emitted by multi_seed_cat4.py and
produces a per-system table with bootstrap 95% CIs averaged across
models, plus the cross-model dispersion (the value currently reported
in Table 7) for direct comparison.

Output: ./tables/multi_seed_cat4_table.tex (LaTeX) plus a JSON
summary.

Usage:
    python aggregate_multi_seed.py \
        --inputs ./results_multi_seed/gemma3_27b.json \
                 ./results_multi_seed/deepseek_r1_32b.json \
                 ./results_multi_seed/qwen3_32b.json \
        --output-tex ./tables/multi_seed_cat4_table.tex \
        --output-json ./tables/multi_seed_cat4_summary.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--output-tex", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    # by_system_model[system][model_name] = {mean, lo, hi}
    by_system_model = defaultdict(dict)
    for in_path in args.inputs:
        data = json.loads(in_path.read_text())
        model = data["model"].replace(":", "_").replace("-", "_")
        for system, s in data["by_system"].items():
            by_system_model[system][model] = s

    # Aggregate across models (per-system).
    rows = []
    for system, per_model in sorted(by_system_model.items()):
        means = [v["mean_em"] for v in per_model.values()]
        if not means:
            continue
        n = len(means)
        mean_of_means = sum(means) / n
        # Cross-model std (sample std with n-1).
        if n > 1:
            var = sum((m - mean_of_means) ** 2 for m in means) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        # Aggregate within-model 95% CI by taking the max of the model
        # CI half-widths -- a conservative "envelope" CI.
        envelope_lo = min(v["ci95_lo"] for v in per_model.values())
        envelope_hi = max(v["ci95_hi"] for v in per_model.values())
        rows.append({
            "system": system,
            "n_models": n,
            "mean_em": round(mean_of_means, 4),
            "cross_model_std": round(std, 4),
            "envelope_ci95_lo": round(envelope_lo, 4),
            "envelope_ci95_hi": round(envelope_hi, 4),
            "per_model": per_model,
        })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2))

    # LaTeX table
    sys_order = ["eskimo", "sudanese", "hawaiian", "iroquois",
                 "dravidian", "crow", "omaha"]
    by_system = {r["system"]: r for r in rows}

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cat.4 multi-seed evaluation (temperature 0.7, "
        "n=5 samples per question, stratified subsample). "
        "Mean EM across 3 open-source models with bootstrap 95\\% "
        "envelope CI; cross-model std reported for comparison with "
        "Table~\\ref{tab:by_system}.}",
        "\\label{tab:multi_seed_cat4}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{System} & \\textbf{EM \\% (95\\% CI)} & \\textbf{$\\pm$ cross-model} \\\\",
        "\\midrule",
    ]
    for system in sys_order:
        if system not in by_system:
            continue
        r = by_system[system]
        em = 100 * r["mean_em"]
        lo = 100 * r["envelope_ci95_lo"]
        hi = 100 * r["envelope_ci95_hi"]
        std = 100 * r["cross_model_std"]
        lines.append(
            f"{system.capitalize()} & {em:.1f} "
            f"[{lo:.1f}, {hi:.1f}] & $\\pm${std:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote LaTeX table to {args.output_tex}")
    print(f"Wrote JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
