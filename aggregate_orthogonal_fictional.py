#!/usr/bin/env python3
"""
Decompose the +5.1pp Cat.4 fictional-label gain into system-name vs kin-term
contributions using a 2x2 orthogonal ablation.

Conditions (per <model>):
  - real          : results_<model>_zero_shot_direct/
  - fictional     : results_fictional_<model>_zero_shot_direct/
  - system_only   : results_fictional_system_only_<model>_zero_shot_direct/
  - term_only     : results_fictional_term_only_<model>_zero_shot_direct/

Restricted to Cat.4 questions on the five non-Western kinship systems
(Hawaiian, Iroquois, Dravidian, Crow, Omaha).

Outputs (in familytree_v6/):
  - orthogonal_fictional_ablation.json : full per-cell numbers
  - orthogonal_fictional_table.tex     : LaTeX appendix table

Stdlib only.
"""

import json
import os
import random
from statistics import mean

BASE = os.path.dirname(os.path.abspath(__file__))

MODELS = ["gemma3_27b", "deepseek_r1_32b", "qwen3_32b"]
SYSTEMS = ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]
CONDITIONS = {
    "real":        "results_{model}_zero_shot_direct",
    "fictional":   "results_fictional_{model}_zero_shot_direct",
    "system_only": "results_fictional_system_only_{model}_zero_shot_direct",
    "term_only":   "results_fictional_term_only_{model}_zero_shot_direct",
}


def load_cat4_em(model, condition_template, system):
    """Return list of 0/1 exact_match values for Cat.4 questions."""
    path = os.path.join(BASE,
                        condition_template.format(model=model),
                        f"{system}_results.json")
    with open(path) as f:
        d = json.load(f)
    qs = d["questions"]
    cat4 = [q for q in qs if q["category"] == 4]
    return [int(bool(q["exact_match"])) for q in cat4]


def percentile(sorted_vals, p):
    """Percentile via linear interpolation on a pre-sorted list (0<=p<=100)."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def bootstrap_ci(values, n_boot=10000, seed=42):
    """Percentile bootstrap 95% CI of the mean over `values` (resample cells)."""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return percentile(means, 2.5), percentile(means, 97.5)


def main():
    cells = []  # one entry per (model, system) cell
    for model in MODELS:
        for system in SYSTEMS:
            cell = {"model": model, "system": system}
            for cond_name, tmpl in CONDITIONS.items():
                ems = load_cat4_em(model, tmpl, system)
                cell[f"{cond_name}_n"] = len(ems)
                cell[f"{cond_name}_em"] = mean(ems) if ems else float("nan")
            cell["delta_full"] = cell["fictional_em"]   - cell["real_em"]
            cell["delta_sys"]  = cell["system_only_em"] - cell["real_em"]
            cell["delta_term"] = cell["term_only_em"]   - cell["real_em"]
            cell["sum_sys_term"] = cell["delta_sys"] + cell["delta_term"]
            cell["interaction"]  = cell["delta_full"] - cell["sum_sys_term"]
            cells.append(cell)

    # Aggregate across the 15 cells
    real_means      = [c["real_em"]        for c in cells]
    fictional_means = [c["fictional_em"]   for c in cells]
    sysonly_means   = [c["system_only_em"] for c in cells]
    termonly_means  = [c["term_only_em"]   for c in cells]

    delta_full = [c["delta_full"] for c in cells]
    delta_sys  = [c["delta_sys"]  for c in cells]
    delta_term = [c["delta_term"] for c in cells]
    interactions = [c["interaction"] for c in cells]

    summary = {
        "n_cells": len(cells),
        "real_mean":        mean(real_means),
        "fictional_mean":   mean(fictional_means),
        "system_only_mean": mean(sysonly_means),
        "term_only_mean":   mean(termonly_means),
        "delta_full_mean":  mean(delta_full),
        "delta_sys_mean":   mean(delta_sys),
        "delta_term_mean":  mean(delta_term),
        "mean_interaction": mean(interactions),
        "delta_full_ci95":  bootstrap_ci(delta_full),
        "delta_sys_ci95":   bootstrap_ci(delta_sys),
        "delta_term_ci95":  bootstrap_ci(delta_term),
    }

    out = {
        "description": (
            "Orthogonal 2x2 decomposition of fictional-label Cat.4 gain "
            "into system-name swap vs kin-term swap. Restricted to the five "
            "non-Western kinship systems (Hawaiian, Iroquois, Dravidian, Crow, "
            "Omaha) across 3 open-source models."
        ),
        "models": MODELS,
        "systems": SYSTEMS,
        "conditions": CONDITIONS,
        "cells": cells,
        "summary": summary,
    }
    json_path = os.path.join(BASE, "orthogonal_fictional_ablation.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {json_path}")

    # ----- Build LaTeX table -----
    def pct(x):
        return f"{100 * x:.1f}"

    def signed_pct(x):
        return f"{100 * x:+.1f}"

    def ci_pct(lo, hi):
        return f"[{100*lo:+.1f}, {100*hi:+.1f}]"

    real_pct = pct(summary["real_mean"])
    rows = [
        ("Real (baseline)",    real_pct, "--", "--"),
        ("+ system-name swap", pct(summary["system_only_mean"]),
         signed_pct(summary["delta_sys_mean"]),
         ci_pct(*summary["delta_sys_ci95"])),
        ("+ kin-term swap",    pct(summary["term_only_mean"]),
         signed_pct(summary["delta_term_mean"]),
         ci_pct(*summary["delta_term_ci95"])),
        ("+ both (full)",      pct(summary["fictional_mean"]),
         signed_pct(summary["delta_full_mean"]),
         ci_pct(*summary["delta_full_ci95"])),
    ]

    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Decomposing the fictional-label gain (Cat.4, "
                 r"Other-5 systems, mean across 3 open-source models). "
                 r"$\Delta$ vs.\ real-label baseline. 95\% bootstrap CI "
                 r"over 15 (system, model) cells.}")
    latex.append(r"\label{tab:fictional_orthogonal}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\toprule")
    latex.append(r"Condition & Real EM & $\Delta$ vs. Real & 95\% CI \\")
    latex.append(r"\midrule")
    for name, em, delta, ci in rows:
        latex.append(f"{name:<20s} & {em} & {delta} & {ci} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex_str = "\n".join(latex) + "\n"

    tex_path = os.path.join(BASE, "orthogonal_fictional_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"Wrote {tex_path}")

    # ----- Console report -----
    print("\n=== Per-cell deltas (pp) ===")
    print(f"{'model':<20s}{'system':<12s}{'real':>7s}{'sys':>8s}"
          f"{'term':>8s}{'full':>8s}{'Δsys':>8s}{'Δterm':>8s}"
          f"{'Δfull':>8s}{'inter':>8s}")
    for c in cells:
        print(f"{c['model']:<20s}{c['system']:<12s}"
              f"{100*c['real_em']:7.1f}{100*c['system_only_em']:8.1f}"
              f"{100*c['term_only_em']:8.1f}{100*c['fictional_em']:8.1f}"
              f"{100*c['delta_sys']:+8.1f}{100*c['delta_term']:+8.1f}"
              f"{100*c['delta_full']:+8.1f}{100*c['interaction']:+8.1f}")

    print("\n=== Summary across 15 cells ===")
    print(f"real            : {100*summary['real_mean']:.1f}")
    print(f"system_only     : {100*summary['system_only_mean']:.1f}  "
          f"Δsys  = {100*summary['delta_sys_mean']:+.2f}  "
          f"CI95 = [{100*summary['delta_sys_ci95'][0]:+.2f}, "
          f"{100*summary['delta_sys_ci95'][1]:+.2f}]")
    print(f"term_only       : {100*summary['term_only_mean']:.1f}  "
          f"Δterm = {100*summary['delta_term_mean']:+.2f}  "
          f"CI95 = [{100*summary['delta_term_ci95'][0]:+.2f}, "
          f"{100*summary['delta_term_ci95'][1]:+.2f}]")
    print(f"fictional (full): {100*summary['fictional_mean']:.1f}  "
          f"Δfull = {100*summary['delta_full_mean']:+.2f}  "
          f"CI95 = [{100*summary['delta_full_ci95'][0]:+.2f}, "
          f"{100*summary['delta_full_ci95'][1]:+.2f}]")
    print(f"mean interaction (Δfull − (Δsys+Δterm)) = "
          f"{100*summary['mean_interaction']:+.2f} pp")


if __name__ == "__main__":
    main()
