#!/usr/bin/env python3
"""
Aggregate 2x2 Fictional-Substitution Ablation
==============================================
Decomposes the full-fictional Cat.4 effect into its two factors:

                    | real kin term      | fictional kin term
  ------------------+--------------------+----------------------
  real system name  | results_<m>_       | results_fictional_
                    |   zero_shot_direct |   term_only_<m>_zsd
  ------------------+--------------------+----------------------
  fic. system name  | results_fictional_ | results_fictional_<m>_
                    | system_only_<m>_zsd|   zero_shot_direct

Per-model, per-system accuracies (Cat.4 only) are extracted from each
of the four results dirs, and three contrasts are computed:

  Δ_system  = system_only - real          (effect of system label)
  Δ_term    = term_only   - real          (effect of kin terms)
  Δ_full    = full_fic    - real          (combined effect)
  Δ_super   = Δ_full - (Δ_system + Δ_term) (super-additive interaction)

Outputs
  --output-json   machine-readable per-(model, system) rows
  --output-tex    LaTeX table fragment ready to \input into v4 sec.5
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

COMPARABLE = ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]
MODELS = ["gemma3_27b", "deepseek_r1_32b", "qwen3_32b"]
MODEL_TEX = {
    "gemma3_27b":      "Gemma3-27B",
    "deepseek_r1_32b": "DeepSeek-R1-32B",
    "qwen3_32b":       "Qwen3-32B",
}
CONDITIONS = {
    "real":        "./results_{model}_zero_shot_direct",
    "system_only": "./results_fictional_system_only_{model}_zero_shot_direct",
    "term_only":   "./results_fictional_term_only_{model}_zero_shot_direct",
    "full":        "./results_fictional_{model}_zero_shot_direct",
}


def load_cat4_em(results_dir: Path) -> dict:
    """Return {system: (em_pct, n_total)} for Cat.4 in this dir."""
    out = {}
    for system in COMPARABLE:
        path = results_dir / f"{system}_results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        by_cat = (data.get("summary", {}).get("by_category")
                  or data.get("by_category") or {})
        cat4 = (by_cat.get("cat_4") or by_cat.get("4")
                or by_cat.get(4) or by_cat.get("Category 4") or {})
        if isinstance(cat4, dict) and cat4.get("total"):
            em_frac = cat4.get("exact_match", cat4.get("exact", 0))
            if em_frac > 1:
                em_frac = em_frac / cat4["total"]
            out[system] = (round(100.0 * em_frac, 2), cat4["total"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-tex",  required=True, type=Path)
    args = ap.parse_args()

    # Load all four conditions for each model
    accs = defaultdict(dict)  # accs[condition][model] = {system: (em, n)}
    missing = []
    for cond, tmpl in CONDITIONS.items():
        for m in MODELS:
            d = Path(tmpl.format(model=m))
            if not d.exists():
                missing.append(f"{cond}/{m}: {d}")
                continue
            accs[cond][m] = load_cat4_em(d)

    if missing:
        print("WARNING: results dirs missing -- 2x2 will be partial:")
        for x in missing:
            print(f"  - {x}")

    rows = []
    for m in MODELS:
        for s in COMPARABLE:
            row = {"model": m, "system": s}
            for cond in CONDITIONS:
                v = accs.get(cond, {}).get(m, {}).get(s)
                if v is None:
                    row[f"{cond}_em"] = None
                    row[f"{cond}_n"]  = None
                else:
                    row[f"{cond}_em"] = v[0]
                    row[f"{cond}_n"]  = v[1]
            if all(row[f"{c}_em"] is not None for c in CONDITIONS):
                row["delta_system"] = round(row["system_only_em"] - row["real_em"], 2)
                row["delta_term"]   = round(row["term_only_em"]   - row["real_em"], 2)
                row["delta_full"]   = round(row["full_em"]        - row["real_em"], 2)
                row["delta_super"]  = round(
                    row["delta_full"] - (row["delta_system"] + row["delta_term"]), 2)
            rows.append(row)

    # Per-system macro-mean across models (when complete) + per-model macro-mean across systems
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    macro_by_model = []
    for m in MODELS:
        recs = [r for r in rows if r["model"] == m
                and all(r[f"{c}_em"] is not None for c in CONDITIONS)]
        if not recs:
            continue
        macro_by_model.append({
            "model":           m,
            "n_systems":       len(recs),
            "real_em":         _mean([r["real_em"]        for r in recs]),
            "system_only_em":  _mean([r["system_only_em"] for r in recs]),
            "term_only_em":    _mean([r["term_only_em"]   for r in recs]),
            "full_em":         _mean([r["full_em"]        for r in recs]),
            "delta_system":    _mean([r["delta_system"]   for r in recs]),
            "delta_term":      _mean([r["delta_term"]     for r in recs]),
            "delta_full":      _mean([r["delta_full"]     for r in recs]),
            "delta_super":     _mean([r["delta_super"]    for r in recs]),
        })

    # Grand mean across (model x system)
    full_recs = [r for r in rows
                 if all(r[f"{c}_em"] is not None for c in CONDITIONS)]
    grand = None
    if full_recs:
        grand = {
            "n":              len(full_recs),
            "real_em":        _mean([r["real_em"]        for r in full_recs]),
            "system_only_em": _mean([r["system_only_em"] for r in full_recs]),
            "term_only_em":   _mean([r["term_only_em"]   for r in full_recs]),
            "full_em":        _mean([r["full_em"]        for r in full_recs]),
            "delta_system":   _mean([r["delta_system"]   for r in full_recs]),
            "delta_term":     _mean([r["delta_term"]     for r in full_recs]),
            "delta_full":     _mean([r["delta_full"]     for r in full_recs]),
            "delta_super":    _mean([r["delta_super"]    for r in full_recs]),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "rows":            rows,
        "macro_by_model":  macro_by_model,
        "grand":           grand,
    }, indent=2))
    print(f"Wrote {len(rows)} rows to {args.output_json}")

    # Console pretty-print
    if grand:
        print(f"\nGrand mean across {grand['n']} (model,system) cells:")
        print(f"  real         {grand['real_em']:.2f}%")
        print(f"  +system_only {grand['system_only_em']:.2f}%   (Δ={grand['delta_system']:+.2f})")
        print(f"  +term_only   {grand['term_only_em']:.2f}%   (Δ={grand['delta_term']:+.2f})")
        print(f"  +full fic.   {grand['full_em']:.2f}%   (Δ={grand['delta_full']:+.2f})")
        print(f"  super-add.   Δ={grand['delta_super']:+.2f}")

    # LaTeX table fragment
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Real & +Sys & +Term & +Full & $\Delta_{\text{sys}}$ & $\Delta_{\text{term}}$ & $\Delta_{\text{full}}$\\")
    lines.append(r"\midrule")
    for rec in macro_by_model:
        lines.append(
            f"{MODEL_TEX[rec['model']]} & "
            f"{rec['real_em']:.1f} & "
            f"{rec['system_only_em']:.1f} & "
            f"{rec['term_only_em']:.1f} & "
            f"{rec['full_em']:.1f} & "
            f"{rec['delta_system']:+.1f} & "
            f"{rec['delta_term']:+.1f} & "
            f"{rec['delta_full']:+.1f}\\\\"
        )
    if grand:
        lines.append(r"\midrule")
        lines.append(
            f"\\textit{{Mean}} & "
            f"{grand['real_em']:.1f} & "
            f"{grand['system_only_em']:.1f} & "
            f"{grand['term_only_em']:.1f} & "
            f"{grand['full_em']:.1f} & "
            f"{grand['delta_system']:+.1f} & "
            f"{grand['delta_term']:+.1f} & "
            f"{grand['delta_full']:+.1f}\\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{2$\times$2 ablation of fictional substitution on Cat.\,4 (macro-averaged over the five non-Western systems). \textbf{+Sys} replaces the system label only (e.g.\ \emph{Hawaiian}$\to$\emph{Zorblax}). \textbf{+Term} replaces the cultural kin-term label only (e.g.\ \emph{classificatory mother}$\to$\emph{tier-mater}). \textbf{+Full} replaces both. Positive $\Delta$ values indicate that the fictional condition is easier than the real one.}")
    lines.append(r"\label{tab:fictional-2x2}")
    lines.append(r"\end{table}")

    args.output_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote LaTeX table to {args.output_tex}")


if __name__ == "__main__":
    main()
