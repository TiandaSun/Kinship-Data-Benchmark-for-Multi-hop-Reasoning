#!/usr/bin/env python3
"""
Analyze Protocol Matrix Results
================================
Produces publication-ready tables from the protocol matrix experiment.
Outputs: LaTeX tables, markdown tables, CSV, and JSON summary.

Usage:
    python analyze_protocol_matrix.py --results-dir . --output-dir ./tables/

Author: Tianda (EMNLP 2026)
"""

import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict


KINSHIP_SYSTEMS = ['eskimo', 'hawaiian', 'iroquois', 'dravidian', 'crow', 'omaha', 'sudanese']
WESTERN = ['eskimo', 'sudanese']
NON_WESTERN = ['hawaiian', 'iroquois', 'dravidian', 'crow', 'omaha']

# Model display names
MODEL_NAMES = {
    "gemma3_27b": "Gemma3-27B",
    "deepseek_r1_32b": "DeepSeek-R1-32B",
    "qwen3_32b": "Qwen3-32B",
    "gpt4o": "GPT-4o-mini",
    "gpt4o_mini": "GPT-4o-mini",
    "gemini": "Gemini-2.5-Flash",
    "gemini_2_5_flash": "Gemini-2.5-Flash",
    "anthropic": "Claude-3.5-Haiku",
    "claude_haiku_4_5": "Claude-Haiku-4.5",
}

PROTOCOL_NAMES = {
    "zero_shot_direct": "Zero-Shot Direct",
    "zero_shot_cot": "Zero-Shot CoT",
    "few_shot_cot": "Few-Shot CoT",
}


def load_results(results_dir, model, protocol):
    """Load combined_results.json for a model/protocol combo."""
    path = Path(results_dir) / f"results_{model}_{protocol}" / "combined_results.json"
    if not path.exists():
        # Try v5 naming (no protocol suffix for zero_shot_direct)
        if protocol == "zero_shot_direct":
            path = Path(results_dir) / f"results_{model}" / "combined_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_extended_results(results_dir, model, protocol):
    """Load extended hop results."""
    path = Path(results_dir) / f"results_extended_{model}_{protocol}" / "combined_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_metrics(data):
    """Compute aggregate metrics from combined results."""
    if not data:
        return None

    systems = list(data.keys())
    n = len(systems)

    # Overall
    overall_em = sum(d["accuracy"]["exact_match"] for d in data.values()) / n * 100

    # Western vs Non-Western
    w = [data[s]["accuracy"]["exact_match"] * 100 for s in WESTERN if s in data]
    nw = [data[s]["accuracy"]["exact_match"] * 100 for s in NON_WESTERN if s in data]
    western = sum(w) / len(w) if w else 0
    non_western = sum(nw) / len(nw) if nw else 0
    gap = western - non_western

    # Cat 4 (non-western only)
    cat4_vals = []
    for s in NON_WESTERN:
        if s in data and "cat_4" in data[s].get("by_category", {}):
            cat4_vals.append(data[s]["by_category"]["cat_4"]["exact_match"] * 100)
    cat4 = sum(cat4_vals) / len(cat4_vals) if cat4_vals else 0

    # By category
    cats = {}
    for cat_key in ["cat_1", "cat_2", "cat_3", "cat_4"]:
        vals = []
        for s in systems:
            if cat_key in data[s].get("by_category", {}):
                vals.append(data[s]["by_category"][cat_key]["exact_match"] * 100)
        cats[cat_key] = sum(vals) / len(vals) if vals else 0

    # By hops
    hops = {}
    for hop_key in ["hop_1", "hop_2", "hop_3", "hop_4", "hop_5", "hop_6"]:
        vals = []
        for s in systems:
            if hop_key in data[s].get("by_hops", {}):
                h = data[s]["by_hops"][hop_key]
                if h["total"] > 0:
                    vals.append(h["exact_match"] * 100)
        hops[hop_key] = sum(vals) / len(vals) if vals else None

    errors = sum(d.get("error_count", 0) for d in data.values())

    return {
        "overall_em": overall_em,
        "western": western,
        "non_western": non_western,
        "gap": gap,
        "cat4": cat4,
        "cats": cats,
        "hops": hops,
        "errors": errors,
    }


def generate_tables(results_dir, output_dir, models, protocols):
    """Generate all analysis tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for model in models:
        for protocol in protocols:
            data = load_results(results_dir, model, protocol)
            if data:
                metrics = compute_metrics(data)
                all_metrics[(model, protocol)] = metrics

    # ---- Table 1: Overall EM by Model × Protocol ----
    print("\n" + "=" * 70)
    print("TABLE 1: Overall Exact Match (%) by Model × Protocol")
    print("=" * 70)

    md_lines = ["| Model | " + " | ".join(PROTOCOL_NAMES.get(p, p) for p in protocols) + " |"]
    md_lines.append("|---|" + "|".join(["---"] * len(protocols)) + "|")

    tex_lines = []
    for model in models:
        row = []
        for protocol in protocols:
            m = all_metrics.get((model, protocol))
            row.append(f"{m['overall_em']:.1f}" if m else "—")
        name = MODEL_NAMES.get(model, model)
        print(f"  {name:<22} " + "  ".join(f"{r:>8}%" for r in row))
        md_lines.append(f"| {name} | " + " | ".join(f"{r}%" for r in row) + " |")
        tex_lines.append(f"  {name} & " + " & ".join(r + "\\%" for r in row) + " \\\\")

    # ---- Table 2: Cat 4 Cultural Override ----
    print("\n" + "=" * 70)
    print("TABLE 2: Cat 4 Cultural Override EM (%)")
    print("=" * 70)

    md2_lines = ["| Model | " + " | ".join(PROTOCOL_NAMES.get(p, p) for p in protocols) + " |"]
    md2_lines.append("|---|" + "|".join(["---"] * len(protocols)) + "|")

    for model in models:
        row = []
        for protocol in protocols:
            m = all_metrics.get((model, protocol))
            row.append(f"{m['cat4']:.1f}" if m else "—")
        name = MODEL_NAMES.get(model, model)
        print(f"  {name:<22} " + "  ".join(f"{r:>8}%" for r in row))
        md2_lines.append(f"| {name} | " + " | ".join(f"{r}%" for r in row) + " |")

    # ---- Table 3: Western vs Non-Western Gap ----
    print("\n" + "=" * 70)
    print("TABLE 3: Western vs Non-Western Gap")
    print("=" * 70)

    for model in models:
        name = MODEL_NAMES.get(model, model)
        for protocol in protocols:
            m = all_metrics.get((model, protocol))
            if m:
                pname = PROTOCOL_NAMES.get(protocol, protocol)
                print(f"  {name:<22} {pname:<20} W={m['western']:.1f}% NW={m['non_western']:.1f}% Gap={m['gap']:.1f}%")

    # ---- Table 4: Hop Scaling ----
    print("\n" + "=" * 70)
    print("TABLE 4: Hop Scaling (Extended Datasets, Zero-Shot Direct)")
    print("=" * 70)

    md4_lines = ["| Model | 1-hop | 2-hop | 3-hop | 4-hop | 5-hop | 6-hop |"]
    md4_lines.append("|---|---|---|---|---|---|---|")

    for model in models:
        data = load_extended_results(results_dir, model, "zero_shot_direct")
        if not data:
            continue
        metrics = compute_metrics(data)
        if not metrics:
            continue
        name = MODEL_NAMES.get(model, model)
        row = []
        for h in ["hop_1", "hop_2", "hop_3", "hop_4", "hop_5", "hop_6"]:
            v = metrics["hops"].get(h)
            row.append(f"{v:.0f}%" if v is not None else "—")
        print(f"  {name:<22} " + "  ".join(f"{r:>7}" for r in row))
        md4_lines.append(f"| {name} | " + " | ".join(row) + " |")

    # ---- Save outputs ----
    # Markdown
    with open(output_dir / "protocol_comparison.md", "w") as f:
        f.write("# Protocol Matrix Results\n\n")
        f.write("## Table 1: Overall Exact Match (%)\n\n")
        f.write("\n".join(md_lines) + "\n\n")
        f.write("## Table 2: Cat 4 Cultural Override EM (%)\n\n")
        f.write("\n".join(md2_lines) + "\n\n")
        f.write("## Table 4: Hop Scaling\n\n")
        f.write("\n".join(md4_lines) + "\n")

    # JSON summary
    summary = {}
    for (model, protocol), metrics in all_metrics.items():
        key = f"{model}__{protocol}"
        summary[key] = metrics
    with open(output_dir / "protocol_matrix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    with open(output_dir / "protocol_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "protocol", "overall_em", "western", "non_western",
                         "gap", "cat4", "cat1", "cat2", "cat3", "errors"])
        for (model, protocol), m in all_metrics.items():
            writer.writerow([
                MODEL_NAMES.get(model, model), PROTOCOL_NAMES.get(protocol, protocol),
                f"{m['overall_em']:.1f}", f"{m['western']:.1f}", f"{m['non_western']:.1f}",
                f"{m['gap']:.1f}", f"{m['cat4']:.1f}",
                f"{m['cats']['cat_1']:.1f}", f"{m['cats']['cat_2']:.1f}",
                f"{m['cats']['cat_3']:.1f}", m["errors"]
            ])

    print(f"\nSaved: {output_dir}/protocol_comparison.md")
    print(f"Saved: {output_dir}/protocol_matrix_summary.json")
    print(f"Saved: {output_dir}/protocol_matrix.csv")


def main():
    parser = argparse.ArgumentParser(description="Analyze protocol matrix results")
    parser.add_argument("--results-dir", default=".", help="Directory containing results_* dirs")
    parser.add_argument("--output-dir", default="./tables", help="Output directory")
    parser.add_argument("--models", type=str, default="gemma3_27b,deepseek_r1_32b,qwen3_32b",
                        help="Comma-separated model names")
    parser.add_argument("--protocols", type=str, default="zero_shot_direct,zero_shot_cot,few_shot_cot",
                        help="Comma-separated protocol names")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    protocols = [p.strip() for p in args.protocols.split(",")]

    generate_tables(args.results_dir, args.output_dir, models, protocols)


if __name__ == "__main__":
    main()
