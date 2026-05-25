#!/usr/bin/env python3
"""
Re-aggregate Multi-Seed Cat.4 Results from Saved Samples
========================================================
Recovers from a key-name bug in the original multi_seed_cat4.py: the
saved per-sample records retain the model's raw response and the
parsed prediction, so we can rerun compare_answers() in pure Python
without re-querying the LLM.

For each saved per_question entry, we look up the original question
(via question_id), determine its question_type, and run the standard
compare_answers() logic. Output schema matches multi_seed_cat4.py so
aggregate_multi_seed.py can run unchanged afterward.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_tester_v6 import (  # noqa: E402
    determine_question_type,
    compare_answers,
    KINSHIP_SYSTEMS,
)


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)


def load_question_index(dataset_dir: Path) -> dict:
    """question_id -> full question dict, for question_type lookup."""
    idx = {}
    for system in KINSHIP_SYSTEMS:
        path = dataset_dir / f"{system}_dataset.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            q = json.loads(line)
            qid = q.get("question_id")
            if qid:
                idx[qid] = q
    return idx


def fix_one(in_path: Path, out_path: Path, qindex: dict):
    data = json.loads(in_path.read_text())
    per_q = data["per_question"]
    n_samples = data["num_samples"]

    fixed_per_q = []
    by_system = defaultdict(list)

    for entry in per_q:
        qid = entry["question_id"]
        q = qindex.get(qid)
        if q is None:
            # Fall back: synthesize minimal record
            q = {
                "question_text": "",
                "ground_truth": entry["ground_truth"],
                "category": 4,
            }
        qtype = determine_question_type(q)
        gt = entry["ground_truth"]

        new_samples = []
        for s in entry["samples"]:
            predicted = s.get("predicted")
            cmp = compare_answers(predicted, gt, qtype)
            exact = bool(cmp.get("exact_match", cmp.get("exact", False)))
            new_samples.append({**s, "exact": exact})

        em_rate = sum(1 for s in new_samples if s["exact"]) / n_samples
        majority_em = em_rate >= 0.5

        fixed_per_q.append({
            **entry,
            "samples": new_samples,
            "em_rate": em_rate,
            "majority_em": majority_em,
        })
        by_system[entry["system"]].append(em_rate)

    summary = {}
    for system, ems in by_system.items():
        mean = sum(ems) / len(ems)
        lo, hi = bootstrap_ci(ems)
        summary[system] = {
            "n_questions": len(ems),
            "mean_em": round(mean, 4),
            "ci95_lo": round(lo, 4),
            "ci95_hi": round(hi, 4),
        }

    overall_ems = [p["em_rate"] for p in fixed_per_q]
    overall_lo, overall_hi = bootstrap_ci(overall_ems)
    overall = {
        "n_questions": len(overall_ems),
        "mean_em": round(sum(overall_ems) / len(overall_ems), 4),
        "ci95_lo": round(overall_lo, 4),
        "ci95_hi": round(overall_hi, 4),
    }

    out = {**data, "by_system": summary, "overall": overall,
           "per_question": fixed_per_q,
           "reaggregated": True}
    out_path.write_text(json.dumps(out, indent=2))
    return overall, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--dataset-dir", type=Path, default=Path("./datasets/"))
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the input file (default: write *.fixed.json)")
    args = ap.parse_args()

    qindex = load_question_index(args.dataset_dir)
    print(f"Indexed {len(qindex)} questions for question_type lookup\n")

    for in_path in args.inputs:
        out_path = in_path if args.in_place else in_path.with_suffix(".fixed.json")
        overall, by_system = fix_one(in_path, out_path, qindex)
        print(f"  {in_path.name}")
        print(f"    overall: EM={overall['mean_em']:.3f} "
              f"[{overall['ci95_lo']:.3f}, {overall['ci95_hi']:.3f}]")
        for system, s in sorted(by_system.items()):
            print(f"    {system:<10} EM={s['mean_em']:.3f} "
                  f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]  n={s['n_questions']}")
        print(f"    -> wrote {out_path}\n")


if __name__ == "__main__":
    main()
