#!/usr/bin/env python3
"""
Sample 4-hop Cat.4 Predictions for AI / Manual Spot-Check
=========================================================
The cross-model spread on 4-hop Other-5 Cat.4 is unusually wide
(Gemma3 ~50%, DeepSeek/Qwen3 ~0-10%), which raises the natural
question of whether the gap reflects a real reasoning failure or
a parser / verbose-output artefact.

This script draws a stratified 50-question sample of 4-hop Cat.4
questions on the Other-5 systems, joins each question with all three
open-source models' raw responses, and writes a single review-friendly
JSONL plus a Markdown rendering for hand inspection or AI review.

Sampling design (default n=50):
    - 5 non-Western systems
    - 10 questions per system, randomly drawn with seed 2026
    - Each question record includes:
        * question text and ground truth
        * all three models' raw_response and parsed predicted answer
        * per-model exact_match flag
This gives 150 (q, model) cells you can read in one sitting.

Usage:
    python sample_4hop_for_review.py \
        --results-dir-template ./results_4hop_{model}_zero_shot_direct \
        --models gemma3_27b deepseek_r1_32b qwen3_32b \
        --n-per-system 10 \
        --output-jsonl ./samples/4hop_cat4_spotcheck.jsonl \
        --output-md ./samples/4hop_cat4_spotcheck.md
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

NON_WESTERN = ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]


def index_by_qid(results_dir: Path) -> dict:
    """{question_id: question record} from per-system results files."""
    out = {}
    for path in sorted(results_dir.glob("*_results.json")):
        if "combined" in path.stem:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for q in data.get("questions", []):
            qid = q.get("question_id")
            if qid:
                out[qid] = q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir-template", required=True,
                    help="e.g. ./results_4hop_{model}_zero_shot_direct")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-per-system", type=int, default=10,
                    help="Questions per non-Western system (default 10 -> 50 total)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output-jsonl", type=Path,
                    default=Path("./samples/4hop_cat4_spotcheck.jsonl"))
    ap.add_argument("--output-md", type=Path,
                    default=Path("./samples/4hop_cat4_spotcheck.md"))
    args = ap.parse_args()

    # Index every model's results once
    indexes = {}
    for model in args.models:
        rdir = Path(args.results_dir_template.format(model=model))
        indexes[model] = index_by_qid(rdir)
        print(f"  {model}: indexed {len(indexes[model])} questions")

    # Find candidate question IDs: 4-hop Cat.4 in non-Western systems
    # Use the first model's index as the source of truth (all models see
    # the same questions; we just need a question_id list).
    first_model = args.models[0]
    candidates_by_system = defaultdict(list)
    for qid, q in indexes[first_model].items():
        if (q.get("category") == 4
                and q.get("n_hops") == 4
                and q.get("kinship_system") in NON_WESTERN):
            candidates_by_system[q["kinship_system"]].append(qid)

    rng = random.Random(args.seed)
    sample_qids = []
    for sys_ in NON_WESTERN:
        pool = candidates_by_system.get(sys_, [])
        rng.shuffle(pool)
        sample_qids.extend(pool[:args.n_per_system])

    print(f"\nSampled {len(sample_qids)} questions across {len(NON_WESTERN)} systems")

    # Build joined records
    records = []
    for qid in sample_qids:
        base = indexes[first_model][qid]
        record = {
            "question_id": qid,
            "kinship_system": base.get("kinship_system"),
            "n_hops": base.get("n_hops"),
            "question_text": base.get("question_text"),
            "context": base.get("context"),
            "ground_truth": base.get("ground_truth"),
            "bio_term": base.get("bio_term"),
            "kin_term": base.get("kin_term"),
            "predictions": {},
        }
        for model in args.models:
            q = indexes[model].get(qid, {})
            record["predictions"][model] = {
                "exact_match": bool(q.get("exact_match", q.get("exact", False))),
                "predicted": q.get("predicted"),
                "raw_response": q.get("raw_response"),
            }
        records.append(record)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {args.output_jsonl}")

    # Markdown rendering for review
    lines = ["# 4-hop Cat.4 Spot-Check Sample\n"]
    lines.append(f"Sampled {len(records)} questions across {len(NON_WESTERN)} non-Western systems "
                 f"(seed={args.seed}).\n")
    lines.append("For each question, we show the question text, context, ground truth, "
                 "and each model's parsed prediction + raw response. The reviewer's task is "
                 "to judge whether each model's response is *substantively* correct (semantic "
                 "match to the ground truth) even if exact-match scoring marked it wrong, "
                 "i.e. to detect parser / verbose-output false negatives.\n")

    by_sys = defaultdict(list)
    for r in records:
        by_sys[r["kinship_system"]].append(r)

    for sys_ in NON_WESTERN:
        lines.append(f"\n## {sys_.capitalize()}\n")
        for i, r in enumerate(by_sys[sys_], 1):
            gt = r["ground_truth"]
            if isinstance(gt, list):
                gt_str = ", ".join(gt)
            else:
                gt_str = str(gt)
            lines.append(f"### {sys_}-{i}  (id={r['question_id']})\n")
            lines.append(f"**Question.** {r['question_text']}\n")
            lines.append(f"**Context.** {r['context']}\n")
            lines.append(f"**Ground truth.** `{gt_str}`")
            if r.get("kin_term") and r.get("bio_term"):
                lines.append(f"  (kin term: *{r['kin_term']}* / bio: *{r['bio_term']}*)")
            lines.append("")
            for model in args.models:
                p = r["predictions"][model]
                em_marker = "✓" if p["exact_match"] else "✗"
                pred = p.get("predicted")
                raw = (p.get("raw_response") or "").strip()
                if len(raw) > 600:
                    raw = raw[:600] + " ..."
                lines.append(f"- **{model}** {em_marker}")
                lines.append(f"  - parsed: `{pred}`")
                lines.append(f"  - raw: > {raw}")
            lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines))
    print(f"Wrote {args.output_md}")

    # Console summary
    n_em = defaultdict(int)
    for r in records:
        for model, p in r["predictions"].items():
            if p["exact_match"]:
                n_em[model] += 1
    print(f"\nEM count in sample (out of {len(records)}):")
    for model in args.models:
        print(f"  {model}: {n_em[model]}/{len(records)}")


if __name__ == "__main__":
    main()
