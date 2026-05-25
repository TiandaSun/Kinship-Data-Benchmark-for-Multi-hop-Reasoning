#!/usr/bin/env python3
"""
Multi-Seed Cat.4 Evaluation with Bootstrap Confidence Intervals
================================================================
Runs n samples (default 5) at temperature=0.7 over a stratified Cat.4
subsample (default 200 questions) for one Ollama model.  Reports:

  - per-question agreement rate across samples
  - per-system mean accuracy and bootstrap 95% CI
  - per-question accuracy as the proportion of samples scoring EM=1

Output JSON consumed by aggregate_multi_seed.py to build a paper-
ready table with proper CIs.

Reuses logic from llm_tester_v6.py (extract_answer_from_cot,
compare_answers, determine_question_type, OllamaProvider).

Usage on Viking:
    python multi_seed_cat4.py \
        --dataset-dir ./datasets/ \
        --model gemma3:27b \
        --output ./results_multi_seed/gemma3_27b.json \
        --num-samples 5 \
        --temperature 0.7 \
        --per-system-cap 40

Author: Tianda (EMNLP 2026 revision)
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Reuse helpers from the production tester so behaviour is identical.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_tester_v6 import (  # noqa: E402
    OllamaProvider,
    create_prompt_zero_shot_direct,
    determine_question_type,
    compare_answers,
    extract_names_from_response,
    extract_number_from_response,
    extract_relationship_from_response,
    KINSHIP_SYSTEMS,
)


def load_cat4(dataset_dir: Path, per_system_cap: int, seed: int) -> list:
    """Stratified Cat.4 subsample, balanced across systems."""
    rng = random.Random(seed)
    out = []
    for system in KINSHIP_SYSTEMS:
        path = dataset_dir / f"{system}_dataset.jsonl"
        if not path.exists():
            continue
        all_q = [json.loads(l) for l in path.open()]
        cat4 = [q for q in all_q if q.get("category") == 4]
        rng.shuffle(cat4)
        out.extend(cat4[:per_system_cap])
    return out


def predict_once(provider, question: dict) -> dict:
    """One generation; return parsed prediction + EM bool."""
    prompt = create_prompt_zero_shot_direct(question)
    response = provider.generate(prompt)
    qtype = determine_question_type(question)

    if qtype == "names":
        predicted = extract_names_from_response(response)
    elif qtype == "number":
        predicted = extract_number_from_response(response)
    elif qtype == "relationship":
        predicted = extract_relationship_from_response(response)
    else:
        predicted = response.strip().split("\n")[0]

    cmp = compare_answers(predicted, question.get("ground_truth"), qtype)
    return {
        "raw_response": response,
        "predicted": predicted,
        "exact": bool(cmp.get("exact_match", cmp.get("exact", False))),
    }


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for the mean."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path("./datasets/"))
    ap.add_argument("--model", required=True,
                    help="Ollama model name (e.g. gemma3:27b)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--per-system-cap", type=int, default=40,
                    help="Max Cat.4 questions per system (40 x 5 systems = 200)")
    ap.add_argument("--subsample-seed", type=int, default=2026)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    questions = load_cat4(args.dataset_dir, args.per_system_cap,
                          args.subsample_seed)
    print(f"Loaded {len(questions)} Cat.4 questions across "
          f"{len(set(q['kinship_system'] for q in questions))} systems")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat()

    provider = OllamaProvider(args.model,
                              temperature=args.temperature,
                              max_tokens=args.max_tokens)

    per_question = []
    for q_idx, q in enumerate(questions):
        sample_results = []
        for s in range(args.num_samples):
            try:
                r = predict_once(provider, q)
            except Exception as e:
                r = {"raw_response": f"[ERROR] {e}", "predicted": None,
                     "exact": False}
            sample_results.append(r)

        em_rate = sum(1 for r in sample_results if r["exact"]) / args.num_samples
        # Agreement = fraction of sample pairs where both/either succeed.
        # We report majority-vote EM and the per-question EM rate.
        majority_em = em_rate >= 0.5
        per_question.append({
            "question_id": q.get("question_id"),
            "system": q.get("kinship_system"),
            "n_hops": q.get("n_hops"),
            "has_cultural_override": q.get("has_cultural_override"),
            "ground_truth": q.get("ground_truth"),
            "samples": sample_results,
            "em_rate": em_rate,
            "majority_em": majority_em,
        })

        if args.verbose and (q_idx + 1) % 20 == 0:
            running = sum(p["em_rate"] for p in per_question) / len(per_question)
            print(f"  [{q_idx+1}/{len(questions)}] running mean EM = {running:.3f}")

    # Aggregate
    by_system = defaultdict(list)
    for p in per_question:
        by_system[p["system"]].append(p["em_rate"])

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

    overall_ems = [p["em_rate"] for p in per_question]
    overall_lo, overall_hi = bootstrap_ci(overall_ems)
    overall = {
        "n_questions": len(overall_ems),
        "mean_em": round(sum(overall_ems) / len(overall_ems), 4),
        "ci95_lo": round(overall_lo, 4),
        "ci95_hi": round(overall_hi, 4),
    }

    out = {
        "model": args.model,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "per_system_cap": args.per_system_cap,
        "subsample_seed": args.subsample_seed,
        "started": started,
        "finished": datetime.now().isoformat(),
        "by_system": summary,
        "overall": overall,
        "per_question": per_question,
    }
    args.output.write_text(json.dumps(out, indent=2))

    print(f"\nWrote {args.output}")
    print(f"\nOverall mean EM = {overall['mean_em']:.3f} "
          f"[95% CI {overall['ci95_lo']:.3f}, {overall['ci95_hi']:.3f}]")
    print(f"\nPer-system:")
    for system, s in sorted(summary.items()):
        print(f"  {system:<10} n={s['n_questions']:<3} "
              f"EM={s['mean_em']:.3f} "
              f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]")


if __name__ == "__main__":
    main()
