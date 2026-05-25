"""
Human-baseline pilot for KinshipQA.

Generates a stratified sample of N questions across 7 systems x 4 categories,
presents each with the natural-language context and question, asks the
annotator (a) whether the question is answerable from the context and (b)
what the answer is, then compares to the gold answer and reports
parseability + accuracy rates per stratum.

The pilot is intentionally small (default n=28: one question per
system x category cell) so that a single human-expert annotator can complete
it in ~45 minutes and the result is interpretable as a serialization /
NL-clarity sanity check, not as a benchmark ceiling.

Usage:
    python human_baseline_pilot.py sample      # writes pilot_sample.jsonl
    python human_baseline_pilot.py annotate    # CLI annotation; appends to pilot_responses.jsonl
    python human_baseline_pilot.py score       # reads pilot_responses.jsonl, prints summary
"""

import json
import random
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "datasets_v6_2"
OUT_SAMPLE = Path(__file__).resolve().parent / "pilot_sample.jsonl"
OUT_RESPONSES = Path(__file__).resolve().parent / "pilot_responses.jsonl"

SYSTEMS = ["eskimo", "sudanese", "hawaiian", "iroquois", "dravidian", "crow", "omaha"]
CATEGORIES = [1, 2, 3, 4]
PER_CELL = 1  # one question per (system, category) -> 7*4 = 28 questions
SEED = 20260507


def normalize_answer(ans: str) -> str:
    return " ".join(ans.strip().lower().split())


def cmd_sample() -> None:
    rng = random.Random(SEED)
    sample = []
    for system in SYSTEMS:
        path = DATA_DIR / f"{system}_dataset.jsonl"
        with path.open() as f:
            qs = [json.loads(line) for line in f]
        for cat in CATEGORIES:
            pool = [q for q in qs if q["category"] == cat]
            if not pool:
                print(f"warn: no questions for {system} cat={cat}")
                continue
            for q in rng.sample(pool, k=min(PER_CELL, len(pool))):
                sample.append({
                    "question_id": q["question_id"],
                    "system": system,
                    "category": cat,
                    "n_hops": q["n_hops"],
                    "context": q["context"],
                    "question_text": q["question_text"],
                    "ground_truth": q["ground_truth"],
                })
    rng.shuffle(sample)
    with OUT_SAMPLE.open("w") as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")
    print(f"wrote {len(sample)} questions to {OUT_SAMPLE}")


def cmd_annotate() -> None:
    if not OUT_SAMPLE.exists():
        sys.exit("run `sample` first")
    items = [json.loads(line) for line in OUT_SAMPLE.open()]
    done_ids = set()
    if OUT_RESPONSES.exists():
        for line in OUT_RESPONSES.open():
            done_ids.add(json.loads(line)["question_id"])
    todo = [it for it in items if it["question_id"] not in done_ids]
    print(f"{len(todo)} questions remaining; type 'quit' at any prompt to stop.\n")
    for i, it in enumerate(todo, 1):
        print(f"\n--- {i}/{len(todo)}  ({it['system']}, cat {it['category']}, {it['n_hops']}-hop) ---")
        print(f"Context:  {it['context']}")
        print(f"Question: {it['question_text']}")
        parseable = input("Answerable from the context? [y/n/quit] ").strip().lower()
        if parseable == "quit":
            break
        if parseable not in ("y", "n"):
            print("skip"); continue
        ans = "" if parseable == "n" else input("Your answer: ").strip()
        if ans.lower() == "quit":
            break
        with OUT_RESPONSES.open("a") as f:
            f.write(json.dumps({
                "question_id": it["question_id"],
                "system": it["system"],
                "category": it["category"],
                "n_hops": it["n_hops"],
                "ground_truth": it["ground_truth"],
                "parseable": parseable == "y",
                "human_answer": ans,
                "gold_match": normalize_answer(ans) == normalize_answer(it["ground_truth"]),
            }) + "\n")
    print(f"\nresponses appended to {OUT_RESPONSES}")


def cmd_score() -> None:
    if not OUT_RESPONSES.exists():
        sys.exit("no responses yet")
    rows = [json.loads(line) for line in OUT_RESPONSES.open()]
    n = len(rows)
    n_parse = sum(r["parseable"] for r in rows)
    n_match = sum(r["gold_match"] for r in rows)
    print(f"\nN annotated:        {n}")
    print(f"Parseable:          {n_parse}/{n}  ({100*n_parse/n:.1f}%)")
    print(f"Gold-match:         {n_match}/{n}  ({100*n_match/n:.1f}%)")
    print(f"Among parseable:    {n_match}/{n_parse}  "
          f"({100*n_match/max(1,n_parse):.1f}%)")
    print("\nBy category:")
    for cat in CATEGORIES:
        subset = [r for r in rows if r["category"] == cat]
        if not subset: continue
        p = sum(r["parseable"] for r in subset)
        m = sum(r["gold_match"] for r in subset)
        print(f"  cat {cat}: parseable {p}/{len(subset)}, gold-match {m}/{len(subset)}")
    print("\nBy system:")
    for sys_ in SYSTEMS:
        subset = [r for r in rows if r["system"] == sys_]
        if not subset: continue
        p = sum(r["parseable"] for r in subset)
        m = sum(r["gold_match"] for r in subset)
        print(f"  {sys_}: parseable {p}/{len(subset)}, gold-match {m}/{len(subset)}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("sample", "annotate", "score"):
        sys.exit(__doc__)
    {"sample": cmd_sample, "annotate": cmd_annotate, "score": cmd_score}[sys.argv[1]]()


if __name__ == "__main__":
    main()
