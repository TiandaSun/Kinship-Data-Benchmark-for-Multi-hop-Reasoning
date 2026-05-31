"""Run Gemma3-27B (Ollama) on the 50 paraphrased Cat.4 questions, using the
same context and prompt template as the zero_shot_direct headline runs."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_tester_v6 import (  # noqa: E402
    OllamaProvider,
    create_prompt_zero_shot_direct,
    compare_answers,
    determine_question_type,
)

DATASETS_DIR = ROOT / "datasets_extended"
PARAPHRASE_PATH = Path(__file__).parent / "paraphrase_set.jsonl"
OUT_PATH = Path(__file__).parent / "paraphrase_results_gemma3.jsonl"
SUMMARY_PATH = Path(__file__).parent / "paraphrase_summary_gemma3.json"


def load_contexts():
    """Build qid -> dataset record map across the five Other-5 systems."""
    qid_to_rec = {}
    for sys_name in ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]:
        with open(DATASETS_DIR / f"{sys_name}_dataset.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                qid_to_rec[rec["question_id"]] = rec
    return qid_to_rec


def main():
    with open(PARAPHRASE_PATH) as f:
        paraphrase_records = [json.loads(l) for l in f]
    qid_to_rec = load_contexts()

    print(f"Loaded {len(paraphrase_records)} paraphrased questions")
    print(f"Loaded {len(qid_to_rec)} dataset records for context lookup")

    print("Initialising Gemma3-27B via Ollama...")
    provider = OllamaProvider(model="gemma3:27b", temperature=0.0, max_tokens=512)

    out_records = []
    t0 = time.time()
    for i, pr in enumerate(paraphrase_records, 1):
        qid = pr["question_id"]
        ds_rec = qid_to_rec.get(qid)
        if ds_rec is None:
            print(f"  [{i}/{len(paraphrase_records)}] {qid}: dataset record missing, skipping")
            continue

        # Build a synthetic question dict that uses the paraphrased text but
        # otherwise matches the original record's context + metadata.
        q_synth = dict(ds_rec)
        q_synth["question_text"] = pr["paraphrased_question"]

        prompt = create_prompt_zero_shot_direct(q_synth, with_rule_context=False)
        try:
            response = provider.generate(prompt)
        except Exception as e:
            response = f"[ERROR: {e}]"

        q_type = determine_question_type(q_synth)
        comp = compare_answers(response, pr["ground_truth"], q_type)

        rec = {
            "question_id": qid,
            "kinship_system": pr["kinship_system"],
            "n_hops": pr["n_hops"],
            "has_cultural_override": pr["has_cultural_override"],
            "original_question": pr["original_question"],
            "paraphrased_question": pr["paraphrased_question"],
            "pattern_idx": pr["pattern_idx"],
            "ground_truth": pr["ground_truth"],
            "raw_response": response,
            "predicted": comp["predicted"],
            "exact_match": comp["exact_match"],
            "partial_match": comp["partial_match"],
            "precision": comp["precision"],
            "recall": comp["recall"],
            "f1": comp["f1"],
            "original_exact_match": pr["original_exact_match"],
            "original_f1": pr["original_f1"],
        }
        out_records.append(rec)
        flip = ""
        if pr["original_exact_match"] and not comp["exact_match"]:
            flip = " (FLIP+→−)"
        elif (not pr["original_exact_match"]) and comp["exact_match"]:
            flip = " (FLIP−→+)"
        print(f"  [{i}/{len(paraphrase_records)}] {qid} | orig EM={int(pr['original_exact_match'])} para EM={int(comp['exact_match'])}{flip}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/len(paraphrase_records):.1f}s/q)")

    with open(OUT_PATH, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary statistics
    n = len(out_records)
    orig_em = sum(r["original_exact_match"] for r in out_records) / n
    para_em = sum(r["exact_match"] for r in out_records) / n
    orig_f1 = sum(r["original_f1"] for r in out_records) / n
    para_f1 = sum(r["f1"] for r in out_records) / n
    flips = sum(1 for r in out_records if r["original_exact_match"] != r["exact_match"])

    by_system = {}
    by_pattern = {}
    for r in out_records:
        for key, bucket in [(r["kinship_system"], by_system), (r["pattern_idx"], by_pattern)]:
            bucket.setdefault(key, {"n": 0, "orig_em": 0, "para_em": 0})
            bucket[key]["n"] += 1
            bucket[key]["orig_em"] += int(r["original_exact_match"])
            bucket[key]["para_em"] += int(r["exact_match"])

    summary = {
        "n_questions": n,
        "model": "gemma3:27b",
        "protocol": "zero_shot_direct",
        "orig_em": round(orig_em * 100, 2),
        "para_em": round(para_em * 100, 2),
        "delta_em_pp": round((para_em - orig_em) * 100, 2),
        "orig_f1": round(orig_f1 * 100, 2),
        "para_f1": round(para_f1 * 100, 2),
        "delta_f1_pp": round((para_f1 - orig_f1) * 100, 2),
        "answer_flips": flips,
        "by_system": {
            k: {
                "n": v["n"],
                "orig_em_pct": round(v["orig_em"] / v["n"] * 100, 1),
                "para_em_pct": round(v["para_em"] / v["n"] * 100, 1),
            } for k, v in by_system.items()
        },
        "by_pattern": {
            str(k): {
                "n": v["n"],
                "orig_em_pct": round(v["orig_em"] / v["n"] * 100, 1),
                "para_em_pct": round(v["para_em"] / v["n"] * 100, 1),
            } for k, v in by_pattern.items()
        },
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(f"  Original EM:    {orig_em*100:.1f}%")
    print(f"  Paraphrase EM:  {para_em*100:.1f}%")
    print(f"  Δ EM:           {(para_em - orig_em)*100:+.1f} pp")
    print(f"  Δ F1:           {(para_f1 - orig_f1)*100:+.1f} pp")
    print(f"  Answer flips:   {flips}/{n}")
    print(f"\nResults: {OUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
