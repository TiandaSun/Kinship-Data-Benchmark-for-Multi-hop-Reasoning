"""Cat.4-only error taxonomy from existing zero_shot_direct outputs across
3 OS models x Other-5 systems. Classifies each non-EM Cat.4 prediction into
one of 6 categories by inspecting precision/recall against the gold set,
ground-truth shape, and (for Pattern C reverse-mapping questions) the
relationship-label structure.

Categories:
  1. Empty/format error      : no parseable answer extracted from response.
  2. Default substitution    : ground-truth is a cultural-override relation;
                               prediction is the unmarked biological default
                               (uncle/aunt/cousin without paternal/maternal
                               or cross/parallel qualifier).
  3. Partial-set (under)     : precision=1.0, recall<1.0 -- correct names
                               but missed some members of the answer set.
  4. Over-inclusion          : recall=1.0, precision<1.0 -- gold members
                               present plus extraneous names.
  5. Partial overlap         : 0<precision<1.0 and 0<recall<1.0 -- mixed.
  6. Wrong / disjoint        : pred set disjoint from gold (recall=0,
                               precision=0, but a parseable answer was
                               produced).
"""
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/mnt/scratch/users/ts1201/Family-Tree-reasoning/familytree_v6")
MODELS = {
    "gemma3_27b":   "results_extended_gemma3_27b_zero_shot_direct",
    "deepseek_r1_32b": "results_extended_deepseek_r1_32b_zero_shot_direct",
    "qwen3_32b":    "results_extended_qwen3_32b_zero_shot_direct",
}
SYSTEMS = ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]

# Bare biological default labels that would indicate the Eskimo-default
# substitution when produced in answer to a cultural-override question.
DEFAULT_TERMS = {
    "uncle", "aunt", "cousin", "nephew", "niece",
    "grandfather", "grandmother", "grandparent", "grandchild",
    "brother", "sister", "sibling",
    "father", "mother", "parent", "child", "son", "daughter",
}

# Bifurcate / cross-cultural markers that, when present, mean the answer is
# NOT a bare-default substitution.
NON_DEFAULT_MARKERS = (
    "paternal", "maternal", "cross", "parallel", "classificatory",
    "fb", "fz", "mb", "mz", "fbc", "fzc", "mbc", "mzc",
    "skewed", "athai", "chitti", "periyamma", "periyappa", "chithappa",
)


def classify_relationship_error(pred_str, gt_str):
    """For Pattern C reverse-mapping questions: detect bare-biological-default
    substitution by checking whether the predicted relationship is one of
    {uncle, aunt, ...} without any cultural marker, while the ground truth
    explicitly contains a marker (paternal/maternal/cross/parallel/...)."""
    if not pred_str:
        return "empty_format"
    pred = pred_str.lower().strip()
    gt = (gt_str or "").lower().strip()

    pred_has_marker = any(m in pred for m in NON_DEFAULT_MARKERS)
    gt_has_marker = any(m in gt for m in NON_DEFAULT_MARKERS)

    # Strip parenthetical (e.g. "paternal uncle (FB)" -> "paternal uncle")
    pred_core = re.sub(r"\s*\([^)]*\)", "", pred).strip()
    gt_core = re.sub(r"\s*\([^)]*\)", "", gt).strip()

    # If GT has a marker but pred is one of the bare-default terms, that's a
    # default-substitution error.
    if gt_has_marker and not pred_has_marker:
        for term in DEFAULT_TERMS:
            if term in pred_core.split():
                return "default_substitution"
        # GT has a marker, pred has neither marker nor a bare-default term
        # -> classify as wrong/disjoint relationship.
        return "wrong_disjoint"

    # Pred and GT both have markers but disagree -> still wrong/disjoint.
    return "wrong_disjoint"


def classify_set_error(predicted, ground_truth, precision, recall):
    """For name-set Cat.4 questions, classify the error shape."""
    pred_list = predicted or []
    gt_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    pred_set = {str(p).strip().lower() for p in pred_list}
    gt_set = {str(g).strip().lower() for g in gt_list}

    if not pred_set:
        return "empty_format"
    if pred_set == gt_set:
        # Shouldn't be in error set, but guard.
        return "correct"
    if precision == 1.0 and recall < 1.0:
        return "partial_under"
    if recall == 1.0 and precision < 1.0:
        return "over_inclusion"
    if precision > 0 and recall > 0 and (precision < 1.0 or recall < 1.0):
        return "partial_overlap"
    # pred_set is disjoint from gt_set (precision=recall=0)
    return "wrong_disjoint"


def is_reverse_mapping(q_text):
    return "biological relationship" in (q_text or "").lower()


def classify_one(question):
    """Return error category for a single non-EM Cat.4 prediction."""
    if question.get("exact_match"):
        return "correct"

    q_text = question.get("question_text", "")
    predicted = question.get("predicted") or []
    ground_truth = question.get("ground_truth")
    raw_response = question.get("raw_response", "")
    precision = question.get("precision", 0.0)
    recall = question.get("recall", 0.0)

    # Empty response (no parseable answer or response is empty/error string)
    if not predicted and (not raw_response or raw_response.startswith("[ERROR")):
        return "empty_format"

    if is_reverse_mapping(q_text):
        # predicted is a list with one extracted relationship string
        pred_str = predicted[0] if predicted else raw_response
        return classify_relationship_error(pred_str, ground_truth)

    return classify_set_error(predicted, ground_truth, precision, recall)


def main():
    # By-category counts, grouped by (model, system) and overall.
    rows = []
    for model_name, dir_name in MODELS.items():
        for sys_name in SYSTEMS:
            path = ROOT / dir_name / f"{sys_name}_results.json"
            if not path.exists():
                print(f"  MISSING: {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            for q in data["questions"]:
                if q["category"] != 4:
                    continue
                cat_label = classify_one(q)
                rows.append({
                    "model": model_name,
                    "system": sys_name,
                    "qid": q["question_id"],
                    "n_hops": q["n_hops"],
                    "is_error": not q["exact_match"],
                    "category": cat_label,
                    "ground_truth": q["ground_truth"],
                    "predicted": q["predicted"],
                    "is_reverse": is_reverse_mapping(q["question_text"]),
                })

    total_questions = len(rows)
    errors = [r for r in rows if r["is_error"]]
    print(f"Total Cat.4 predictions (3 models x Other-5): {total_questions}")
    print(f"Errors (non-EM): {len(errors)} ({len(errors)/total_questions*100:.1f}%)")

    # Aggregate error category distribution
    cat_counts = defaultdict(int)
    for r in errors:
        cat_counts[r["category"]] += 1
    print("\n=== Cat.4 error category distribution (aggregated over 3 models x Other-5) ===")
    for cat in ["default_substitution", "wrong_disjoint", "partial_under",
                "over_inclusion", "partial_overlap", "empty_format"]:
        n = cat_counts[cat]
        pct = n / len(errors) * 100 if errors else 0
        print(f"  {cat:25s}  {n:4d}  {pct:5.1f}%")

    # Per-system breakdown
    print("\n=== Per-system error category share (3 models combined) ===")
    sys_cat = defaultdict(lambda: defaultdict(int))
    sys_err_n = defaultdict(int)
    for r in errors:
        sys_cat[r["system"]][r["category"]] += 1
        sys_err_n[r["system"]] += 1
    print(f"  {'system':10s}  {'n':>4s}  " + "  ".join(
        f"{c[:8]:>9s}" for c in ["default_", "wrong_d", "part_u", "over_in", "part_ov", "empty_f"]))
    for sys_name in SYSTEMS:
        if sys_err_n[sys_name] == 0:
            continue
        n = sys_err_n[sys_name]
        pcts = [sys_cat[sys_name][c] / n * 100 for c in
                ["default_substitution", "wrong_disjoint", "partial_under",
                 "over_inclusion", "partial_overlap", "empty_format"]]
        print(f"  {sys_name:10s}  {n:>4d}  " + "  ".join(f"{p:8.1f}%" for p in pcts))

    # Split by forward-vs-reverse mapping
    print("\n=== Forward vs reverse mapping ===")
    for is_rev_label, is_rev in [("Forward (Pattern A/B/D)", False),
                                  ("Reverse (Pattern C)", True)]:
        sub = [r for r in errors if r["is_reverse"] == is_rev]
        if not sub:
            continue
        print(f"\n  {is_rev_label}: n={len(sub)}")
        cc = defaultdict(int)
        for r in sub:
            cc[r["category"]] += 1
        for cat in ["default_substitution", "wrong_disjoint", "partial_under",
                    "over_inclusion", "partial_overlap", "empty_format"]:
            n = cc[cat]
            pct = n / len(sub) * 100 if sub else 0
            if n > 0:
                print(f"    {cat:25s}  {n:4d}  {pct:5.1f}%")

    # Save raw classifications for App table
    out_path = ROOT / "cat4_error_taxonomy.json"
    summary = {
        "total_cat4_predictions": total_questions,
        "total_errors": len(errors),
        "category_counts": dict(cat_counts),
        "per_system": {
            s: {c: sys_cat[s][c] for c in sys_cat[s]} for s in SYSTEMS
        },
        "models": list(MODELS.keys()),
        "systems": SYSTEMS,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
