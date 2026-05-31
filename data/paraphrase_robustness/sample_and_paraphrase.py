"""Sample 50 Cat.4 questions (10 per Other-5 system) and apply rule-based
paraphrasing that preserves entity names, system identifier, and kin term
while varying surface form."""
import json
import random
import re
from pathlib import Path

random.seed(20260523)

RESULTS_DIR = Path("/mnt/scratch/users/ts1201/Family-Tree-reasoning/familytree_v6/results_extended_gemma3_27b_zero_shot_direct")
OUT_PATH = Path(__file__).parent / "paraphrase_set.jsonl"

SYSTEMS = ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]
N_PER_SYSTEM = 10

# Paraphrase rules. Each rule has a regex to match an original pattern and a
# template for the paraphrase, using \1, \2 ... for captured groups.
# The captures preserve entity names, system identifier, and kin term verbatim.
PATTERNS = [
    # Pattern A: "In the {system} system, who is X's {KT}?"
    # Pattern A': "In the {system} kinship system, who is X's {KT}?"
    (
        re.compile(r"^In the (\w+) (?:kinship )?system, who is ([\w' ]+?)'s ([\w' ()/-]+)\?$"),
        "Under the {0} kinship convention, identify {1}'s {2}.",
    ),
    (
        re.compile(r"^In the (\w+) (?:kinship )?system, who are ([\w' ]+?)'s ([\w' ()/-]+s)\?$"),
        "Under the {0} kinship convention, enumerate {1}'s {2}.",
    ),
    # Pattern B: "List all of X's {KT}s according to the {system} system."
    (
        re.compile(r"^List all of ([\w' ]+?)'s ([\w' ()/-]+) according to the (\w+) (?:kinship )?system\.?$"),
        "Within the {2} system, give the complete list of {0}'s {1}.",
    ),
    # Pattern C: "In the {system} system, X is called Y's '{KT}'. What is their actual biological relationship?"
    (
        re.compile(r"^In the (\w+) (?:kinship )?system, ([\w' ]+?) is called ([\w' ]+?)'s '([\w' ()/-]+)'\. What is their actual biological relationship\?$"),
        "Per {0} kinship conventions, {1} holds the role of '{3}' relative to {2}. State their actual biological relationship.",
    ),
    # Pattern D (Dravidian variant with parenthetical native term): "List all of X's KT (...)s according to the Dravidian system."
    (
        re.compile(r"^List all of ([\w' ]+?)'s (.+?) according to the (\w+) (?:kinship )?system\.?$"),
        "Per {2} kinship conventions, enumerate every {1} of {0}.",
    ),
]


def paraphrase(q_text):
    """Try each pattern in order; return (paraphrased, pattern_idx) or None."""
    for idx, (rgx, tpl) in enumerate(PATTERNS):
        m = rgx.match(q_text.strip())
        if m:
            return tpl.format(*m.groups()), idx
    return None


def main():
    out_records = []
    miss = []
    for sys in SYSTEMS:
        with open(RESULTS_DIR / f"{sys}_results.json") as f:
            d = json.load(f)
        cat4 = [q for q in d["questions"] if q["category"] == 4]
        random.shuffle(cat4)
        picked = 0
        for q in cat4:
            if picked >= N_PER_SYSTEM:
                break
            pp = paraphrase(q["question_text"])
            if pp is None:
                miss.append((sys, q["question_text"]))
                continue
            paraphrased, pat_idx = pp
            out_records.append({
                "question_id": q["question_id"],
                "kinship_system": sys,
                "n_hops": q["n_hops"],
                "has_cultural_override": q["has_cultural_override"],
                "original_question": q["question_text"],
                "paraphrased_question": paraphrased,
                "pattern_idx": pat_idx,
                "ground_truth": q["ground_truth"],
                "original_predicted": q["predicted"],
                "original_exact_match": q["exact_match"],
                "original_f1": q["f1"],
                "bio_term": q["bio_term"],
                "kin_term": q["kin_term"],
            })
            picked += 1
        print(f"{sys}: picked {picked}/{N_PER_SYSTEM} (skipped {len(miss)} unmatched cumulatively)")

    with open(OUT_PATH, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out_records)} records to {OUT_PATH}")
    print(f"Pattern coverage: {sorted({r['pattern_idx'] for r in out_records})}")
    if miss:
        print(f"\n{len(miss)} unmatched (first 5):")
        for s, q in miss[:5]:
            print(f"  [{s}] {q}")

    # Sanity check: original-EM on the 50-question subsample for Gemma3 anchor
    orig_em = sum(r["original_exact_match"] for r in out_records) / len(out_records)
    print(f"\nGemma3-27B EM on the sampled 50 originals: {orig_em*100:.1f}%")


if __name__ == "__main__":
    main()
