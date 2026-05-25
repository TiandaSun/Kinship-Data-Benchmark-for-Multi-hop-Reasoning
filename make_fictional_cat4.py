#!/usr/bin/env python3
"""
Make Fictional Cat.4 Datasets
=============================
Creates fictional-system variants of the existing Cat.4 questions to
disentangle 'cultural knowledge' from 'in-context rule application'.

For each non-Western kinship system we substitute:
  - the system name in the question text (e.g. "Hawaiian" -> "Zorblax")
  - the cultural kin term (e.g. "classificatory mother" -> "tier-mother")

The structural rule (e.g. MZ classified the same as M) is preserved
exactly: same context, same anchor, same target persons, same ground
truth biological relation. Only the labels referring to the cultural
system are renamed.

Output:
  datasets_fictional/<system>_dataset.jsonl

Each output contains the original Cat.1, Cat.2, Cat.3 questions
unchanged plus relabelled Cat.4 questions, so the same llm_tester_v6.py
runs on it without modification.

Usage:
  python make_fictional_cat4.py \
      --input-dir ./datasets/ \
      --output-dir ./datasets_fictional/
"""

import argparse
import json
import re
from pathlib import Path

# Real-system -> fictional-system mappings.
# Fictional names are deliberately phonetically distant from any real
# culture and have no Wikipedia presence (verified manually).
SYSTEM_RENAME = {
    "hawaiian": "Zorblax",
    "iroquois": "Krendari",
    "dravidian": "Glabran",
    "crow":      "Vespian",
    "omaha":     "Tarskan",
}

# Cultural kin terms used inside Cat.4 question stems.  We rename only
# the descriptive *label*, not the rule it refers to, so the structural
# task remains identical.
TERM_RENAME = {
    "classificatory father":   "tier-pater",
    "classificatory fathers":  "tier-paters",
    "classificatory mother":   "tier-mater",
    "classificatory mothers":  "tier-maters",
    "classificatory sibling":  "tier-sibling",
    "classificatory siblings": "tier-siblings",
    "classificatory brother":  "tier-brother",
    "classificatory brothers": "tier-brothers",
    "classificatory sister":   "tier-sister",
    "classificatory sisters":  "tier-sisters",
    "classificatory parent":   "tier-parent",
    "classificatory parents":  "tier-parents",
    "classificatory child":    "tier-child",
    "classificatory children": "tier-children",
    "classificatory son":      "tier-son",
    "classificatory sons":     "tier-sons",
    "classificatory daughter": "tier-daughter",
    "classificatory daughters":"tier-daughters",
    "cross-cousin":            "off-axis cousin",
    "cross cousin":            "off-axis cousin",
    "parallel cousin":         "on-axis cousin",
    "parallel-cousin":         "on-axis cousin",
}


def relabel_question(q: dict, real_system: str, fictional_system: str,
                     mode: str = "full") -> dict:
    """Return a copy of q with labels renamed according to ``mode``.

    Modes
    -----
    full         rename system name AND kin terms (original behaviour)
    system_only  rename system name only; kin terms stay real
    term_only    rename kin terms only; system name stays real
    """
    out = dict(q)
    text = out.get("question_text", "")

    if mode in ("full", "system_only"):
        # Replace system name (case-insensitive, only in question text)
        pattern = re.compile(re.escape(real_system), re.IGNORECASE)
        text = pattern.sub(fictional_system, text)

        # Replace the formal "<System> system" / "<System> kinship" wording
        # in case the substitution above introduced odd capitalization.
        text = re.sub(r"the\s+" + re.escape(fictional_system) + r"\s+system",
                      f"the {fictional_system} system", text, flags=re.IGNORECASE)
        text = re.sub(r"In\s+the\s+" + re.escape(fictional_system),
                      f"In the {fictional_system}", text, flags=re.IGNORECASE)

    if mode in ("full", "term_only"):
        # Replace cultural kin terms.  Sort by length desc so longer phrases
        # match before shorter substrings.
        for src in sorted(TERM_RENAME, key=len, reverse=True):
            if src in text:
                text = text.replace(src, TERM_RENAME[src])
            cap = src.capitalize()
            if cap in text:
                text = text.replace(cap, TERM_RENAME[src].capitalize())

    out["question_text"] = text
    out["kinship_system_original"] = real_system
    out["kinship_system"] = f"fictional_{real_system}_{mode}"
    out["fictional_label"] = fictional_system if mode != "term_only" else None
    out["fictional_mode"] = mode
    return out


def process_file(in_path: Path, out_path: Path, mode: str = "full") -> dict:
    real_system = in_path.stem.replace("_dataset", "")
    if real_system not in SYSTEM_RENAME:
        # Western systems (eskimo, sudanese): copy unchanged
        questions = [json.loads(l) for l in in_path.open()]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")
        return {"system": real_system, "fictional": None, "cat4": 0,
                "total": len(questions), "skipped": True, "mode": mode}

    fictional = SYSTEM_RENAME[real_system]
    questions = [json.loads(l) for l in in_path.open()]
    n_cat4 = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for q in questions:
            if q.get("category") == 4:
                q = relabel_question(q, real_system, fictional, mode=mode)
                n_cat4 += 1
            f.write(json.dumps(q) + "\n")
    return {"system": real_system, "fictional": fictional,
            "cat4": n_cat4, "total": len(questions), "skipped": False,
            "mode": mode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="./datasets/", type=Path)
    ap.add_argument("--output-dir", default="./datasets_fictional/", type=Path)
    ap.add_argument("--mode", default="full",
                    choices=["full", "system_only", "term_only"],
                    help="full = rename system+terms (default); "
                         "system_only = rename system label only; "
                         "term_only = rename kin terms only.")
    args = ap.parse_args()

    summary = []
    for in_path in sorted(args.input_dir.glob("*_dataset.jsonl")):
        out_path = args.output_dir / in_path.name
        info = process_file(in_path, out_path, mode=args.mode)
        summary.append(info)
        if info["skipped"]:
            print(f"[skip] {info['system']}: copied {info['total']} questions unchanged")
        else:
            print(f"[ok]   {info['system']}: relabelled {info['cat4']} Cat.4 "
                  f"(of {info['total']}) -> system='{info['fictional']}' mode={info['mode']}")

    summary_path = args.output_dir / "fictional_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "mode": args.mode,
            "system_rename": SYSTEM_RENAME if args.mode in ("full", "system_only") else {},
            "term_rename": TERM_RENAME if args.mode in ("full", "term_only") else {},
            "files": summary,
        }, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
