#!/usr/bin/env python3
"""
Augment Cat.4 with 4-hop Cultural-Override Questions
=====================================================
The original v6.1 pipeline did not define 4-hop entries in
KINSHIP_OVERRIDES, so no Cat.4 cultural-override questions exist at
4 hops in the released datasets.  This makes Tables 6/13's 4-hop
column a category-mix artefact (it is all Cat.2 biological multi-hop)
and produces a misleading 4-hop "spike" in the depth-scaling plot.

This script extends KINSHIP_OVERRIDES with anthropologically reasonable
4-hop override paths for each non-Western system, reuses the existing
ontologies in ./ontologies/, and generates new 4-hop Cat.4 questions
for the five Other-5 systems.  Output is a copy of the dataset with
the new questions appended (IDs starting at <system>_0501).

Anthropological grounding for each override:

  Hawaiian (generational lumping):
    (FB, C, C)  -> "classificatory child"      (G-1 relative is own child)
    (MZ, C, C)  -> "classificatory child"
    (FF, B, C)  -> "classificatory father"     (G+1 relative is own father)
    (MM, Z, C)  -> "classificatory mother"

  Iroquois (parallel/cross extended):
    (FB, C, C)  -> "parallel niece/nephew"     (parallel chain stays parallel)
    (MZ, C, C)  -> "parallel niece/nephew"
    (FF, B, C)  -> "classificatory father"     (parallel-uncle line)
    (MM, Z, C)  -> "classificatory mother"

  Dravidian (parallel/cross + cross-cousin marriage):
    (FB, C, C)  -> "parallel niece/nephew"
    (MZ, C, C)  -> "parallel niece/nephew"
    (FF, B, C)  -> "classificatory father"
    (MM, Z, C)  -> "classificatory mother"

  Crow (matrilineal skewing extended):
    (FZ, C, C)  -> "father/female father (skewed - same matrilineage)"
    (FB, C, C)  -> "classificatory niece/nephew"  (parallel side)
    (FF, Z, C)  -> "father/female father (skewed)"

  Omaha (patrilineal skewing extended, mirror of Crow):
    (MB, C, C)  -> "mother/male mother (skewed - same patrilineage)"
    (MZ, C, C)  -> "classificatory niece/nephew"
    (MM, B, C)  -> "mother/male mother (skewed)"

Usage (no GPU required):
    python augment_cat4_4hop.py \
        --in-dataset-dir ./datasets/ \
        --in-ontology-dir ./ontologies/ \
        --out-dataset-dir ./datasets_v6_2/ \
        --questions-per-system 50 \
        --seed 2026
"""

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


HERE = Path(__file__).resolve().parent


# 4-hop override paths to inject. Keys are tuples of biological predicate
# names; values match the structure of pipeline.KINSHIP_OVERRIDES entries.
EXTRA_OVERRIDES_4HOP: Dict[str, Dict[Tuple[str, ...], Dict]] = {
    "hawaiian": {
        ("hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "paternal uncle's grandchild",
            "kin_term": "classificatory child",
            "kin_predicate": "kin:hasClassificatoryChild",
        },
        ("hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "maternal aunt's grandchild",
            "kin_term": "classificatory child",
            "kin_predicate": "kin:hasClassificatoryChild",
        },
        ("hasFather", "hasFather", "hasBrother", "hasChild"): {
            "bio_term": "father's paternal uncle's child",
            "kin_term": "classificatory father",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild"): {
            "bio_term": "mother's maternal aunt's child",
            "kin_term": "classificatory mother",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
    },
    "iroquois": {
        ("hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "paternal uncle's grandchild",
            "kin_term": "parallel niece/nephew (classificatory)",
            "kin_predicate": "kin:hasParallelNieceNephew",
        },
        ("hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "maternal aunt's grandchild",
            "kin_term": "parallel niece/nephew (classificatory)",
            "kin_predicate": "kin:hasParallelNieceNephew",
        },
        ("hasFather", "hasFather", "hasBrother", "hasChild"): {
            "bio_term": "father's paternal uncle's child",
            "kin_term": "classificatory father",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild"): {
            "bio_term": "mother's maternal aunt's child",
            "kin_term": "classificatory mother",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
    },
    "dravidian": {
        ("hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "paternal uncle's grandchild",
            "kin_term": "parallel niece/nephew (classificatory)",
            "kin_predicate": "kin:hasParallelNieceNephew",
        },
        ("hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "maternal aunt's grandchild",
            "kin_term": "parallel niece/nephew (classificatory)",
            "kin_predicate": "kin:hasParallelNieceNephew",
        },
        ("hasFather", "hasFather", "hasBrother", "hasChild"): {
            "bio_term": "father's paternal uncle's child",
            "kin_term": "classificatory father (Periyappa/Chithappa)",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild"): {
            "bio_term": "mother's maternal aunt's child",
            "kin_term": "classificatory mother (Chitti/Periyamma)",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
    },
    "crow": {
        # Matrilineal skewing extended: FZ-line grandchild stays skewed
        ("hasFather", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "paternal aunt's grandchild",
            "kin_term": "father/female father (skewed - same matrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        # Parallel side (FB line) is non-skewed: niece/nephew at 4-hop
        ("hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "paternal uncle's grandchild",
            "kin_term": "classificatory niece/nephew",
            "kin_predicate": "kin:hasClassificatoryNieceNephew",
        },
        # Father's father's sister line continues skewing
        ("hasFather", "hasFather", "hasSister", "hasChild"): {
            "bio_term": "father's paternal aunt's child",
            "kin_term": "father/female father (skewed)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
    },
    "omaha": {
        # Patrilineal skewing extended: MB-line grandchild stays skewed
        ("hasMother", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "maternal uncle's grandchild",
            "kin_term": "mother/male mother (skewed - same patrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        # Parallel side (MZ line) non-skewed
        ("hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "maternal aunt's grandchild",
            "kin_term": "classificatory niece/nephew",
            "kin_predicate": "kin:hasClassificatoryNieceNephew",
        },
        # Mother's mother's brother line continues skewing
        ("hasMother", "hasMother", "hasBrother", "hasChild"): {
            "bio_term": "mother's maternal uncle's child",
            "kin_term": "mother/male mother (skewed)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
    },
}


def load_pipeline_module():
    """Import pipeline module despite the dotted filename."""
    spec = importlib.util.spec_from_file_location(
        "kinshipqa_pipeline_v6_1",
        HERE / "kinshipqa_pipeline_v6.1.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kinshipqa_pipeline_v6_1"] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_overrides(pipeline_mod) -> None:
    """Inject 4-hop override paths into the pipeline's KINSHIP_OVERRIDES."""
    for system, extra in EXTRA_OVERRIDES_4HOP.items():
        if system not in pipeline_mod.KINSHIP_OVERRIDES:
            pipeline_mod.KINSHIP_OVERRIDES[system] = {}
        for path, info in extra.items():
            if path in pipeline_mod.KINSHIP_OVERRIDES[system]:
                continue  # do not clobber an existing definition
            pipeline_mod.KINSHIP_OVERRIDES[system][path] = info


def question_to_dict(q) -> dict:
    """Match the shape of existing dataset records."""
    return {
        "question_id": q.question_id,
        "question_text": q.question_text,
        "category": q.category,
        "n_hops": q.n_hops,
        "kinship_system": q.kinship_system,
        "anchor_person": q.anchor_person,
        "anchor_name": q.anchor_name,
        "target_persons": q.target_persons,
        "target_names": q.target_names,
        "ground_truth": q.ground_truth,
        "context": q.context,
        "path": list(q.path) if q.path else None,
        "bio_term": q.bio_term,
        "kin_term": q.kin_term,
        "has_cultural_override": q.has_cultural_override,
        "proof_graph_size": q.proof_graph_size,
    }


def augment_one(pipeline_mod, system: str,
                ontology_dir: Path, in_dataset_dir: Path,
                out_dataset_dir: Path, count: int, seed: int) -> dict:
    ttl_path = ontology_dir / f"{system}.ttl"
    if not ttl_path.exists():
        raise FileNotFoundError(f"Missing ontology: {ttl_path}")

    kg = pipeline_mod.KinshipGraph(str(ttl_path), system)
    qg = pipeline_mod.QuestionGenerator(kg, system, seed=seed)

    # Set the next-id counter so new question IDs do not collide with
    # the existing 1..432 range.  We reserve 0501..0999 for 4-hop additions.
    if hasattr(qg, "_next_id_counter"):
        qg._next_id_counter = 500
    elif hasattr(qg, "next_id"):
        qg.next_id = 500
    elif hasattr(qg, "id_counter"):
        qg.id_counter = 500

    paths_to_use = list(EXTRA_OVERRIDES_4HOP[system].keys())
    new_questions = qg._generate_cat4_for_hops(paths_to_use, n_hops=4, count=count)

    # Force the question_id prefix.  The pipeline uses _next_id() returning
    # a string like "{system}_{counter:04d}".  We reformat to the expected
    # form if it differs.
    for q in new_questions:
        # Repair id if the counter started at 0
        try:
            num = int(str(q.question_id).split("_")[-1])
        except ValueError:
            num = 0
        if num < 500:
            num = 500 + new_questions.index(q) + 1
            q.question_id = f"{system}_{num:04d}"

    # Append to existing dataset
    in_path = in_dataset_dir / f"{system}_dataset.jsonl"
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dataset_dir / f"{system}_dataset.jsonl"

    n_orig = 0
    with out_path.open("w") as outf:
        if in_path.exists():
            for line in in_path.open():
                outf.write(line)
                n_orig += 1
        for q in new_questions:
            outf.write(json.dumps(question_to_dict(q)) + "\n")

    return {
        "system": system,
        "n_original": n_orig,
        "n_new_4hop_cat4": len(new_questions),
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dataset-dir", default="./datasets/", type=Path)
    ap.add_argument("--in-ontology-dir", default="./ontologies/", type=Path)
    ap.add_argument("--out-dataset-dir", default="./datasets_v6_2/", type=Path)
    ap.add_argument("--questions-per-system", type=int, default=50,
                    help="Target number of new 4-hop Cat.4 questions per non-Western system")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--copy-western-passthrough", action="store_true", default=True,
                    help="Also copy Eskimo and Sudanese unchanged into out-dataset-dir")
    args = ap.parse_args()

    pipeline_mod = load_pipeline_module()
    patch_overrides(pipeline_mod)

    summary = []
    for system in EXTRA_OVERRIDES_4HOP.keys():
        info = augment_one(
            pipeline_mod=pipeline_mod,
            system=system,
            ontology_dir=args.in_ontology_dir,
            in_dataset_dir=args.in_dataset_dir,
            out_dataset_dir=args.out_dataset_dir,
            count=args.questions_per_system,
            seed=args.seed,
        )
        summary.append(info)
        print(f"[ok] {system}: kept {info['n_original']} originals, "
              f"added {info['n_new_4hop_cat4']} new 4-hop Cat.4 -> {info['out_path']}")

    if args.copy_western_passthrough:
        for system in ("eskimo", "sudanese"):
            in_p = args.in_dataset_dir / f"{system}_dataset.jsonl"
            out_p = args.out_dataset_dir / f"{system}_dataset.jsonl"
            if in_p.exists():
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_bytes(in_p.read_bytes())
                summary.append({"system": system, "copied_unchanged": True,
                                "n_original": sum(1 for _ in in_p.open()),
                                "out_path": str(out_p)})
                print(f"[skip] {system}: copied unchanged ({summary[-1]['n_original']} questions)")

    # Write summary JSON
    sum_path = args.out_dataset_dir / "augmentation_summary.json"
    with sum_path.open("w") as f:
        json.dump({
            "extra_overrides_4hop": {
                sys_: list("/".join(p) for p in paths.keys())
                for sys_, paths in EXTRA_OVERRIDES_4HOP.items()
            },
            "files": summary,
        }, f, indent=2)
    print(f"\nSummary -> {sum_path}")


if __name__ == "__main__":
    main()
