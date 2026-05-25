#!/usr/bin/env python3
"""
Augment Cat.4 with 5-6 hop Cultural-Override Questions
=======================================================
Extends the 4-hop augmentation logic (augment_cat4_4hop.py) to 5- and
6-hop override paths.  The 4-hop script populated KINSHIP_OVERRIDES at
hop-length 4 for the five Other-5 systems (Hawaiian, Iroquois,
Dravidian, Crow, Omaha).  This script does the same at hop-lengths 5
and 6, so that Cat.4 "cultural override" coverage is no longer empty
at the deep-hop slice of Table 5.

The original v6.1 pipeline only defines biological 5-6 hop *paths*
(BIOLOGICAL_PATHS) for a limited set of (F,F,B,C,C,...) / (M,M,Z,C,C,...)
shapes.  We therefore also patch BIOLOGICAL_PATHS for the newly used
shapes so that resolve_path_to_terms() can match them.

Anthropological grounding (generation tracking deltas: F/M = +1,
B/Z = 0, C = -1):

  Hawaiian (pure generational lumping - net generation determines class):
    5-hop:
      (F,F,B,C,C)   G0  -> classificatory sibling
      (M,M,Z,C,C)   G0  -> classificatory sibling
      (F,F,F,B,C)   G+2 -> classificatory grandparent
      (M,M,M,Z,C)   G+2 -> classificatory grandparent
      (F,B,C,C,C)   G-2 -> classificatory grandchild
      (M,Z,C,C,C)   G-2 -> classificatory grandchild
    6-hop:
      (F,F,B,C,C,C) G-1 -> classificatory child
      (M,M,Z,C,C,C) G-1 -> classificatory child
      (F,F,F,B,C,C) G+1 -> classificatory parent
      (M,M,M,Z,C,C) G+1 -> classificatory parent

  Iroquois / Dravidian (parallel side stays parallel, generation tracked):
    5-hop:
      (F,F,B,C,C)   G0  -> parallel cousin (classificatory sibling)
      (M,M,Z,C,C)   G0  -> parallel cousin (classificatory sibling)
      (F,F,F,B,C)   G+2 -> classificatory paternal grandfather
      (M,M,M,Z,C)   G+2 -> classificatory maternal grandmother
    6-hop:
      (F,F,B,C,C,C) G-1 -> parallel niece/nephew (classificatory)
      (M,M,Z,C,C,C) G-1 -> parallel niece/nephew (classificatory)
      (F,F,F,B,C,C) G+1 -> classificatory father (parallel)
      (M,M,M,Z,C,C) G+1 -> classificatory mother (parallel)

  Crow (matrilineal skewing - F-matrilineage stays "father-class"):
    Following the same approximation as the existing 4-hop rule -
    any (F, ..., Z, C, C+) chain treated as skewed because the FZ-line
    is in F's matrilineage (Crow conflates generation within the line).
    5-hop:
      (F,Z,C,C,C)     -> skewed father (FZ matrilineal descendant)
      (F,F,Z,C,C)     -> skewed father (FFZ-line)
      (F,B,C,C,C)     -> classificatory grandchild (parallel side, not skewed)
    6-hop:
      (F,Z,C,C,C,C)   -> skewed father (FZ matrilineal descendant)
      (F,F,Z,C,C,C)   -> skewed father (FFZ-line)
      (F,B,C,C,C,C)   -> classificatory great-grandchild (parallel)

  Omaha (mirror of Crow, M/B swap):
    5-hop:
      (M,B,C,C,C)     -> skewed mother (MB patrilineal descendant)
      (M,M,B,C,C)     -> skewed mother (MMB-line)
      (M,Z,C,C,C)     -> classificatory grandchild (parallel side)
    6-hop:
      (M,B,C,C,C,C)   -> skewed mother (MB patrilineal descendant)
      (M,M,B,C,C,C)   -> skewed mother (MMB-line)
      (M,Z,C,C,C,C)   -> classificatory great-grandchild (parallel)

Per-system ontology selection (longer horizons needed for deep paths):
  Hawaiian, Crow, Omaha   -> ontologies_200yr/
  Iroquois, Dravidian     -> ontologies_extended/

Output goes to a NEW directory `datasets_v6_3/` so existing datasets
(v6_2, extended, 200yr) are not overwritten.

Usage (CPU only, fast):
    python augment_cat4_5_6hop.py \
        --out-dataset-dir ./datasets_v6_3/ \
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


# -----------------------------------------------------------------------------
# Biological-path extensions needed so resolve_path_to_terms() can match.
# These extend BIOLOGICAL_PATHS at runtime; we do not modify the on-disk
# pipeline file.  Each path is annotated with the biological term and hop
# count, mirroring the structure already present in the pipeline.
# -----------------------------------------------------------------------------
EXTRA_BIO_PATHS: Dict[Tuple[str, ...], Dict] = {
    # 5-hop biological paths
    ("hasFather", "hasFather", "hasBrother", "hasChild", "hasChild"):
        {"term": "father's paternal uncle's grandchild", "hops": 5},
    ("hasMother", "hasMother", "hasSister", "hasChild", "hasChild"):
        {"term": "mother's maternal aunt's grandchild", "hops": 5},
    ("hasFather", "hasFather", "hasFather", "hasBrother", "hasChild"):
        {"term": "great-grandfather's brother's child", "hops": 5},
    ("hasMother", "hasMother", "hasMother", "hasSister", "hasChild"):
        {"term": "great-grandmother's sister's child", "hops": 5},
    ("hasFather", "hasBrother", "hasChild", "hasChild", "hasChild"):
        {"term": "paternal uncle's great-grandchild", "hops": 5},
    ("hasMother", "hasSister", "hasChild", "hasChild", "hasChild"):
        {"term": "maternal aunt's great-grandchild", "hops": 5},
    # Crow skewing paths
    ("hasFather", "hasSister", "hasChild", "hasChild", "hasChild"):
        {"term": "paternal aunt's great-grandchild", "hops": 5},
    ("hasFather", "hasFather", "hasSister", "hasChild", "hasChild"):
        {"term": "father's paternal aunt's grandchild", "hops": 5},
    # Omaha skewing paths
    ("hasMother", "hasBrother", "hasChild", "hasChild", "hasChild"):
        {"term": "maternal uncle's great-grandchild", "hops": 5},
    ("hasMother", "hasMother", "hasBrother", "hasChild", "hasChild"):
        {"term": "mother's maternal uncle's grandchild", "hops": 5},

    # 6-hop biological paths
    ("hasFather", "hasFather", "hasBrother", "hasChild", "hasChild", "hasChild"):
        {"term": "father's paternal uncle's great-grandchild", "hops": 6},
    ("hasMother", "hasMother", "hasSister", "hasChild", "hasChild", "hasChild"):
        {"term": "mother's maternal aunt's great-grandchild", "hops": 6},
    ("hasFather", "hasFather", "hasFather", "hasBrother", "hasChild", "hasChild"):
        {"term": "great-grandfather's brother's grandchild", "hops": 6},
    ("hasMother", "hasMother", "hasMother", "hasSister", "hasChild", "hasChild"):
        {"term": "great-grandmother's sister's grandchild", "hops": 6},
    # Crow skewing paths
    ("hasFather", "hasSister", "hasChild", "hasChild", "hasChild", "hasChild"):
        {"term": "paternal aunt's great-great-grandchild", "hops": 6},
    ("hasFather", "hasFather", "hasSister", "hasChild", "hasChild", "hasChild"):
        {"term": "father's paternal aunt's great-grandchild", "hops": 6},
    ("hasFather", "hasBrother", "hasChild", "hasChild", "hasChild", "hasChild"):
        {"term": "paternal uncle's great-great-grandchild", "hops": 6},
    # Omaha skewing paths
    ("hasMother", "hasBrother", "hasChild", "hasChild", "hasChild", "hasChild"):
        {"term": "maternal uncle's great-great-grandchild", "hops": 6},
    ("hasMother", "hasMother", "hasBrother", "hasChild", "hasChild", "hasChild"):
        {"term": "mother's maternal uncle's great-grandchild", "hops": 6},
    ("hasMother", "hasSister", "hasChild", "hasChild", "hasChild", "hasChild"):
        {"term": "maternal aunt's great-great-grandchild", "hops": 6},
}


# -----------------------------------------------------------------------------
# Per-system 5-6 hop override entries.  Each entry mirrors the structure of
# KINSHIP_OVERRIDES (see kinshipqa_pipeline_v6.1.py lines 121-337).
# -----------------------------------------------------------------------------
EXTRA_OVERRIDES_5_6HOP: Dict[str, Dict[Tuple[str, ...], Dict]] = {
    "hawaiian": {
        # 5-hop: net generation determines class
        ("hasFather", "hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "father's paternal uncle's grandchild",
            "kin_term": "classificatory sibling",
            "kin_predicate": "kin:hasClassificatorySibling",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "mother's maternal aunt's grandchild",
            "kin_term": "classificatory sibling",
            "kin_predicate": "kin:hasClassificatorySibling",
        },
        ("hasFather", "hasFather", "hasFather", "hasBrother", "hasChild"): {
            "bio_term": "great-grandfather's brother's child",
            "kin_term": "classificatory grandparent",
            "kin_predicate": "kin:hasClassificatoryGrandparent",
        },
        ("hasMother", "hasMother", "hasMother", "hasSister", "hasChild"): {
            "bio_term": "great-grandmother's sister's child",
            "kin_term": "classificatory grandparent",
            "kin_predicate": "kin:hasClassificatoryGrandparent",
        },
        ("hasFather", "hasBrother", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "paternal uncle's great-grandchild",
            "kin_term": "classificatory grandchild",
            "kin_predicate": "kin:hasClassificatoryGrandchild",
        },
        ("hasMother", "hasSister", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "maternal aunt's great-grandchild",
            "kin_term": "classificatory grandchild",
            "kin_predicate": "kin:hasClassificatoryGrandchild",
        },
        # 6-hop
        ("hasFather", "hasFather", "hasBrother", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "father's paternal uncle's great-grandchild",
            "kin_term": "classificatory child",
            "kin_predicate": "kin:hasClassificatoryChild",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "mother's maternal aunt's great-grandchild",
            "kin_term": "classificatory child",
            "kin_predicate": "kin:hasClassificatoryChild",
        },
        ("hasFather", "hasFather", "hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "great-grandfather's brother's grandchild",
            "kin_term": "classificatory parent",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasMother", "hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "great-grandmother's sister's grandchild",
            "kin_term": "classificatory parent",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
    },

    "iroquois": {
        # 5-hop parallel chains (all same-sex sibling links)
        ("hasFather", "hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "father's paternal uncle's grandchild",
            "kin_term": "parallel cousin (classificatory sibling)",
            "kin_predicate": "kin:hasParallelCousin",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "mother's maternal aunt's grandchild",
            "kin_term": "parallel cousin (classificatory sibling)",
            "kin_predicate": "kin:hasParallelCousin",
        },
        ("hasFather", "hasFather", "hasFather", "hasBrother", "hasChild"): {
            "bio_term": "great-grandfather's brother's child",
            "kin_term": "classificatory paternal grandfather (parallel)",
            "kin_predicate": "kin:hasClassificatoryGrandparent",
        },
        ("hasMother", "hasMother", "hasMother", "hasSister", "hasChild"): {
            "bio_term": "great-grandmother's sister's child",
            "kin_term": "classificatory maternal grandmother (parallel)",
            "kin_predicate": "kin:hasClassificatoryGrandparent",
        },
        # 6-hop parallel chains
        ("hasFather", "hasFather", "hasBrother", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "father's paternal uncle's great-grandchild",
            "kin_term": "parallel niece/nephew (classificatory)",
            "kin_predicate": "kin:hasParallelNieceNephew",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "mother's maternal aunt's great-grandchild",
            "kin_term": "parallel niece/nephew (classificatory)",
            "kin_predicate": "kin:hasParallelNieceNephew",
        },
        ("hasFather", "hasFather", "hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "great-grandfather's brother's grandchild",
            "kin_term": "classificatory father (parallel)",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasMother", "hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "great-grandmother's sister's grandchild",
            "kin_term": "classificatory mother (parallel)",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
    },

    "dravidian": {
        # Same parallel-side logic as Iroquois (Tamil-flavoured terms)
        ("hasFather", "hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "father's paternal uncle's grandchild",
            "kin_term": "parallel cousin (classificatory sibling)",
            "kin_predicate": "kin:hasParallelCousin",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "mother's maternal aunt's grandchild",
            "kin_term": "parallel cousin (classificatory sibling)",
            "kin_predicate": "kin:hasParallelCousin",
        },
        ("hasFather", "hasFather", "hasFather", "hasBrother", "hasChild"): {
            "bio_term": "great-grandfather's brother's child",
            "kin_term": "classificatory paternal grandfather (parallel, Periya-Thatha)",
            "kin_predicate": "kin:hasClassificatoryGrandparent",
        },
        ("hasMother", "hasMother", "hasMother", "hasSister", "hasChild"): {
            "bio_term": "great-grandmother's sister's child",
            "kin_term": "classificatory maternal grandmother (parallel, Periya-Patti)",
            "kin_predicate": "kin:hasClassificatoryGrandparent",
        },
        # 6-hop parallel chains
        ("hasFather", "hasFather", "hasBrother", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "father's paternal uncle's great-grandchild",
            "kin_term": "parallel niece/nephew (classificatory)",
            "kin_predicate": "kin:hasParallelNieceNephew",
        },
        ("hasMother", "hasMother", "hasSister", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "mother's maternal aunt's great-grandchild",
            "kin_term": "parallel niece/nephew (classificatory)",
            "kin_predicate": "kin:hasParallelNieceNephew",
        },
        ("hasFather", "hasFather", "hasFather", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "great-grandfather's brother's grandchild",
            "kin_term": "classificatory father (parallel, Periyappa/Chithappa)",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
        ("hasMother", "hasMother", "hasMother", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "great-grandmother's sister's grandchild",
            "kin_term": "classificatory mother (parallel, Chitti/Periyamma)",
            "kin_predicate": "kin:hasClassificatoryParent",
        },
    },

    "crow": {
        # Matrilineal skewing extended.  Following the existing 4-hop
        # convention: any (F, ..., Z, C+) chain is treated as skewed.
        # The FZ-line is in F's matrilineage; Crow classifies all
        # matrilineally-descended members of F's matrilineage as
        # belonging to F's generation ("father / female father").
        # 5-hop
        ("hasFather", "hasSister", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "paternal aunt's great-grandchild",
            "kin_term": "father/female father (skewed - same matrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        ("hasFather", "hasFather", "hasSister", "hasChild", "hasChild"): {
            "bio_term": "father's paternal aunt's grandchild",
            "kin_term": "father/female father (skewed - FFZ matrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        # Parallel side (FB-line) is non-skewed - generational class only
        ("hasFather", "hasBrother", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "paternal uncle's great-grandchild",
            "kin_term": "classificatory grandchild (parallel)",
            "kin_predicate": "kin:hasClassificatoryGrandchild",
        },
        # 6-hop
        ("hasFather", "hasSister", "hasChild", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "paternal aunt's great-great-grandchild",
            "kin_term": "father/female father (skewed - same matrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        ("hasFather", "hasFather", "hasSister", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "father's paternal aunt's great-grandchild",
            "kin_term": "father/female father (skewed - FFZ matrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        ("hasFather", "hasBrother", "hasChild", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "paternal uncle's great-great-grandchild",
            "kin_term": "classificatory great-grandchild (parallel)",
            "kin_predicate": "kin:hasClassificatoryGrandchild",
        },
    },

    "omaha": {
        # Patrilineal skewing extended - mirror of Crow.
        # 5-hop
        ("hasMother", "hasBrother", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "maternal uncle's great-grandchild",
            "kin_term": "mother/male mother (skewed - same patrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        ("hasMother", "hasMother", "hasBrother", "hasChild", "hasChild"): {
            "bio_term": "mother's maternal uncle's grandchild",
            "kin_term": "mother/male mother (skewed - MMB patrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        # Parallel side (MZ-line) is non-skewed
        ("hasMother", "hasSister", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "maternal aunt's great-grandchild",
            "kin_term": "classificatory grandchild (parallel)",
            "kin_predicate": "kin:hasClassificatoryGrandchild",
        },
        # 6-hop
        ("hasMother", "hasBrother", "hasChild", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "maternal uncle's great-great-grandchild",
            "kin_term": "mother/male mother (skewed - same patrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        ("hasMother", "hasMother", "hasBrother", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "mother's maternal uncle's great-grandchild",
            "kin_term": "mother/male mother (skewed - MMB patrilineage)",
            "kin_predicate": "kin:hasSkewedGeneration",
        },
        ("hasMother", "hasSister", "hasChild", "hasChild", "hasChild", "hasChild"): {
            "bio_term": "maternal aunt's great-great-grandchild",
            "kin_term": "classificatory great-grandchild (parallel)",
            "kin_predicate": "kin:hasClassificatoryGrandchild",
        },
    },
}


# Per-system ontology directories.  Longer horizons are required for
# the deeper systems (5-6 hops need enough generations to be realisable).
PER_SYSTEM_ONTOLOGY: Dict[str, str] = {
    "hawaiian": "ontologies_200yr",
    "crow":     "ontologies_200yr",
    "omaha":    "ontologies_200yr",
    "iroquois": "ontologies_extended",
    "dravidian":"ontologies_extended",
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


def patch_biological_paths(pipeline_mod) -> None:
    """Extend BIOLOGICAL_PATHS so resolve_path_to_terms() can match new shapes."""
    for path, info in EXTRA_BIO_PATHS.items():
        if path not in pipeline_mod.BIOLOGICAL_PATHS:
            pipeline_mod.BIOLOGICAL_PATHS[path] = info


def patch_overrides(pipeline_mod) -> None:
    """Inject 5-6 hop override paths into the pipeline's KINSHIP_OVERRIDES."""
    for system, extra in EXTRA_OVERRIDES_5_6HOP.items():
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
                ontology_dir: Path,
                out_dataset_dir: Path,
                count_per_hop: int, seed: int) -> dict:
    """Generate 5- and 6-hop Cat.4 cultural-override questions for one system."""
    ttl_path = ontology_dir / f"{system}.ttl"
    if not ttl_path.exists():
        raise FileNotFoundError(f"Missing ontology: {ttl_path}")

    kg = pipeline_mod.KinshipGraph(str(ttl_path), system)
    qg = pipeline_mod.QuestionGenerator(kg, system, seed=seed)

    # Reserve a non-overlapping ID range.  v6_2 used 0501-0700 for 4-hop;
    # we start at 0801 to leave a clear gap.
    qg.question_counter = 800

    sys_overrides = EXTRA_OVERRIDES_5_6HOP[system]

    new_questions = []
    per_hop_counts = {}
    for n_hops in (5, 6):
        paths_at_hop = [p for p in sys_overrides.keys()
                        if pipeline_mod.BIOLOGICAL_PATHS.get(p, {}).get("hops") == n_hops]
        if not paths_at_hop:
            per_hop_counts[n_hops] = 0
            continue
        qs = qg._generate_cat4_for_hops(paths_at_hop, n_hops, count_per_hop)
        new_questions.extend(qs)
        per_hop_counts[n_hops] = len(qs)

    # Write a fresh dataset file containing ONLY the new 5-6 hop Cat.4
    # additions.  We deliberately do NOT copy the existing 1-4 hop
    # questions here; downstream eval can either evaluate this slice
    # alone, or merge it with the existing dataset offline.  This keeps
    # the new file small and clearly labelled as the 5-6 hop slice.
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dataset_dir / f"{system}_dataset.jsonl"
    with out_path.open("w") as outf:
        for q in new_questions:
            outf.write(json.dumps(question_to_dict(q)) + "\n")

    return {
        "system": system,
        "ontology": str(ttl_path),
        "n_new_5hop": per_hop_counts.get(5, 0),
        "n_new_6hop": per_hop_counts.get(6, 0),
        "n_total": len(new_questions),
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dataset-dir", default="./datasets_v6_3/", type=Path)
    ap.add_argument("--questions-per-hop", type=int, default=25,
                    help="Target number of new Cat.4 questions PER HOP-LENGTH "
                         "(so total per system is up to 2*this).")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    pipeline_mod = load_pipeline_module()
    patch_biological_paths(pipeline_mod)
    patch_overrides(pipeline_mod)

    summary = []
    for system in EXTRA_OVERRIDES_5_6HOP.keys():
        onto_dir = HERE / PER_SYSTEM_ONTOLOGY[system]
        info = augment_one(
            pipeline_mod=pipeline_mod,
            system=system,
            ontology_dir=onto_dir,
            out_dataset_dir=args.out_dataset_dir,
            count_per_hop=args.questions_per_hop,
            seed=args.seed,
        )
        summary.append(info)
        print(f"[ok] {system}: ontology={info['ontology']} "
              f"5-hop={info['n_new_5hop']} 6-hop={info['n_new_6hop']} "
              f"total={info['n_total']} -> {info['out_path']}")

    sum_path = args.out_dataset_dir / "augmentation_summary_5_6hop.json"
    with sum_path.open("w") as f:
        json.dump({
            "per_system_ontology": PER_SYSTEM_ONTOLOGY,
            "extra_overrides_5_6hop": {
                sys_: ["/".join(p) for p in paths.keys()]
                for sys_, paths in EXTRA_OVERRIDES_5_6HOP.items()
            },
            "files": summary,
        }, f, indent=2)
    print(f"\nSummary -> {sum_path}")


if __name__ == "__main__":
    main()
