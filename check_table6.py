"""
Verify the Table 6 (Performance by kinship system) numbers in v3 against
several plausible source data configurations, to identify which one (if any)
reproduces the published values.

Published Table 6 (from v3, also identical to arxiv v1):
  Eskimo    All Cat. 94.3, w/o Ov. 93.8
  Sudanese  All Cat. 94.2, w/o Ov. 93.8
  Hawaiian  All Cat. 82.8, w/Ov. 72.4, w/o Ov. 90.2, Δ 17.7
  Iroquois  All Cat. 80.2, w/Ov. 68.5, w/o Ov. 88.7, Δ 20.2
  Dravidian All Cat. 84.0, w/Ov. 78.5, w/o Ov. 88.0, Δ  9.6
  Crow      All Cat. 80.6, w/Ov. 60.9, w/o Ov. 91.4, Δ 30.5
  Omaha     All Cat. 77.4, w/Ov. 51.9, w/o Ov. 91.7, Δ 39.8
  Other-5 avg.       w/Ov. 66.4, w/o Ov. 90.0, Δ 23.6

Caption: "mean ± std across 3 open-source models".
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEMS = ["eskimo","sudanese","hawaiian","iroquois","dravidian","crow","omaha"]
OPEN = ["gemma3_27b", "deepseek_r1_32b", "qwen3_32b"]

PUBLISHED_ALL = {"eskimo":94.3,"sudanese":94.2,"hawaiian":82.8,"iroquois":80.2,"dravidian":84.0,"crow":80.6,"omaha":77.4}
PUBLISHED_W   = {"hawaiian":72.4,"iroquois":68.5,"dravidian":78.5,"crow":60.9,"omaha":51.9}
PUBLISHED_WO  = {"eskimo":93.8,"sudanese":93.8,"hawaiian":90.2,"iroquois":88.7,"dravidian":88.0,"crow":91.4,"omaha":91.7}


def load(d, s):
    p = ROOT / d / f"{s}_results.json"
    return json.loads(p.read_text()).get("questions", []) if p.exists() else []


def em_pct(qs):
    qs = [q for q in qs if q.get("exact_match") is not None]
    return 100*sum(bool(q["exact_match"]) for q in qs)/len(qs) if qs else None


def check_config(label, dir_template, q_filter):
    """Run aggregation under one configuration and report drift vs published."""
    print(f"\n=== {label} ===")
    print(f"  dirs: {dir_template}")
    drift_all = []
    for s in SYSTEMS:
        per_model = []
        for m in OPEN:
            d = dir_template.format(model=m)
            qs = q_filter(load(d, s))
            v = em_pct(qs)
            if v is not None: per_model.append(v)
        if not per_model:
            continue
        mean = sum(per_model)/len(per_model)
        pub = PUBLISHED_ALL.get(s)
        drift = abs(mean - pub) if pub else None
        if drift is not None: drift_all.append(drift)
        flag = "✓" if (drift is not None and drift < 0.5) else ("≈" if (drift is not None and drift < 2) else "✗")
        pub_str = f" (pub {pub})" if pub else ""
        print(f"  {s:<10}: recomputed {mean:5.1f}{pub_str}  {flag}")
    if drift_all:
        print(f"  mean drift: {sum(drift_all)/len(drift_all):.2f}pp (max {max(drift_all):.2f}pp)")


def check_split(label, dir_template, hop_filter):
    """Recover the w/Ov and w/o Ov columns under various assumptions."""
    print(f"\n=== {label}: w/Ov vs w/o Ov split ===")
    # Method A: split Cat.4 by has_cultural_override field
    print("  Method A — split Cat.4 by has_cultural_override field:")
    for s in ["hawaiian","iroquois","dravidian","crow","omaha"]:
        per_model_w, per_model_wo = [], []
        for m in OPEN:
            d = dir_template.format(model=m)
            cat4 = [q for q in load(d, s)
                    if q.get("category") == 4
                    and (hop_filter is None or q.get("n_hops") in hop_filter)]
            w  = [q for q in cat4 if q.get("has_cultural_override") is True]
            wo = [q for q in cat4 if q.get("has_cultural_override") is False]
            v_w  = em_pct(w)
            v_wo = em_pct(wo)
            if v_w  is not None: per_model_w.append(v_w)
            if v_wo is not None: per_model_wo.append(v_wo)
        ew = sum(per_model_w)/len(per_model_w) if per_model_w else None
        ewo = sum(per_model_wo)/len(per_model_wo) if per_model_wo else None
        pw, pwo = PUBLISHED_W.get(s), PUBLISHED_WO.get(s)
        ew_str = "--" if ew is None else f"{ew:5.1f}"
        ewo_str = "--" if ewo is None else f"{ewo:5.1f}"
        print(f"    {s:<10}: w/Ov={ew_str} (pub {pw}), w/o Ov={ewo_str} (pub {pwo})")
    # Method B: w/o Ov from Cat.1+Cat.2+Cat.3 (biological), w/Ov from Cat.4
    print("\n  Method B — w/o Ov = Cat.1+2+3 mean, w/Ov = Cat.4 mean (per Other-5 system):")
    for s in ["hawaiian","iroquois","dravidian","crow","omaha"]:
        per_model_bio, per_model_cat4 = [], []
        for m in OPEN:
            d = dir_template.format(model=m)
            qs = load(d, s)
            bio = [q for q in qs if q.get("category") in (1,2,3)
                   and (hop_filter is None or q.get("n_hops") in hop_filter)]
            cat4 = [q for q in qs if q.get("category") == 4
                    and (hop_filter is None or q.get("n_hops") in hop_filter)]
            v_bio = em_pct(bio)
            v_cat4 = em_pct(cat4)
            if v_bio is not None: per_model_bio.append(v_bio)
            if v_cat4 is not None: per_model_cat4.append(v_cat4)
        e_bio = sum(per_model_bio)/len(per_model_bio) if per_model_bio else None
        e_cat4 = sum(per_model_cat4)/len(per_model_cat4) if per_model_cat4 else None
        pw, pwo = PUBLISHED_W.get(s), PUBLISHED_WO.get(s)
        e_cat4_str = "--" if e_cat4 is None else f"{e_cat4:5.1f}"
        e_bio_str = "--" if e_bio is None else f"{e_bio:5.1f}"
        print(f"    {s:<10}: w/Ov(=Cat.4)={e_cat4_str} (pub {pw}), w/o Ov(=Cat.1-3)={e_bio_str} (pub {pwo})")


# === 1. All-Cat column verification ===
check_config(
    "Config 1: results_<model>_zero_shot_direct/, ALL questions, no augmentation",
    "results_{model}_zero_shot_direct",
    lambda qs: qs,
)

check_config(
    "Config 2: results_4hop_<model>_zero_shot_direct/, ALL questions, with v6.2 augmentation",
    "results_4hop_{model}_zero_shot_direct",
    lambda qs: qs,
)

check_config(
    "Config 3: results_<model>_zero_shot_direct/, exclude Cat.4 (Cat.1-3 only)",
    "results_{model}_zero_shot_direct",
    lambda qs: [q for q in qs if q.get("category") != 4],
)

check_config(
    "Config 4: results_<model>_zero_shot_direct/, hops 1-3 only (no 4-hop)",
    "results_{model}_zero_shot_direct",
    lambda qs: [q for q in qs if q.get("n_hops") in (1,2,3)],
)

# === 2. w/Ov and w/o Ov column verification ===
check_split(
    "Config 1: results_<model>_zero_shot_direct/",
    "results_{model}_zero_shot_direct",
    None,
)
check_split(
    "Config 1b: results_<model>_zero_shot_direct/, hops 2-3 only",
    "results_{model}_zero_shot_direct",
    [2, 3],
)
check_split(
    "Config 2: results_4hop_<model>_zero_shot_direct/ (with augmentation)",
    "results_4hop_{model}_zero_shot_direct",
    None,
)
check_split(
    "Config 2b: results_4hop_<model>_zero_shot_direct/, hops 2-3 only",
    "results_4hop_{model}_zero_shot_direct",
    [2, 3],
)
