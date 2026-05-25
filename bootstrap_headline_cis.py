#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals for the four headline numerical claims in
the KinshipQA paper (acl_latex_v3.tex), plus a recompute of the Cat.1 average
row in Table 4 (the row a reviewer flagged as inconsistent).

Output:
  /mnt/scratch/users/ts1201/Family-Tree-reasoning/familytree_v6/headline_cis.json

Notes
-----
* All bootstrapping uses N=10000 resamples and the 2.5/97.5 percentile method.
* The three open-source models with full per-question data are:
    gemma3_27b, deepseek_r1_32b, qwen3_32b
  Their per-question results live in
    results_<model>_zero_shot_direct/<system>_results.json
  with one entry per question containing keys
    category, n_hops, kinship_system, exact_match, has_cultural_override
* "Other-5" = hawaiian, iroquois, dravidian, crow, omaha
  "Esk&Sud" = eskimo, sudanese
* Per-question field `has_cultural_override` is set at the path level: True iff
  the bio path resolves through the system's override library (so on Other-5
  systems it includes all Cat.4 questions plus those Cat.2 questions whose
  asked term has an override mapping). The paper's "23.6pp Other-5 avg" gap
  is computed using exactly this field across all questions on Other-5 (not
  within Cat.4 only). We follow the paper's actual aggregation here so the
  point estimate is reproducible from the same data.
* For the fictional vs real comparison we use unpaired bootstrap (different
  question instances after system/term replacement).
* Cross-model averaging strategy: for each bootstrap resample we resample
  questions independently *within each model* and compute the model's mean,
  then average the three model means; the reported CI is the 2.5/97.5
  percentile of these averaged-mean differences across the 10000 resamples.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
from typing import Dict, List, Tuple

random.seed(20260507)

BASE = "/mnt/scratch/users/ts1201/Family-Tree-reasoning/familytree_v6"
OUT = os.path.join(BASE, "headline_cis.json")

OPEN_MODELS = ["gemma3_27b", "deepseek_r1_32b", "qwen3_32b"]
ALL_SYSTEMS = ["eskimo", "sudanese", "hawaiian", "iroquois", "dravidian", "crow", "omaha"]
OTHER5 = ["hawaiian", "iroquois", "dravidian", "crow", "omaha"]
ESKSUD = ["eskimo", "sudanese"]

N_BOOT = 10000


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_questions(results_dir: str, system: str) -> List[Dict]:
    path = os.path.join(BASE, results_dir, f"{system}_results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        d = json.load(f)
    return d["questions"]


# ----------------------------------------------------------------------------
# Bootstrap helpers
# ----------------------------------------------------------------------------
def boot_mean(values: List[int], rng: random.Random) -> float:
    n = len(values)
    s = 0
    for _ in range(n):
        s += values[rng.randrange(n)]
    return s / n


def percentile(xs: List[float], p: float) -> float:
    xs_sorted = sorted(xs)
    if not xs_sorted:
        return float("nan")
    k = (len(xs_sorted) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def ci(values: List[float]) -> Tuple[float, float]:
    return percentile(values, 0.025), percentile(values, 0.975)


# ----------------------------------------------------------------------------
# Task 1: Within-(Other-5) override gap (the "23.6pp" claim).
#   Per the paper's actual aggregation, the gap is between
#   has_cultural_override=True vs has_cultural_override=False questions
#   on Other-5 systems, pooled across all categories. (The table caption
#   labels this a Cat.4 split but the underlying numbers come from the
#   summary["by_cultural_override"] field, which uses all questions.)
# ----------------------------------------------------------------------------
def collect_override_split(model: str) -> Tuple[List[int], List[int]]:
    em_with, em_without = [], []
    for s in OTHER5:
        for q in load_questions(f"results_{model}_zero_shot_direct", s):
            em = int(q["exact_match"])
            if q["has_cultural_override"]:
                em_with.append(em)
            else:
                em_without.append(em)
    return em_with, em_without


def task1_override_gap() -> Dict:
    rng = random.Random(20260507)
    per_model_data = {m: collect_override_split(m) for m in OPEN_MODELS}
    point_per_model = {}
    for m, (w, wo) in per_model_data.items():
        point_per_model[m] = {
            "with_override_em": sum(w) / len(w) * 100,
            "without_override_em": sum(wo) / len(wo) * 100,
            "gap_pp": (sum(wo) / len(wo) - sum(w) / len(w)) * 100,
            "n_with": len(w),
            "n_without": len(wo),
        }
    point_avg_gap = sum(d["gap_pp"] for d in point_per_model.values()) / 3

    # Bootstrap: each resample, draw within each (model, group) independently;
    # average the three per-model gaps to get one bootstrap statistic.
    boot_avgs = []
    for _ in range(N_BOOT):
        gaps = []
        for m in OPEN_MODELS:
            w, wo = per_model_data[m]
            mean_w = boot_mean(w, rng)
            mean_wo = boot_mean(wo, rng)
            gaps.append((mean_wo - mean_w) * 100)
        boot_avgs.append(sum(gaps) / 3)
    lo, hi = ci(boot_avgs)
    return {
        "point_paper": 23.6,
        "point_recomputed": point_avg_gap,
        "ci_low": lo,
        "ci_high": hi,
        "n_questions_per_model": {
            m: {"with": d["n_with"], "without": d["n_without"]}
            for m, d in point_per_model.items()
        },
        "per_model_point": point_per_model,
        "method": (
            "Unpaired bootstrap, N=10000. Within each model, resample questions "
            "with replacement separately within has_cultural_override=True vs "
            "False groups on Other-5 systems; compute mean EM difference; average "
            "across the 3 open-source models for each bootstrap iteration. CI is "
            "the 2.5/97.5 percentile of the resulting averaged-gap distribution. "
            "NOTE: the 23.6pp point in the paper was computed from a 6-model "
            "average (3 open + GPT-4o-mini, Gemini-2.5-Flash, Claude-3.5-Haiku); "
            "restricted to the 3 open-source models the gap is larger (~30pp)."
        ),
    }


# ----------------------------------------------------------------------------
# Task 2: Fictional vs real Cat.4 gap (the "+5.1pp" claim).
#   Compare per-question Cat.4 EM on Other-5 systems between
#     results_fictional_<model>_zero_shot_direct/  (fictional system/term names)
#   and
#     results_<model>_zero_shot_direct/            (real)
# ----------------------------------------------------------------------------
def task2_fictional_gap() -> Dict:
    rng = random.Random(20260508)
    per_model_data = {}
    for m in OPEN_MODELS:
        real, fict = [], []
        for s in OTHER5:
            for q in load_questions(f"results_{m}_zero_shot_direct", s):
                if q["category"] == 4:
                    real.append(int(q["exact_match"]))
            for q in load_questions(f"results_fictional_{m}_zero_shot_direct", s):
                if q["category"] == 4:
                    fict.append(int(q["exact_match"]))
        per_model_data[m] = (real, fict)

    point_per_model = {}
    for m, (real, fict) in per_model_data.items():
        point_per_model[m] = {
            "real_em": sum(real) / len(real) * 100,
            "fict_em": sum(fict) / len(fict) * 100,
            "gap_pp": (sum(fict) / len(fict) - sum(real) / len(real)) * 100,
            "n_real": len(real),
            "n_fict": len(fict),
        }
    point_avg_gap = sum(d["gap_pp"] for d in point_per_model.values()) / 3

    boot_avgs = []
    for _ in range(N_BOOT):
        gaps = []
        for m in OPEN_MODELS:
            real, fict = per_model_data[m]
            mean_r = boot_mean(real, rng)
            mean_f = boot_mean(fict, rng)
            gaps.append((mean_f - mean_r) * 100)
        boot_avgs.append(sum(gaps) / 3)
    lo, hi = ci(boot_avgs)
    return {
        "point_paper": 5.1,
        "point_recomputed": point_avg_gap,
        "ci_low": lo,
        "ci_high": hi,
        "per_model_point": point_per_model,
        "method": (
            "Unpaired bootstrap, N=10000 (different question instances after "
            "system/term replacement, so paired pairing is not appropriate). "
            "Within each model, resample real and fictional Cat.4 questions on "
            "Other-5 systems with replacement separately; compute mean EM "
            "difference (fictional - real); average across the 3 open-source "
            "models per iteration. CI is 2.5/97.5 percentile."
        ),
    }


# ----------------------------------------------------------------------------
# Task 3: Hop-matched Cat.2 vs Cat.4 gap at 2-hop and 3-hop (Table 8 of the
# paper, line 556 of acl_latex_v3.tex).
#   Cat.2 questions exist on all 7 systems; Cat.4 questions in this analysis
#   are restricted to Other-5 (where cultural override applies; Esk&Sud have
#   no Cat.4 in the v6.1 dataset version that holds 2/3-hop Cat.4 paths).
# ----------------------------------------------------------------------------
def collect_cat_hop(model: str, cat: int, hop: int, systems: List[str]) -> List[int]:
    out = []
    for s in systems:
        for q in load_questions(f"results_{model}_zero_shot_direct", s):
            if q["category"] == cat and q["n_hops"] == hop:
                out.append(int(q["exact_match"]))
    return out


def task3_hop_matched(hop: int, point_paper: float) -> Dict:
    rng = random.Random(20260509 + hop)
    per_model_data = {}
    for m in OPEN_MODELS:
        cat2 = collect_cat_hop(m, 2, hop, ALL_SYSTEMS)
        cat4 = collect_cat_hop(m, 4, hop, OTHER5)
        per_model_data[m] = (cat2, cat4)

    point_per_model = {}
    for m, (c2, c4) in per_model_data.items():
        point_per_model[m] = {
            "cat2_em": sum(c2) / len(c2) * 100 if c2 else float("nan"),
            "cat4_em": sum(c4) / len(c4) * 100 if c4 else float("nan"),
            "gap_pp": (sum(c2) / len(c2) - sum(c4) / len(c4)) * 100 if c2 and c4 else float("nan"),
            "n_cat2": len(c2),
            "n_cat4": len(c4),
        }
    point_avg_gap = sum(d["gap_pp"] for d in point_per_model.values()) / 3

    boot_avgs = []
    for _ in range(N_BOOT):
        gaps = []
        for m in OPEN_MODELS:
            c2, c4 = per_model_data[m]
            mean_c2 = boot_mean(c2, rng)
            mean_c4 = boot_mean(c4, rng)
            gaps.append((mean_c2 - mean_c4) * 100)
        boot_avgs.append(sum(gaps) / 3)
    lo, hi = ci(boot_avgs)
    return {
        "point_paper": point_paper,
        "point_recomputed": point_avg_gap,
        "ci_low": lo,
        "ci_high": hi,
        "per_model_point": point_per_model,
        "method": (
            f"Unpaired bootstrap, N=10000. Within each model, separately resample "
            f"Cat.2 questions (at {hop}-hop, all 7 systems) and Cat.4 questions "
            f"(at {hop}-hop, Other-5 systems) with replacement; compute mean EM "
            f"difference (Cat.2 - Cat.4); average across the 3 open-source "
            f"models per iteration. CI is 2.5/97.5 percentile."
        ),
    }


# ----------------------------------------------------------------------------
# Task 4: Recompute Cat.1 average row in Table 4 (5 models, excluding the
# zero-shot-CoT-only Claude-Haiku-4.5).
# ----------------------------------------------------------------------------
def task4_cat1_average() -> Dict:
    model_dirs = [
        ("Qwen3-32B", "results_qwen3_32b_zero_shot_direct"),
        ("Gemma3-27B", "results_gemma3_27b_zero_shot_direct"),
        ("DeepSeek-R1-32B", "results_deepseek_r1_32b_zero_shot_direct"),
        ("GPT-4o-mini", "results_gpt4o"),
        ("Gemini-2.5-Flash", "results_gemini"),
    ]
    per_model = {}
    for name, d in model_dirs:
        ems = []
        for s in ALL_SYSTEMS:
            for q in load_questions(d, s):
                if q["category"] == 1:
                    ems.append(int(q["exact_match"]))
        per_model[name] = {
            "cat1_em": sum(ems) / len(ems) * 100,
            "n": len(ems),
        }
    avg = sum(d["cat1_em"] for d in per_model.values()) / len(per_model)
    return {
        "reported_in_table_v3": 97.7,
        "reviewer_flagged_value": 98.1,
        "recomputed_5model": round(avg, 1),
        "models_included": [name for name, _ in model_dirs],
        "per_model": per_model,
        "method": (
            "Per-question Cat.1 exact-match rate computed for each of the 5 "
            "models with zero-shot-direct numbers (Qwen3-32B, Gemma3-27B, "
            "DeepSeek-R1-32B from results_<model>_zero_shot_direct/; GPT-4o-mini "
            "from results_gpt4o/ and Gemini-2.5-Flash from results_gemini/, "
            "which contain the only zero-shot-direct closed-source data). "
            "Cat.1 average is the unweighted mean of the 5 per-model rates."
        ),
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    results = {
        "override_gap": task1_override_gap(),
        "fictional_real_gap": task2_fictional_gap(),
        "hop_matched_2hop": task3_hop_matched(hop=2, point_paper=38.0),
        "hop_matched_3hop": task3_hop_matched(hop=3, point_paper=56.3),
        "cat1_average": task4_cat1_average(),
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {OUT}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
