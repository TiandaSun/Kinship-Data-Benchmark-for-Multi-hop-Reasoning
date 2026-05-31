# Reproduction Guide

This guide gives step-by-step commands to reproduce every main result in the paper using the artifacts in this repository.

## Prerequisites

- Python 3.10 or newer
- `pip install rdflib requests pandas numpy`
- Plus your providers as needed:
  - Local open-source models: [Ollama](https://ollama.com/) with `gemma3:27b`, `deepseek-r1:32b`, `qwen3:32b` pulled
  - Closed-source: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` in environment

## 1. (Optional) Regenerate the datasets

The released datasets are in `data/`. To regenerate from scratch with a new seed:

```bash
# Generate ontologies (one .ttl per kinship system)
python pipeline/kinship_tree_generator.py --all --output-dir data/ontologies/

# Generate the main 1–4 hop dataset
python pipeline/kinshipqa_pipeline.py --all \
    --ttl-dir data/ontologies/ \
    --output-dir data/kinshipqa/

# 4-hop Cat. 4 path augmentation (the v6_2 increment)
python pipeline/augment_cat4_4hop.py \
    --in-ontology-dir data/ontologies/ \
    --in-dataset-dir data/kinshipqa/ \
    --out-dataset-dir data/kinshipqa/

# 5–6 hop Cat. 4 augmentation (Other-5 only)
python pipeline/augment_cat4_5_6hop.py \
    --ontologies-200yr data/ontologies_200yr/ \
    --ontologies-extended data/ontologies_extended/ \
    --output-dir data/kinshipqa_extended_5_6hop_aug/

# Fictional-rule variants (for the fictional control ablation)
python pipeline/make_fictional_cat4.py \
    --input-dir data/kinshipqa/ \
    --output-dir data/fictional/full/ \
    --mode full
python pipeline/make_fictional_cat4.py \
    --input-dir data/kinshipqa/ \
    --output-dir data/fictional/system_only/ \
    --mode system_only
python pipeline/make_fictional_cat4.py \
    --input-dir data/kinshipqa/ \
    --output-dir data/fictional/term_only/ \
    --mode term_only
```

## 2. Run an evaluation

The unified `llm_tester.py` supports three protocols (`zero_shot_direct`, `zero_shot_cot`, `few_shot_cot`) and four providers (`ollama`, `openai`, `gemini`, `anthropic`).

```bash
# Example: Gemma-3 27B, zero-shot direct, all 7 systems
python pipeline/llm_tester.py --all \
    --dataset-dir data/kinshipqa/ \
    --provider ollama --model gemma3:27b \
    --protocol zero_shot_direct \
    --output-dir my_results_gemma3_zsd/

# Example: GPT-4o, few-shot CoT
python pipeline/llm_tester.py --all \
    --dataset-dir data/kinshipqa/ \
    --provider openai --model gpt-4o \
    --protocol few_shot_cot \
    --few-shot-file pipeline/few_shot_examples.json \
    --output-dir my_results_gpt4o_fscot/

# In-context rule probe (provides the kinship rule in the prompt)
python pipeline/llm_tester.py --all \
    --dataset-dir data/kinshipqa/ \
    --provider ollama --model gemma3:27b \
    --protocol zero_shot_direct --with-rule-context \
    --output-dir my_results_with_rule/
```

Each run produces:
- `<output_dir>/<system>_results.json` — full per-question logs (large, gitignored).
- `<output_dir>/combined_results.json` — aggregated accuracy stats.

The 17 reference runs from the paper (with the CoT-parser fix applied) are in `results/per_model_runs/`, each with a `combined_results.json` you can compare your reproduction against.

## 3. Cross-protocol / cross-model comparisons

The per-model accuracy summaries you'd produce are exactly the contents of `results/per_model_runs/`. To regenerate the per-protocol comparison tables, refer to the paper's main repository for the aggregation tooling. The aggregated tables are pre-computed in this repo at `results/tables/`.

## 4. Fictional-rule ablation

```bash
# Evaluate on each fictional variant
for variant in full system_only term_only; do
    python pipeline/llm_tester.py --all \
        --dataset-dir data/fictional/$variant/ \
        --provider ollama --model gemma3:27b \
        --protocol zero_shot_direct \
        --output-dir my_results_fictional_$variant/
done
```

The pre-computed 2×2 decomposition is in `results/orthogonal_fictional_ablation.json` and `results/tables/fictional_2x2_summary.json`.

## 5. Cat. 4 error taxonomy

The Cat. 4 error taxonomy reported in Section 5.3 of the paper (6-category breakdown including the 53.3% biological-default leakage finding) is pre-computed in `results/cat4_error_taxonomy.json`.

## Notes

- Per-question raw responses (`*_results.json`) are intentionally NOT
  committed — they are reproducible from the commands above. Aggregated
  `combined_results.json` files (in `results/per_model_runs/`) are
  sufficient to verify every number in the paper.
- All scripts accept `--help` for full CLI options.
- The CLUTRR cross-benchmark eval uses the public CLUTRR dataset from
  Hugging Face — see the per-model summaries in
  `results/cross_benchmark/`.
