# results/

Pre-computed aggregated results from the paper.

## Top-level files

| File | Contents |
|---|---|
| `cat4_error_taxonomy.json` | 6-category breakdown of Cat. 4 non-EM errors (P2 probe). The 53.3% biological-default leakage finding lives here. |
| `headline_cis.json` | 95% bootstrap confidence intervals on all headline aggregates. |
| `orthogonal_fictional_ablation.json` | 2×2 decomposition of the +6.1% fictional-rule effect into system-name vs kin-term factors. |

## Subdirectories

| Directory | Contents |
|---|---|
| `tables/` | LaTeX, CSV, JSON, and Markdown for every table in the paper. |
| `figures/` | Plots — hop-scaling curve, etc. |
| `cross_benchmark/` | Per-model results on the CLUTRR cross-benchmark evaluation. |
| `per_model_runs/` | Per-(model × protocol × dataset) aggregated accuracy summaries. Each subdirectory contains a `combined_results.json`. |

## What's NOT here

Per-question raw model responses (the `*_results.json` files emitted by `pipeline/llm_tester.py`) are intentionally **not** committed — they are large and reproducible from the pipeline. See [`../docs/REPRODUCTION.md`](../docs/REPRODUCTION.md).

The 17 reference runs in `per_model_runs/` carry only the aggregated `combined_results.json` (with `parser_fix_applied: true` indicating the CoT answer-line parser fix has been applied). Raw logs are available on request and will be deposited on Zenodo upon publication.

## Tables index

| File | Paper reference |
|---|---|
| `tables/table4.{tex,csv,json}` | Headline accuracy by model × protocol. |
| `tables/table5.{tex,csv,json}` | Per-category breakdown. |
| `tables/table6.{tex,csv,json}` | Per-system × protocol matrix. |
| `tables/table7.{tex,csv,json}` | Cross-benchmark comparison. |
| `tables/cross_benchmark_table.tex` + `cross_benchmark_comparison.md` | KinshipQA vs CLUTRR. |
| `tables/protocol_matrix.csv` + `protocol_matrix_summary.json` | Backing detail for Table 6. |
| `tables/fictional_2x2_summary.json` + `fictional_2x2_table.tex` | 2×2 fictional decomposition table. |
| `tables/multi_seed_cat4_summary.json` + `multi_seed_cat4_table.tex` | Multi-seed Cat. 4 confidence intervals. |
| `tables/all7_5_6hop_summary.json` | 5–6 hop summary across all 7 systems. |
| `tables/cat4_4hop_summary.json` + `cat4_4hop_table.tex` | 4-hop Cat. 4 augmentation table. |
| `tables/fictional_control_comparison.json` | Real vs fictional Cat. 4 comparison. |
