# KinshipQA: A Cross-Cultural Kinship Multi-hop Reasoning Benchmark

> 📄 **Paper:** [arXiv:2601.07794](https://arxiv.org/abs/2601.07794) 
A procedurally-generated benchmark for evaluating multi-hop reasoning and
cultural-rule application in Large Language Models, covering **seven
anthropologically-documented kinship systems**.

## Overview

KinshipQA tests whether LLMs can chain biological-relation hops AND apply
culture-specific classification rules that override the biological default.
The benchmark covers Morgan's seven kinship typologies:

| System | Type | Key Rule |
|---|---|---|
| Eskimo | Descriptive | F ≠ FB (Western nuclear-family focus) |
| Sudanese | Descriptive | All terms unique |
| Hawaiian | Generational | F = FB (same-generation merging) |
| Iroquois | Bifurcate | Parallel ≠ Cross cousins |
| Dravidian | Bifurcate | Cross-cousin = potential spouse |
| Crow | Mat. Skewing | FZS = F (matrilineal generation skipping) |
| Omaha | Pat. Skewing | MBS = MB (patrilineal generation skipping) |

### Headline findings

- **40.9% accuracy drop** when reasoning shifts from biological multi-hop (Cat. 1–3) to culturally-marked classification (Cat. 4) on the five non-descriptive systems.
- Drop follows the anthropological hierarchy **descriptive > bifurcate > generational > skewing**.
- Persists under chain-of-thought, few-shot prompting, depth-matched controls, stochastic decoding, and frontier-scale evaluation (Claude Opus 4.7).
- **Compounds with chain depth**: 10.6% Other-5 Cat. 4 EM at 5–6 hops.
- **Two interventions distinguish contributing factors**:
  - Fictional-rule control (system-name + kin-term swap): +6.1% via surface-feature interaction.
  - In-context rule probe: +17.1 pp on skewing systems but **−13.4 pp** on high-baseline non-skewing systems.
- **Human baseline** (*n*=280, IAA 96.8%): humans given the same in-context rule reach **89.0%** on Other-5 Cat. 4 vs. **50.7%** for LLMs.

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pipeline/                    dataset generation and LLM evaluation
│   ├── kinshipqa_pipeline.py    end-to-end benchmark generator
│   ├── kinship_tree_generator.py  population simulator + ontology builder
│   ├── llm_tester.py            evaluation harness (Ollama / OpenAI / Anthropic / Gemini)
│   ├── few_shot_examples.json   demos for few-shot CoT
│   ├── make_fictional_cat4.py   fictional-rule dataset builder
│   └── augment_cat4_*.py        Cat. 4 path-augmentation utilities
├── data/
│   ├── kinshipqa/               main 1–4 hop dataset (3,354 questions, 7 systems)
│   ├── kinshipqa_extended_full/ 1–6 hop full extension
│   ├── kinshipqa_extended_5_6hop_aug/  5–6 hop Cat. 4 augmentation (Other-5, 249 q)
│   ├── kinshipqa_200yr/         200-year ontology variant for extended-hop runs
│   ├── fictional/
│   │   ├── full/                joint system-name + kin-term swap
│   │   ├── system_only/         orthogonal decomp: system-name only
│   │   └── term_only/           orthogonal decomp: kin-term only
│   ├── ontologies/              50-year RDF/OWL ontologies (7 systems)
│   ├── ontologies_extended/     150-year ontologies (5–6 hop Cat. 2)
│   ├── ontologies_200yr/        200-year ontologies (5–6 hop Cat. 4 on Other-5)
│   └── paraphrase_robustness/   Cat. 4 question-phrasing robustness probe
├── results/
│   ├── tables/                  LaTeX, CSV, JSON, Markdown for every paper table
│   ├── figures/                 hop-scaling curve, etc.
│   ├── cross_benchmark/         CLUTRR cross-benchmark eval results
│   ├── per_model_runs/          per-(model × protocol × dataset) accuracy summaries
│   ├── cat4_error_taxonomy.json  Cat. 4 error breakdown (P2 probe)
│   ├── headline_cis.json        95% bootstrap CIs on headline numbers
│   └── orthogonal_fictional_ablation.json  2×2 fictional decomposition
└── docs/
    ├── REPRODUCTION.md          step-by-step reproduction guide
    └── DATASHEET.md             datasheet for the benchmark
```

## Quick start

```bash
# 1. Install dependencies
pip install rdflib requests pandas numpy
# Plus your providers as needed:
pip install ollama openai anthropic google-genai

# 2. Inspect the main dataset
head -1 data/kinshipqa/eskimo_dataset.jsonl | python -m json.tool

# 3. Generate the benchmark from scratch (optional — pre-built in data/)
python pipeline/kinshipqa_pipeline.py --all \
    --ttl-dir data/ontologies/ \
    --output-dir my_datasets/

# 4. Evaluate a model (zero-shot direct)
python pipeline/llm_tester.py --all \
    --dataset-dir data/kinshipqa/ \
    --provider ollama --model gemma3:27b \
    --protocol zero_shot_direct \
    --output-dir my_results/

# 5. Few-shot CoT
python pipeline/llm_tester.py --all \
    --dataset-dir data/kinshipqa/ \
    --provider ollama --model gemma3:27b \
    --protocol few_shot_cot \
    --few-shot-file pipeline/few_shot_examples.json \
    --output-dir my_results_fewshot/
```

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for the full per-table reproduction protocol.

## Probes and ablations

| Probe | What it measures | Where the results live |
|---|---|---|
| Cat. 4 error taxonomy | 6-category breakdown of non-EM Cat. 4 errors; identifies biological-default leakage | `results/cat4_error_taxonomy.json` |
| Paraphrase robustness | Cat. 4 accuracy under rule-paraphrased questions (entity / system / kin term preserved, frame varies) | `data/paraphrase_robustness/` |
| Fictional-rule control | +6.1% Cat. 4 gain when both system label + kin terms swapped to invented strings | `data/fictional/full/` |
| Orthogonal 2×2 decomposition | Decomposes +6.1% into system-name vs kin-term factors (interaction, not additive) | `data/fictional/system_only/` + `data/fictional/term_only/`; aggregate in `results/orthogonal_fictional_ablation.json` |
| In-context rule probe | +17.1 pp on skewing; −13.4 pp on high-baseline non-skewing | `pipeline/llm_tester.py --with-rule-context` |
| Multi-seed Cat. 4 | T=0.7, n=5 stochastic decoding; preserves system-type ordering | `results/tables/multi_seed_cat4_*` |
| Bootstrap CIs | 95% percentile CIs on headline aggregates | `results/headline_cis.json` |

## Citation

See [`CITATION.cff`](CITATION.cff). A full bibliographic entry will be added upon publication.

```bibtex
@inproceedings{kinshipqa2026,
  title={KinshipQA: A Multi-Hop Kinship Reasoning Benchmark Across Anthropological Kinship Systems},
  author={[Anonymous]},
  booktitle={Submission under review},
  year={2026}
}
```

## License

[MIT](LICENSE)

## Data availability

Full per-question prediction logs (raw model responses, several MB per
(model, protocol, system) cell) are not included in this public repository
to keep clone size small. The summary statistics in
`results/per_model_runs/*/combined_results.json` are sufficient to verify
all numbers reported in the paper. Raw logs are available on request and
will be deposited on Zenodo upon publication.
