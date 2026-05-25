# KinshipQA: A Cross-Cultural Kinship Multi-hop Reasoning Benchmark

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

### Headline findings (v6)

- **40.9% accuracy drop** when reasoning shifts from biological multi-hop (Cat.~1--3) to culturally-marked classification (Cat.~4) on the five non-descriptive systems
- Drop follows the anthropological hierarchy **descriptive > bifurcate > generational > skewing**
- Persists under chain-of-thought, few-shot prompting, depth-matched controls, stochastic decoding, and frontier-scale evaluation (Claude Opus 4.7)
- **Compounds with chain depth**: 10.6% Other-5 Cat.~4 EM at 5--6 hops
- **Two interventions distinguish contributing factors**:
  - Fictional-rule control (system-name + kin-term swap): +6.1% via surface-feature interaction
  - In-context rule probe: +17.1 pp on skewing systems but **−13.4 pp** on high-baseline non-skewing systems
- **Human baseline** (*n*=280, IAA 96.8%): humans given the same in-context rule reach **89.0%** on Other-5 Cat.~4 vs. **50.7%** for LLMs

## Repository Structure

```
.
├── kinship_tree_generator.py        # Population simulator + ontology builder
├── kinshipqa_pipeline.py            # End-to-end benchmark generator
├── llm_tester.py                    # Evaluation harness (Ollama / OpenAI / Anthropic / Gemini)
├── few_shot_examples.json           # Demos for few-shot CoT
│
├── ontologies/                      # 50-year RDF/OWL ontologies (7 systems)
├── ontologies_extended/             # 150-year ontologies (5–6 hop Cat.2)
├── ontologies_200yr/                # 200-year ontologies (5–6 hop Cat.4 on Other-5)
│
├── datasets_v6_2/                   # Main 1–4 hop dataset (3,354 questions)
├── datasets_extended/               # 1–6 hop full extension
├── datasets_v6_3/                   # 5–6 hop Cat.4 augmentation (Other-5, 249 q)
├── datasets_fictional/              # Fictional-rule control (joint swap)
├── datasets_fictional_system_only/  # Orthogonal decomp: system-name swap only
├── datasets_fictional_term_only/    # Orthogonal decomp: kin-term swap only
│
├── paraphrase_robustness/           # Cat.4 question-phrasing robustness probe
│
├── results_*/                       # Evaluation summaries (per-protocol per-model)
│   └── combined_results.json        # Aggregate stats only — full per-question logs
│                                    #   available on request (omitted from public repo)
│
├── jobs/                            # SLURM scripts (Viking HPC; account: cs-ontrel-2021)
│
├── error_taxonomy_cat4.py           # Cat.4 error classifier (P2)
├── cat4_error_taxonomy.json         # Output: 6-category breakdown (53.3% biological-default leakage)
├── make_fictional_cat4.py           # Fictional-rule dataset builder
├── compare_fictional_to_real.py     # Fictional control analysis
├── aggregate_fictional_2x2.py       # Orthogonal 2×2 decomposition
├── orthogonal_fictional_ablation.json
├── multi_seed_cat4.py               # Multi-seed (T=0.7, n=5) Cat.4 evaluator
├── aggregate_multi_seed.py          # Multi-seed aggregation
├── reaggregate_multi_seed.py
├── bootstrap_headline_cis.py        # 95% bootstrap CIs on headline numbers
├── headline_cis.json
├── augment_cat4_4hop.py             # 4-hop Cat.4 path augmentation
├── augment_cat4_5_6hop.py           # 5–6 hop Cat.4 path augmentation
├── error_analysis_cot_v2.py         # CoT error analysis (manual annotation)
├── analyze_protocol_matrix.py       # Protocol-matrix cross-model analysis
├── analyze_results.py
├── check_table6.py                  # Hop-matched Cat.2 vs Cat.4 sanity check
├── clutrr_eval.py                   # CLUTRR baseline comparison harness
├── human_baseline_pilot.py          # Human baseline study tooling (n=280)
└── fix_cot_parser.py                # Answer-line parser fixes for CoT responses
```

## Quick start

```bash
# 1. Install dependencies
pip install llama-index-llms-ollama openai anthropic google-genai

# 2. Generate the benchmark from scratch (optional — datasets/ pre-built)
python kinshipqa_pipeline.py --all --output-dir ./my_datasets/

# 3. Evaluate a model (zero-shot direct)
python llm_tester.py --all --dataset-dir ./datasets_v6_2/ \
    --provider ollama --model gemma3:27b \
    --protocol zero_shot_direct \
    --output-dir ./results/

# 4. Multi-protocol evaluation
python llm_tester.py --all --dataset-dir ./datasets_v6_2/ \
    --provider ollama --model gemma3:27b \
    --protocol few_shot_cot --few-shot-file few_shot_examples.json \
    --output-dir ./results/
```

## Probes and ablations

| Probe | Script | What it measures |
|---|---|---|
| Cat.4 error taxonomy | `error_taxonomy_cat4.py` | 6-category breakdown of non-EM Cat.4 errors; identifies biological-default leakage |
| Paraphrase robustness | `paraphrase_robustness/sample_and_paraphrase.py` + `run_paraphrase_inference.py` | Cat.4 accuracy under rule-paraphrased questions (entity/system/kin-term preserved, frame varies) |
| Fictional-rule control | `make_fictional_cat4.py` + `compare_fictional_to_real.py` | +6.1% Cat.4 gain when both system label + kin terms swapped to invented strings |
| Orthogonal 2×2 decomp | `aggregate_fictional_2x2.py` | Decomposes +6.1% into system-name vs kin-term factors (interaction, not additive) |
| In-context rule probe | `llm_tester.py --with-rule-context` | +17.1 pp on skewing; −13.4 pp on high-baseline non-skewing |
| Multi-seed | `multi_seed_cat4.py` | T=0.7, n=5 stochastic decoding; preserves system-type ordering |
| Bootstrap CIs | `bootstrap_headline_cis.py` | 95% percentile CIs on headline aggregates |

## Citation

```bibtex
@inproceedings{kinshipqa2026,
  title={Kinship Data Benchmark for Multi-hop Reasoning},
  author={Anonymous Authors},
  booktitle={Submission under review},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).

## Data availability

Full per-question prediction logs (raw model responses, several MB per
(model, protocol, system) cell) are not included in this public repository
to keep clone size small. They are available on request and will be
deposited on Zenodo upon publication. The summary statistics
(`combined_results.json` in each `results_*/`) are sufficient to verify
all numbers reported in the paper.
