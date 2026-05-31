# data/

The released KinshipQA datasets, the RDF/OWL ontologies they are generated from, and dataset variants used in the paper's probes and ablations.

## Datasets

| Directory | Description | Question count |
|---|---|---|
| `kinshipqa/` | **Main 1–4 hop benchmark**, all 7 systems, includes the 4-hop Cat. 4 path augmentation. | 3,354 |
| `kinshipqa_extended_full/` | Full 1–6 hop extension, all 7 systems (basic — no Cat. 4 augmentation at 5–6 hops). | varies |
| `kinshipqa_extended_5_6hop_aug/` | 5–6 hop Cat. 4 augmentation on the Other-5 systems (the cultural-rule depth probe). | 249 |
| `kinshipqa_200yr/` | Variant generated from the 200-year population simulation. Used for the extended-hop runs to provide a deeper kinship structure. | varies |

## Fictional-rule variants

For the fictional-rule control ablation (3.7% of Cat. 4 vocabulary swapped with invented strings):

| Directory | Substitution |
|---|---|
| `fictional/full/` | Both system name AND kin terms swapped to fictional strings. |
| `fictional/system_only/` | Only the system name swapped (orthogonal decomposition). |
| `fictional/term_only/` | Only the kin terms swapped (orthogonal decomposition). |

The 2×2 decomposition is reported in `../results/orthogonal_fictional_ablation.json`.

## Ontologies

| Directory | Population span | Used by |
|---|---|---|
| `ontologies/` | 50 years | `kinshipqa/` |
| `ontologies_extended/` | 150 years | `kinshipqa_extended_full/` for 5–6 hop Cat. 2 |
| `ontologies_200yr/` | 200 years | `kinshipqa_200yr/` + 5–6 hop Cat. 4 runs |

## Probes

| Directory | Description |
|---|---|
| `paraphrase_robustness/` | Cat. 4 question-phrasing robustness probe — measures how much accuracy changes when the question frame is paraphrased while entity, system, and kin term are held constant. Contains both the paraphrase set and the inference scripts. |

## Question schema

Each line of `<dataset>/<system>_dataset.jsonl` is one question. See [`../docs/DATASHEET.md`](../docs/DATASHEET.md) for the full per-field schema.

## Per-system file naming

```
<system>_dataset.jsonl          # one question per line
<system>_dataset.summary.json   # per-category and per-hop counts
```

Where `<system>` ∈ `{eskimo, sudanese, hawaiian, iroquois, dravidian, crow, omaha}`.

## Regenerating

All datasets are reproducible from the ontologies using `pipeline/kinshipqa_pipeline.py` and the augmentation scripts. See [`../docs/REPRODUCTION.md`](../docs/REPRODUCTION.md).
