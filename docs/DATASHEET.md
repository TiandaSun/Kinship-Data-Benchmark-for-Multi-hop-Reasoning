# KinshipQA — Datasheet

Following the datasheet template of Gebru et al. (2021).

## 1. Motivation

**Q: For what purpose was the dataset created?**

KinshipQA was created to evaluate whether large language models can perform multi-hop kinship reasoning under different anthropological kinship systems. Most prior kinship reasoning benchmarks (e.g. CLUTRR, bAbI) implicitly assume the Western (Eskimo) system, where kin terms align with biological relations one-to-one. In non-Western systems (Hawaiian, Iroquois, Crow, Omaha, Dravidian, Sudanese), the same biological relation can map to different kin terms depending on the system's classification rules — for example, a "cousin" in Eskimo may be a "brother" in Hawaiian or a "father" in Crow. KinshipQA tests whether LLMs can apply these culturally-specific rules instead of defaulting to English biological terms.

**Q: Who created the dataset?**

The authors of the accompanying paper.

## 2. Composition

**Q: What do the instances represent?**

Each instance is a kinship reasoning question grounded in a procedurally-generated family tree. The question asks about a kinship relation between two named individuals (or, in Category 3, about the count of relations).

**Q: How many instances are there in total?**

| Split | Hops | Per system | Systems | Total |
|---|---|---|---|---|
| `kinshipqa/` (base) | 1–4 | 432–487 | 7 | ~3,150 |
| `kinshipqa_extended/` | 5–6 | ~varies | 7 | ~3,000 |

Exact counts are in each system's `*_dataset.summary.json`.

**Q: What does an instance look like?**

Each line of `data/kinshipqa/<system>_dataset.jsonl` is one question:

```json
{
  "question_id": "eskimo_0001",
  "question_text": "Who is Justin Williams's father?",
  "category": 1,
  "n_hops": 1,
  "kinship_system": "eskimo",
  "anchor_person": "person_30",
  "anchor_name": "Justin Williams",
  "target_persons": ["person_20"],
  "target_names": ["Larry Williams"],
  "ground_truth": "Larry Williams",
  "context": "Justin Williams's father is Larry Williams.",
  "path": ["hasFather"],
  "bio_term": "father",
  "kin_term": null,
  "has_cultural_override": false,
  "proof_graph_size": 2
}
```

Fields:

- `category` ∈ {1, 2, 3, 4} — question type:
  - 1: direct retrieval (1-hop)
  - 2: chain composition (2–4 hops)
  - 3: aggregation (e.g. "how many siblings does X have?")
  - 4: system-aware disambiguation (the kin term depends on the kinship system)
- `n_hops` — number of edges in the canonical reasoning path
- `kinship_system` — one of `{eskimo, sudanese, hawaiian, iroquois, dravidian, crow, omaha}`
- `anchor_*`, `target_*` — the persons referenced in the question
- `ground_truth` — the canonical answer (a name, a number, or a list)
- `context` — the supporting facts presented to the model
- `path` — the sequence of primitive predicates (e.g. `["hasFather", "hasMother"]`) that traces the reasoning chain
- `bio_term` — the biological kin term (English default), if applicable
- `kin_term` — the system-specific kin term, if it differs from `bio_term`
- `has_cultural_override` — true iff the system-specific term differs from the English default
- `proof_graph_size` — number of nodes + edges traversed during answer derivation

## 3. Collection process

**Q: How was the data collected?**

The dataset is **procedurally generated**, not collected from human sources. The pipeline is:

1. `kinship_tree_generator.py` — for each kinship system, run a population simulation (configurable start/end year, marriage rules, child-bearing rules per system) to build a family graph. Output: an OWL/RDF ontology in TTL form.
2. `kinshipqa_pipeline.py` — given the ontology, enumerate kinship paths of length 1–4 (and 5–6 for the extended split), instantiate each path as a question with category/anchor/target metadata, and emit JSONL.

All randomness is seeded; the released `data/kinshipqa/` was produced with the default seed.

**Q: Are persons in the dataset real?**

No. Every name in the dataset is sampled from common English first-name and surname lists; no individual is intended to refer to a real person.

## 4. Preprocessing / labelling

**Q: Was the data preprocessed?**

The released JSONL is the direct output of the generation pipeline. No filtering, no human curation, no label cleanup. The `ground_truth` field is computed deterministically from the underlying family graph.

## 5. Uses

**Q: Has the dataset been used for any tasks already?**

The accompanying paper uses it to evaluate seven LLMs (Gemma-3, DeepSeek-R1, Qwen-3, GPT-4o, GPT-4o-mini, Gemini, Claude Haiku 4.5) under three protocols (zero-shot direct, zero-shot CoT, few-shot CoT).

**Q: What are the intended uses?**

- LLM benchmarking on multi-hop reasoning
- Cross-cultural reasoning evaluation
- Probing whether LLMs default to English kin terms vs. apply culture-specific rules
- LLM+symbolic-reasoning hybrid evaluation (e.g. LLM+Prolog)

**Q: Are there tasks the dataset should not be used for?**

The dataset is **not** appropriate for:

- Real-world genealogical research
- Cultural anthropology research about specific living communities
- Any task requiring real-person grounding

## 6. Distribution

**Q: How will the dataset be distributed?**

This GitHub repository hosts the dataset alongside the generation code under an MIT license.

## 7. Maintenance

**Q: Who is supporting / hosting / maintaining the dataset?**

The paper authors. Issues and questions: please open a GitHub issue.

**Q: Is there an erratum?**

None at release time. Future corrections will be tracked via GitHub releases.
