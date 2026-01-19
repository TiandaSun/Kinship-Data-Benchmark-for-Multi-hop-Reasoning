# KinshipQA

A benchmark for evaluating multi-hop reasoning in Large Language Models using culturally diverse kinship systems.

## Overview

KinshipQA tests whether LLMs can perform multi-hop reasoning over kinship relations while accounting for cultural variation. The benchmark covers **7 anthropologically-documented kinship systems** with distinct classification rules:

| System | Type | Key Feature |
|--------|------|-------------|
| Eskimo | Descriptive | Western nuclear-family focus |
| Sudanese | Descriptive | Unique term for each relative |
| Hawaiian | Generational | All same-generation relatives merged |
| Iroquois | Bifurcate | Parallel/cross cousin distinction |
| Dravidian | Bifurcate | Cross-cousins as potential spouses |
| Crow | Mat. Skewing | Father's sister's line skewed upward |
| Omaha | Pat. Skewing | Mother's brother's line skewed upward |

### Key Findings

- **14.1% accuracy gap** between Western (Eskimo/Sudanese) and non-Western systems
- **23.6% drop** when cultural rules override biological intuitions
- Performance degrades at **3-hop complexity** where cultural rules most frequently apply

## Repository Structure

```
kinshipqa/
├── ontologies/          # RDF/OWL ontologies for 7 kinship systems
│   ├── eskimo.ttl
│   ├── hawaiian.ttl
│   └── ...
├── datasets/            # Pre-generated QA datasets (JSONL)
│   ├── eskimo_dataset.jsonl
│   ├── hawaiian_dataset.jsonl
│   └── ...
├── kinship_tree_generator.py   # Generate family trees → RDF ontologies
├── kinshipqa_pipeline.py       # Generate QA datasets from ontologies
└── llm_tester.py               # Evaluate LLMs on datasets
```

## Installation

```bash
pip install rdflib llama-index-llms-ollama openai anthropic google-genai
```

## Quick Start

### 1. Generate Ontologies (optional - pre-generated in `ontologies/`)

```bash
# Generate all 7 kinship systems
python kinship_tree_generator.py --all --output-dir ./ontologies/

# Generate single system
python kinship_tree_generator.py --system dravidian --output dravidian.ttl
```

### 2. Generate QA Dataset (optional - pre-generated in `datasets/`)

```bash
# Generate all datasets
python kinshipqa_pipeline.py --all --ttl-dir ./ontologies --output-dir ./datasets

# Generate single dataset
python kinshipqa_pipeline.py --ttl ./ontologies/hawaiian.ttl --system hawaiian
```

### 3. Evaluate LLMs

```bash
# Test all systems with a model
python llm_tester.py --all --dataset-dir ./datasets/ --provider ollama --model qwen3:32b

# Test single system
python llm_tester.py ./datasets/crow_dataset.jsonl --provider openai --model gpt-4o-mini

# Supported providers: ollama, openai, anthropic, gemini
```

## Dataset Format

Each question in the JSONL files contains:

```json
{
  "question_id": "hawaiian_0001",
  "question_text": "Who is John's classificatory father?",
  "category": 4,
  "n_hops": 2,
  "kinship_system": "hawaiian",
  "context": "John's father is Robert. Robert's brother is Larry.",
  "ground_truth": ["Larry"],
  "has_cultural_override": true
}
```

### Question Categories

| Category | Description | Hops |
|----------|-------------|------|
| 1 | Fact Retrieval | 1 |
| 2 | Multi-hop Biological | 2-4 |
| 3 | Counting/Filtering | 1-2 |
| 4 | Cultural Disambiguation | 2-4 |

## Citation

```bibtex
@inproceedings{sun2026kinshipqa,
  title={Kinship Data Benchmark for Multi-hop Reasoning},
  author={Sun, Tianda and Kazakov, Dimitar},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- Tianda Sun - tianda.sun@york.ac.uk
- Dimitar Kazakov - dimitar.kazakov@york.ac.uk

University of York, Department of Computer Science
