# pipeline/

Dataset generation and LLM evaluation code.

| File | Purpose |
|---|---|
| `kinship_tree_generator.py` | Simulate a population (births, marriages, generations) under each of seven kinship systems and emit an RDF/OWL ontology in TTL form. |
| `kinshipqa_pipeline.py` | Given an ontology, enumerate kinship paths and instantiate questions across the four categories. Emits JSONL datasets. |
| `llm_tester.py` | Unified evaluator. Four providers (`ollama`, `openai`, `gemini`, `anthropic`), three protocols (`zero_shot_direct`, `zero_shot_cot`, `few_shot_cot`), and `--with-rule-context` for the in-context rule probe. |
| `few_shot_examples.json` | Worked examples used by the `few_shot_cot` protocol. |
| `make_fictional_cat4.py` | Build fictional-rule variants of Cat. 4 (modes: `full` / `system_only` / `term_only`). |
| `augment_cat4_4hop.py` | Add extra 4-hop Cat. 4 paths to the base dataset. |
| `augment_cat4_5_6hop.py` | Add extra 5–6 hop Cat. 4 paths (Other-5 systems). |

See [`../docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) for end-to-end commands.

All scripts accept `--help` for full CLI options.
