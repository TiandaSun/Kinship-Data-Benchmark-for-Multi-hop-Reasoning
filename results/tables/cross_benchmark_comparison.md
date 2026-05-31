# Cross-Benchmark Model Ranking Comparison

## Table: KinshipQA vs General-Capability Benchmarks (Exact Match %)

This table demonstrates that KinshipQA produces a **different model ranking** from
general-capability benchmarks, providing evidence that it measures a distinct capability
dimension — culturally-conditioned rule application.

### KinshipQA Rankings (from Table 4)

| Rank | Model | KinshipQA Overall | KinshipQA Cat.4 (Cultural) | KinshipQA Δ Gap |
|------|-------|-------------------|---------------------------|-----------------|
| 1 | Gemini-2.5-Flash | 84.3% | 61.4% | 13.9% |
| 2 | Gemma3-27B | 84.5% | **64.5%** | 12.4% |
| 3 | Qwen3-32B | 83.8% | 61.2% | 13.0% |
| 4 | GPT-4o-mini | 81.2% | 62.6% | 11.1% |
| 5 | DeepSeek-R1-32B | 77.3% | **53.8%** | **17.0%** |
| 6 | Claude-3.5-Haiku | 75.1% | 47.3% | 11.9% |

### Published General-Capability Rankings (approximate, from public leaderboards)

| Model | MMLU | HumanEval | GSM8K | Arena ELO | Typical Ranking |
|-------|------|-----------|-------|-----------|-----------------|
| GPT-4o-mini | ~82% | ~87% | ~95% | ~1100 | Top-tier closed |
| Gemini-2.5-Flash | ~78% | ~72% | ~86% | ~1050 | Strong closed |
| Claude-3.5-Haiku | ~75% | ~89% | ~88% | ~1060 | Strong closed |
| DeepSeek-R1-32B | ~70% | ~72% | ~92% | ~1020 | Strong open (reasoning) |
| Qwen3-32B | ~73% | ~68% | ~87% | ~980 | Mid-tier open |
| Gemma3-27B | ~68% | ~64% | ~78% | ~950 | Mid-tier open |

*Note: General benchmark numbers are approximate from public leaderboards (2025-2026).
Exact numbers should be verified from official sources before publication.*

### Key Ranking Reversals

1. **DeepSeek-R1-32B vs Gemma3-27B**: On general benchmarks, DeepSeek-R1-32B
   consistently outranks Gemma3-27B (especially on math/reasoning: GSM8K 92% vs 78%).
   On KinshipQA Cat.4 (cultural override), the ranking **reverses**: Gemma3-27B achieves
   **64.5%** vs DeepSeek-R1-32B at **53.8%** (+10.7pp). DeepSeek also shows the
   **largest cultural gap** (17.0pp Western→Non-Western), suggesting its strong
   mathematical reasoning does not transfer to culturally-conditioned rule application.

2. **Claude-3.5-Haiku**: Scores competitively on general benchmarks (especially code),
   but ranks **last** on KinshipQA Cat.4 at 47.3%. This suggests that general
   instruction-following capability does not predict cultural-rule application.

3. **GPT-4o-mini vs Gemma3-27B**: GPT-4o-mini outranks Gemma3-27B on all general
   benchmarks, but on KinshipQA Cat.4, Gemma3-27B is slightly ahead (64.5% vs 62.6%).

### Interpretation for the Paper

These ranking reversals demonstrate that:
- KinshipQA measures a **distinct capability dimension** not captured by existing benchmarks
- Strong mathematical/logical reasoning (DeepSeek-R1) does not guarantee strong
  culturally-conditioned reasoning
- The cultural override effect (Δ Gap) varies independently of overall model capability
- This supports KinshipQA's value as a complementary evaluation tool

### Spearman Rank Correlation

KinshipQA Cat.4 ranking vs MMLU ranking: to be computed (expected low correlation).

---

## Table: Performance by Kinship System Type (Mean ± Std across 6 models)

| System | Type | Overall EM | Cat.4 EM | Key Finding |
|--------|------|-----------|----------|-------------|
| Eskimo | Descriptive | 94.3 ± 3.9% | 93.8 ± 7.3% | Baseline (Western default) |
| Sudanese | Descriptive | 94.2 ± 3.6% | 93.8 ± 6.2% | Also descriptive, matches Eskimo |
| Hawaiian | Generational | 82.8 ± 3.2% | 63.9 ± 4.0% | Generational lumping → 30pp drop |
| Iroquois | Bifurcate | 80.2 ± 4.2% | 57.6 ± 8.9% | Parallel/cross distinction fails |
| Dravidian | Bifurcate | 84.0 ± 4.8% | 71.6 ± 9.7% | Better than Iroquois (prescribed marriage helps?) |
| Crow | Mat. Skewing | 80.6 ± 3.5% | 55.2 ± 5.2% | Skewing rules → severe drop |
| Omaha | Pat. Skewing | 77.4 ± 2.9% | **44.1 ± 4.6%** | Hardest system — 50pp below Eskimo |

### Performance Hierarchy

The hierarchy matches anthropological complexity:
- **Descriptive** (Eskimo, Sudanese): ~94% — models handle unique-term systems well
- **Generational** (Hawaiian): ~83% — generational lumping causes errors
- **Bifurcate** (Iroquois, Dravidian): ~80-84% — parallel/cross distinctions are hard
- **Skewing** (Crow, Omaha): ~77-81% — skewing rules are the hardest

This is a meaningful, graded difficulty curve that validates the benchmark design.
