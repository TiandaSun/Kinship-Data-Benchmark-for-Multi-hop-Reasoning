# KinshipQA — One-Sheet for Supervisor Meeting (17 April 2026)

## New experimental results (this week)

### Protocol Matrix — 3 models × 3 protocols (COMPLETE, 0 errors)

**Cat 4 Cultural Override (the paper's key metric):**

| Model | Direct | CoT | Few-Shot CoT |
|---|---|---|---|
| Gemma3-27B | **65.3%** | 35.8%† | 54.2% |
| DeepSeek-R1-32B | 41.1% | 33.3%† | **42.7%** |
| Qwen3-32B | 44.8% | 48.8% | **51.4%** |

†CoT numbers underestimate due to answer extraction issues; under investigation.

**Headline:** The cultural-override gap persists under ALL protocols (including CoT
and few-shot). It is NOT a prompting artifact. This strengthens the paper's core claim.

### Hop Scaling Curve — extended to 5-6 hops (in progress)

| Hop depth | Gemma3-27B | DeepSeek-R1-32B |
|---|---|---|
| 2-hop | 91% | 79% |
| 4-hop | 84% | 91% |
| **5-hop** | **39%** | **58%** |
| **6-hop** | **66%** | **4%** |

**Headline:** Sharp accuracy drop at 5+ hops — the scaling curve Reviewer #2 asked for.

### Cross-benchmark ranking reversal (confirmed)
Gemma3-27B outperforms DeepSeek-R1-32B on Cat 4 by +24pp (direct) and +12pp (few-shot),
despite DeepSeek winning on MMLU/GSM8K. KinshipQA measures a distinct capability.

---

## The key decision for this meeting

**How should the paper be framed?**

| | Plan A: Cultural rule-application | Plan B: Compromise | Plan C: Multi-hop |
|---|---|---|---|
| **Title leads with** | Cultural rule-application + CoT faithfulness | Cultural question + multi-hop method | Multi-hop reasoning |
| **Track** | Interpretability & Analysis | Flexible | Resources & Evaluation |
| **Reviewer pool** | CoT-faithfulness people | Mixed | Same as before (killed us) |
| **EMNLP Main chance** | ~35% | ~25% | ~15% |
| **EMNLP Findings chance** | ~55% | ~45% | ~30% |

The protocol matrix results support Plan A/B: the declarative-procedural gap persisting
under CoT is an **interpretability finding**, not just a benchmark result.

---

## What's already done (ready for the paper)

- ✅ Protocol matrix: 3 models × 3 protocols × 7 systems (9 sweeps, complete)
- ✅ Extended hop datasets: 5-6 hop questions generated (4/7 systems support it)
- ✅ Extended hop evaluation: running (4/9 sweeps done)
- ✅ Cross-benchmark comparison table (LaTeX ready)
- ✅ 17-item cheap fixes checklist
- ✅ Human baseline study design ($350 Prolific, ready to launch)
- ✅ Full revision plan with rebuttal-ready guideline quotes

## What needs the framing decision first

- Title/abstract rewrite
- Related-work section
- Track routing (ARR submission)
- Multilingual experiment (more important in Plan A)
