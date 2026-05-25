#!/usr/bin/env python3
"""
CLUTRR Evaluation Script
=========================
Evaluates LLMs on the CLUTRR kinship reasoning benchmark for cross-benchmark
comparison with KinshipQA. Uses the same provider infrastructure as llm_tester_v6.py.

CLUTRR tests biological kinship chain reasoning (English-only, no cultural rules).
Comparison with KinshipQA isolates cultural rule-application as the distinguishing variable.

Usage:
    python clutrr_eval.py --provider ollama --model gemma3:27b --output results_clutrr_gemma3_27b.json
    python clutrr_eval.py --provider openai --model gpt-4o-mini --output results_clutrr_gpt4o_mini.json

Author: Tianda (EMNLP 2026)
"""

import json
import time
import argparse
import re
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# Import providers from llm_tester_v6
import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_tester_v6 import (
    TestConfig, create_provider, OllamaProvider,
    OpenAIProvider, GeminiProvider, AnthropicProvider
)


# Standard kinship terms for normalization
KINSHIP_TERMS = {
    "grandfather", "grandmother", "grandson", "granddaughter",
    "father", "mother", "son", "daughter",
    "brother", "sister", "uncle", "aunt",
    "nephew", "niece", "cousin", "husband", "wife",
    "father-in-law", "mother-in-law", "son-in-law", "daughter-in-law",
    "brother-in-law", "sister-in-law",
    "great-grandfather", "great-grandmother", "great-grandson", "great-granddaughter",
    "great-uncle", "great-aunt", "great-nephew", "great-niece",
    "granduncle", "grandaunt",
}


def create_clutrr_prompt(story: str, query: tuple) -> str:
    """Create a zero-shot prompt for CLUTRR."""
    person_a, person_b = query
    return f"""Read the following story about a family and answer the question.
Be concise and provide only the kinship term without explanation.

Story: {story}

Question: What is the relationship of {person_b} to {person_a}?
(For example: father, mother, grandmother, uncle, cousin, etc.)

Answer:"""


def normalize_kinship_term(response: str) -> str:
    """Extract and normalize kinship term from model response."""
    if not response:
        return ""
    response = response.strip().lower()
    response = re.sub(r'[^\w\s-]', '', response)

    # Direct match
    for term in KINSHIP_TERMS:
        if term in response:
            return term

    # Common aliases
    aliases = {
        "grandpa": "grandfather", "grandma": "grandmother",
        "mom": "mother", "dad": "father",
        "bro": "brother", "sis": "sister",
        "grand father": "grandfather", "grand mother": "grandmother",
        "grand son": "grandson", "grand daughter": "granddaughter",
    }
    for alias, canonical in aliases.items():
        if alias in response:
            return canonical

    # Last resort: return first word
    words = response.split()
    return words[0] if words else ""


def evaluate_clutrr(provider, config, max_examples=None, verbose=False):
    """Run CLUTRR evaluation."""
    ds = load_dataset('CLUTRR/v1', 'gen_train234_test2to10', split='test')

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    results_by_hop = defaultdict(lambda: {"total": 0, "correct": 0})
    all_results = []
    total_correct = 0
    total = 0
    errors = 0

    rate_limit = provider.get_rate_limit_delay()

    for i, example in enumerate(ds):
        story = example['clean_story']
        query = eval(example['query']) if isinstance(example['query'], str) else example['query']
        target = example['target_text'].lower().strip()
        n_hops = len(example['edge_types'].split(',')) if isinstance(example['edge_types'], str) else len(example['edge_types'])

        prompt = create_clutrr_prompt(story, query)

        start_time = time.time()
        is_error = False
        try:
            response = provider.generate(prompt)
            if response is None or response.startswith("[ERROR"):
                is_error = True
                errors += 1
        except Exception as e:
            response = f"[ERROR: {str(e)[:100]}]"
            is_error = True
            errors += 1

        elapsed = time.time() - start_time

        predicted = normalize_kinship_term(response) if not is_error else ""
        correct = predicted == target

        total += 1
        if correct:
            total_correct += 1
        results_by_hop[n_hops]["total"] += 1
        if correct:
            results_by_hop[n_hops]["correct"] += 1

        result = {
            "id": example['id'],
            "story": story[:200],
            "query": str(query),
            "target": target,
            "predicted": predicted,
            "correct": correct,
            "n_hops": n_hops,
            "response_time": elapsed,
            "raw_response": str(response)[:300],
            "is_error": is_error,
        }
        all_results.append(result)

        if verbose:
            status = "✓" if correct else ("⚠" if is_error else "✗")
            print(f"  [{i+1}/{len(ds)}] {status} hops={n_hops} GT={target} Pred={predicted}")

        if rate_limit > 0:
            time.sleep(rate_limit)

    # Summary
    accuracy = total_correct / total if total else 0
    hop_summary = {}
    for hop in sorted(results_by_hop.keys()):
        h = results_by_hop[hop]
        hop_summary[f"hop_{hop}"] = {
            "total": h["total"],
            "correct": h["correct"],
            "accuracy": h["correct"] / h["total"] if h["total"] else 0
        }

    summary = {
        "benchmark": "CLUTRR",
        "config": "gen_train234_test2to10",
        "model": provider.get_model_info(),
        "total": total,
        "correct": total_correct,
        "accuracy": accuracy,
        "errors": errors,
        "by_hops": hop_summary,
    }

    return summary, all_results


def main():
    parser = argparse.ArgumentParser(description="CLUTRR Evaluation")
    parser.add_argument("--provider", type=str, default="ollama",
                        choices=["ollama", "openai", "gemini", "anthropic"])
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--output", type=str, default="results_clutrr.json")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = TestConfig(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
    )

    print(f"Initializing {config.provider} / {config.model}...")
    provider = create_provider(config)

    print(f"Running CLUTRR evaluation...")
    summary, results = evaluate_clutrr(
        provider, config,
        max_examples=args.limit,
        verbose=args.verbose
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"CLUTRR Results: {summary['model']}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {summary['accuracy']*100:.1f}% ({summary['correct']}/{summary['total']})")
    print(f"Errors: {summary['errors']}")
    print(f"\nBy Hop Depth:")
    print(f"  {'Hops':<8} {'Total':>6} {'Correct':>8} {'Acc':>8}")
    print(f"  {'-'*32}")
    for hop_key, hop_data in sorted(summary['by_hops'].items()):
        print(f"  {hop_key:<8} {hop_data['total']:>6} {hop_data['correct']:>8} {hop_data['accuracy']*100:>7.1f}%")

    # Save
    output = {"summary": summary, "questions": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
